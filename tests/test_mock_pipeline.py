"""全链路测试：入库 → 检索 → 生成 → 引用校验，桩模式与真实模式跑同一段代码。

两条纪律：

1. **用真 PDF 走真 loader / chunker / 入库**，不拿假 Chunk 对象凑数。假对象会让
   「文档名、页码、向量列类型、去重」这些真正容易出错的地方全部失去覆盖。
2. **断言与模型无关**。桩模式与真实模式共用同一组断言：引用只能来自本次检索到的
   片段、空语料必须拒答、向量以 float32 字节入库。这才是「管道与模型解耦」的证据。
"""
from __future__ import annotations

import fitz
import numpy as np
import pytest
from sqlalchemy import delete, select

from app.config import settings
from app.db import SessionLocal, init_db
from app.ingest import pipeline as ingest_pipeline
from app.llm import embed, pipeline
from app.models import Chunk, Document, EmbeddingCache, QaLog
from app.retrieval import rebuild_indexes, search_with_ranks
from app.retrieval.vector import reset_index

DOC_NAME = "CP600空压机一体机手册.pdf"

# 两页：第一页讲接线选型，第二页讲上电无显示的排查
PAGES = [
    ["4 接线", "4.4 主功率线缆", "4.4.1 选型", "按表4-2 选择线径，4T22G 机型推荐 10mm² 铜芯线。"],
    ["6 故障排查", "6.2 上电无显示", "检查主保险丝与显示排线，若 Err23 持续报出则更换控制板。"],
]


def _write_pdf(path) -> None:  # noqa: ANN001 - pathlib.Path
    """现场生成一份真实 PDF；用 PyMuPDF 内置中文字体，避免依赖外部字体文件。"""
    document = fitz.open()
    for lines in PAGES:
        page = document.new_page()
        y = 72.0
        for line in lines:
            page.insert_text((72, y), line, fontsize=12, fontname="china-s")
            y += 20.0
    document.save(str(path))
    document.close()


def _build_corpus(tmp_path) -> "object":  # noqa: ANN001 - pathlib.Path
    """清库 → 生成 PDF → 入库 → 建向量与 BM25 索引，返回文档路径。"""
    init_db()
    with SessionLocal() as session:
        # 同一轮测试共用一个库文件，先清掉上一组用例的数据
        session.execute(delete(Chunk))
        session.execute(delete(Document))
        session.execute(delete(QaLog))
        session.execute(delete(EmbeddingCache))
        session.commit()

        path = tmp_path / DOC_NAME
        _write_pdf(path)
        result = ingest_pipeline.ingest_file(session, path, tmp_path)
        assert result.status == "indexed", f"入库失败：{result.reason}"
        assert result.pages == 2

        stats = rebuild_indexes(session)
        assert stats["embedded"] > 0
    reset_index()
    return path


@pytest.fixture()
def corpus(tmp_path):
    """用当前配置（真实模型或桩）建语料；存储与检索类测试用它。"""
    return _build_corpus(tmp_path)


@pytest.fixture()
def mock_corpus(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """**先**把 MOCK_MODE 打开再建语料。

    顺序很关键：向量与查询必须由同一个模型产生。如果先按真实模型建索引、再在测试体里
    切到桩模式，查询向量会是 512 维而库内是 1024 维，`vector._build` 会把所有块当成
    「模型不一致」跳过，检索直接空——那测的就不是引用规则了。
    """
    monkeypatch.setattr(settings, "mock_mode", True)
    return _build_corpus(tmp_path)


def test_vectors_are_stored_as_float32_blob(corpus) -> None:
    """向量必须是原始字节：JSON 文本会慢一个数量级，pgvector 类型则绑死 Postgres。"""
    with SessionLocal() as session:
        blob, model = session.execute(select(Chunk.embedding, Chunk.embed_model).limit(1)).one()

    assert isinstance(blob, bytes)
    assert len(blob) % 4 == 0
    assert len(blob) // 4 == settings.embed_dim
    assert model == embed.model_name()


def test_search_reaches_the_right_page(corpus) -> None:
    """检索要能落到具体页码——引用能核对的前提是页码正确。"""
    hits = search_with_ranks("4T22G 主功率线缆选多粗", top_k=3, mode="hybrid")

    assert hits, "索引建好了却检索不到任何块"
    assert hits[0].chunk.document.filename == DOC_NAME
    # BM25 一定会把第一页排在最前；桩向量也让第一页靠前，用前两名兜住实现细节
    assert 1 in [hit.chunk.page_no for hit in hits[:2]]


def test_answer_cites_only_retrieved_passages(mock_corpus) -> None:
    """回答里的每个来源都必须能在本次检索到的片段里找到；这就是 P3 的核心验收。

    用桩模型跑，是为了让「作答」这件事确定发生——真实模型可能合理地拒答，
    那种情况由下面的真实模型用例按「作答或拒答，但绝不编造」来判定。
    """
    result = pipeline.answer_once("4T22G 主功率线缆选多粗", "hybrid", 3)

    assert result["answer_status"] == "answered"
    check = result["citation_check"]
    assert check["unique"] >= 1, "作答却没有引用，无法溯源"
    assert check["invalid"] == [], f"出现了不存在的来源：{check['invalid']}"
    assert check["accuracy"] == 1.0
    for citation in result["citations"]:
        assert citation["doc"] == DOC_NAME
        assert citation["page"] in (1, 2)
        assert citation["snippet"], "引用列表必须带块内容摘要"
    assert "结论" in result["answer"] and "来源" in result["answer"]
    assert "MOCK" in result["answer"], "桩模型的输出必须自我标记，不能冒充真实模型"


def test_qalog_records_usage_and_timings(mock_corpus) -> None:
    """每次调用都要留下用量与两段耗时，否则没法核算成本、也分不清是谁慢。"""
    pipeline.answer_once("上电无显示怎么排查", "hybrid", 2)

    with SessionLocal() as session:
        row = session.execute(select(QaLog).order_by(QaLog.id.desc()).limit(1)).scalar_one()

    assert row.answer_status == "answered"
    assert row.retrieval_ms is not None and row.retrieval_ms >= 0
    assert row.generation_ms is not None and row.generation_ms >= 0
    assert row.total_tokens and row.total_tokens > 0
    assert row.usage_estimated is True
    assert row.llm_model and "mock" in row.llm_model
    assert row.retrieved_chunk_ids


def test_empty_corpus_must_refuse(mock_corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """检索不到片段时必须拒答并说明缺什么，不许凭空作答。"""
    monkeypatch.setattr(pipeline, "search_with_ranks", lambda *args, **kwargs: [])

    result = pipeline.answer_once("资料里没有的东西是什么", "hybrid", 3)

    assert result["answer_status"] == "refused"
    assert "资料中未找到相关内容" in result["answer"]
    assert "缺少的信息" in result["answer"]
    assert result["citation_check"]["unique"] == 0


def test_second_identical_text_hits_local_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一段文本第二次编码不许再发网络请求——这是 embedding_cache 存在的唯一理由。"""
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.setattr(settings, "embed_base_url", "https://example.invalid/v1")
    monkeypatch.setattr(settings, "embed_api_key", "test-key")
    monkeypatch.setattr(settings, "embed_model", "cache-test-model")
    init_db()

    calls: list[list[str]] = []

    def fake_post(batch: list[str]) -> np.ndarray:
        calls.append(list(batch))
        eyes = np.eye(settings.embed_dim, dtype=np.float32)
        return eyes[: len(batch)]

    monkeypatch.setattr(embed, "_post_batch", fake_post)

    first = embed.encode_documents(["油压低于 0.15MPa 时检查油滤"])
    second = embed.encode_documents(["油压低于 0.15MPa 时检查油滤"])

    assert len(calls) == 1, f"第二次仍然发了请求：{len(calls)} 次"
    assert np.allclose(first, second)


def test_cache_write_handles_duplicates_and_large_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """150 行、含大量重复文本的批量写缓存不能崩。

    真实语料里重复段落（页眉残留、同一张表出现在两章）很常见，同一批出现两个相同 key
    会撞 UNIQUE 约束，整批缓存白写；203 行的多值 INSERT 还会超过 SQLite 变量数上限。
    """
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.setattr(settings, "embed_model", "cache-batch-model")
    init_db()
    with SessionLocal() as session:
        session.execute(delete(EmbeddingCache))
        session.commit()

    texts = [f"重复段落 {index % 120}" for index in range(150)]
    vector = np.linspace(0.1, 0.2, settings.embed_dim, dtype=np.float32)
    items = [(embed._cache_key(text), text, vector) for text in texts]

    embed._write_cache(items)

    assert embed.cache_size() == 120, "去重后应只写 120 行"


@pytest.mark.skipif(
    settings.mock_mode,
    reason="未配置真实模型接口（EMBED_* / LLM_*）；桩模式已覆盖管道逻辑，真实模式需 .env",
)
def test_same_assertions_pass_with_real_models(corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    """配好 .env 后跑同一段链路：检索、引用、拒答三条规矩都不变。

    这里**不**断言「一定作答」：真实模型面对两页合成语料时可能合理地拒答，
    那不是 bug。要守住的契约是二者之一——要么作答且引用全部真实，
    要么拒答并说明缺什么，绝不能编。
    """
    monkeypatch.setattr(settings, "mock_mode", False)

    hits = search_with_ranks("4T22G 主功率线缆选多粗", top_k=3, mode="hybrid")
    assert hits and hits[0].chunk.document.filename == DOC_NAME

    result = pipeline.answer_once("4T22G 主功率线缆选多粗", "hybrid", 3)
    check = result["citation_check"]

    assert "MOCK" not in result["answer"]
    assert check["invalid"] == [], f"真实模型编造了来源：{check['invalid']}"
    if result["answer_status"] == "answered":
        assert check["unique"] >= 1 and check["accuracy"] == 1.0
    else:
        assert result["answer_status"] == "refused"
        assert "资料中未找到相关内容" in result["answer"]
        assert "缺少的信息" in result["answer"]
        assert check["unique"] == 0
