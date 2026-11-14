"""生成层单元测试：提示词约束、引用校验、错误结构化、SSE 事件顺序。

这里既不下载模型也不连数据库：真实模型与真实检索的验证放在
`tests/check_citations.py`，pytest 只测「不依赖外部服务也一定会错」的那部分逻辑。
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api.chat import router
from app.config import settings
from app.db import Base
from app.llm import citations, client, pipeline, prompt
from app.llm.citations import REFUSAL_MARKER, Citation
from app.models import QaLog

DOC = "CP600空压机一体机手册.pdf"


BODY = "4.4.1 主功率线缆\n按表4-2 选择线径，4T22G 机型推荐 10mm²。"


def _citation(page: int | None = 24, doc: str = DOC, chunk_id: int = 7) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        doc=doc,
        page=page,
        section="4 接线 > 4.4 主功率线缆",
        content=BODY,
        snippet=BODY,
    )


def _plan(question: str = "4T22G 主功率线缆选多粗", pages: list[int] | None = None) -> pipeline.AnswerPlan:
    passages = [_citation(page=page, chunk_id=100 + index) for index, page in enumerate(pages or [24])]
    return pipeline.AnswerPlan(
        question=question,
        mode="hybrid",
        top_k=len(passages),
        retrieval_ms=12.5,
        citations=passages,
        allowed=citations.allowed_keys(passages),
        messages=prompt.build_messages(question, passages),
        chunk_ids=[item.chunk_id for item in passages],
    )


# ---------------- 提示词 ----------------


def test_system_prompt_carries_every_hard_constraint() -> None:
    """约束必须写在提示词里，否则模型没有任何理由拒答或标来源。"""
    text = prompt.SYSTEM_PROMPT

    assert "只依据 <片段>" in text
    assert "片段以外的知识一律不许使用" in text
    assert "[文档名 p.页码]" in text
    assert "原样复制" in text
    assert REFUSAL_MARKER in text
    assert "缺少的信息" in text
    assert "互相矛盾" in text
    for section in ("结论", "排查步骤", "来源"):
        assert section in text


def test_system_prompt_has_no_persona_fluff() -> None:
    """人设空话会把模型推向「敢编」，这里不允许出现（「专家」只作为禁止词出现）。"""
    text = prompt.SYSTEM_PROMPT
    for word in ("你是", "资深", "经验丰富", "作为一名"):
        assert word not in text
    assert "不写「作为专家」" in text


def test_messages_are_single_turn_with_static_system() -> None:
    """单轮：只有 system + 一条 user；系统提示词不掺入片段。"""
    messages = prompt.build_messages("上电无显示怎么排查", [_citation()])

    assert [item["role"] for item in messages] == ["system", "user"]
    assert messages[0]["content"] == prompt.SYSTEM_PROMPT
    assert "<cite>[CP600空压机一体机手册.pdf p.24]</cite>" in messages[1]["content"]
    assert "上电无显示怎么排查" in messages[1]["content"]


def test_passage_without_page_has_no_invented_page_number() -> None:
    """DOCX 没有页码，引用标签里不能凭空补一个页码。"""
    passage = _citation(page=None)
    text = prompt.build_passages([passage])

    assert "<cite>[CP600空压机一体机手册.pdf]</cite>" in text
    assert passage.label == f"[{DOC}]"


def test_empty_passages_are_explicit() -> None:
    """没有检索结果时要明说，不能让模型面对空白上下文自由发挥。"""
    assert "没有返回任何片段" in prompt.build_passages([])


def test_passage_clip_keeps_head_and_marks_truncation() -> None:
    """超长表格块按字符截断，并留下可识别的截断标记。"""
    passage = Citation(chunk_id=1, doc=DOC, page=3, section=None, content="A" * 50, snippet="A" * 50)
    text = prompt.build_passages([passage], clip=10)

    assert "A" * 10 in text
    assert "已截断" in text
    assert "A" * 11 not in text


# ---------------- 引用校验 ----------------


def test_valid_citation_passes() -> None:
    """文档名与页码都对得上，就是一条有效引用。"""
    check = citations.validate_answer(
        f"结论：选 10mm² [{DOC} p.24]", citations.allowed_keys([_citation()])
    )

    assert check.occurrences == 1
    assert check.unique == 1
    assert check.invalid == []
    assert check.accuracy == 1.0


def test_wrong_page_is_flagged() -> None:
    """页码对不上就是幻觉，即使文档名是对的。"""
    check = citations.validate_answer(
        f"结论：选 10mm² [{DOC} p.99]", citations.allowed_keys([_citation()])
    )

    assert check.accuracy == 0.0
    assert "99" in check.invalid[0].reason


def test_unknown_document_is_flagged() -> None:
    check = citations.validate_answer(
        "结论：见 [CP900用户手册.pdf p.1]", citations.allowed_keys([_citation()])
    )

    assert len(check.invalid) == 1
    assert "不在本次检索结果中" in check.invalid[0].reason


def test_page_range_is_treated_as_invalid_not_ignored() -> None:
    """畸形写法不能被「没识别出来」掩盖，否则准确率会虚高。"""
    check = citations.validate_answer(
        f"结论：见 [{DOC} p.24-25]", citations.allowed_keys([_citation()])
    )

    assert check.occurrences == 1
    assert check.accuracy == 0.0


def test_duplicate_citations_counted_once() -> None:
    """同一来源重复出现只算一次，避免用重复引用刷准确率。"""
    allowed = citations.allowed_keys([_citation(), _citation(page=25, chunk_id=8)])
    answer = f"结论 [{DOC} p.24] 步骤 [{DOC} p.24] 来源 [{DOC} p.25]"
    check = citations.validate_answer(answer, allowed)

    assert check.occurrences == 3
    assert check.unique == 2
    assert check.accuracy == 1.0


def test_no_citation_gives_no_accuracy() -> None:
    """拒答本来就没有引用，此时准确率应为 None 而不是 0 或 100%。"""
    check = citations.validate_answer(f"{REFUSAL_MARKER}\n缺少的信息：液压参数", set())

    assert check.unique == 0
    assert check.accuracy is None
    assert citations.is_refusal(f"{REFUSAL_MARKER}\n缺少的信息：液压参数")


def test_filename_variants_are_tolerated() -> None:
    """模型把 `.pdf` 写丢或加空格不算幻觉，但页码照旧严格核对。"""
    allowed = citations.allowed_keys([_citation()])
    answer = "结论：见 [CP600 空压机一体机手册 p.24]"
    check = citations.validate_answer(answer, allowed)

    assert check.invalid == []


def test_document_only_citation_without_page_is_invalid_for_pdf() -> None:
    """PDF 片段有页码却只写文档名 → 无法定位，算无效引用。"""
    check = citations.validate_answer(
        f"结论：见 [{DOC}]", citations.allowed_keys([_citation()])
    )

    assert check.occurrences == 1
    assert check.accuracy == 0.0
    assert "缺少页码" in check.invalid[0].reason


def test_document_only_citation_is_valid_when_passage_has_no_page() -> None:
    """DOCX 片段本来就没有页码，此时只写文档名是合法引用。"""
    passage = _citation(page=None, doc="维保规范.docx")
    check = citations.validate_answer(
        "结论：见 [维保规范.docx]", citations.allowed_keys([passage])
    )

    assert check.invalid == []
    assert check.accuracy == 1.0


def test_unknown_filename_without_page_still_counts() -> None:
    """编一个不带页码的文件名也算幻觉，不能被「没有页码」放过。"""
    check = citations.validate_answer(
        "结论：见 [完全不存在的手册.pdf]", citations.allowed_keys([_citation()])
    )

    assert check.occurrences == 1
    assert check.accuracy == 0.0


def test_non_citation_brackets_are_ignored() -> None:
    """正文里的 `[片段1]`、`[图4-2]` 不是引用，不能算进分母。"""
    check = citations.validate_answer(
        f"结论：见 [片段 1] 与 [图4-2] [{DOC} p.24]", citations.allowed_keys([_citation()])
    )

    assert check.occurrences == 1
    assert check.accuracy == 1.0


# ---------------- 客户端 ----------------


def test_endpoint_accepts_base_or_full_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_url 给到 /v1 或直接给到 /chat/completions 都要能拼对。"""
    monkeypatch.setattr(settings, "llm_base_url", "https://api.example.com/v1/")
    assert client._endpoint() == "https://api.example.com/v1/chat/completions"

    monkeypatch.setattr(settings, "llm_base_url", "https://api.example.com/v1/chat/completions")
    assert client._endpoint() == "https://api.example.com/v1/chat/completions"


def test_missing_config_yields_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """未配置且没开 MOCK_MODE 时返回 config_missing 事件，不抛异常、不发请求。"""
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.setattr(settings, "llm_base_url", "")
    events = list(client.stream_chat([{"role": "user", "content": "你好"}]))

    assert [event.kind for event in events] == ["error"]
    assert events[0].error is not None
    assert events[0].error.code == "config_missing"
    assert "LLM_BASE_URL" in events[0].error.message


def test_http_status_maps_retryable() -> None:
    """429/5xx 可重试，400/401 不可重试。"""
    assert client._http_error(429, "slow down").retryable
    assert client._http_error(503, "busy").retryable
    assert not client._http_error(401, "bad key").retryable
    assert client._http_error(401, "bad key").status_code == 401


def test_estimate_usage_is_marked_estimated() -> None:
    """服务端不回 usage 时按字数估算，必须打上 estimated 标记。"""
    usage = client._estimate_usage([{"role": "user", "content": "中" * 60}], "答" * 30)

    assert usage.estimated is True
    assert usage.prompt_tokens >= 40
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


# ---------------- 编排：事件顺序与日志 ----------------


def _fake_stream(answer: str):
    def stream(messages):  # noqa: ANN001 - 测试替身
        for index in range(0, len(answer), 5):
            yield client.LlmEvent(kind="delta", text=answer[index : index + 5])
        yield client.LlmEvent(
            kind="done",
            text=answer,
            usage=client.TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    return stream


def _patch_log(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    saved: list[dict] = []

    def fake_save(plan, **kwargs):  # noqa: ANN001 - 测试替身
        saved.append({"question": plan.question, **kwargs})
        return True

    monkeypatch.setattr(pipeline, "save_call_log", fake_save)
    return saved


def test_stream_answer_emits_delta_then_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """事件顺序固定：若干 delta → 一条 citations；citations 里带引用列表与校验结果。"""
    answer = f"结论：10mm² [{DOC} p.24]\n排查步骤\n1. 核对表4-2 [{DOC} p.24]\n来源\n[{DOC} p.24]"
    monkeypatch.setattr(client, "stream_chat", _fake_stream(answer))
    saved = _patch_log(monkeypatch)

    events = list(pipeline.stream_answer(_plan()))

    assert {item["event"] for item in events} == {"delta", "citations"}
    assert events[-1]["event"] == "citations"
    payload = events[-1]["data"]
    assert payload["answer_status"] == "answered"
    assert payload["citation_check"]["accuracy"] == 1.0
    assert payload["citations"][0]["doc"] == DOC
    assert payload["usage"]["total_tokens"] == 120
    assert "".join(item["data"]["text"] for item in events if item["event"] == "delta") == answer
    assert len(saved) == 1
    assert saved[0]["status"] == "answered"


def test_refusal_is_recorded_as_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """拒答要能被识别，日志里单独一个状态，便于统计「答不出来」的比例。"""
    monkeypatch.setattr(client, "stream_chat", _fake_stream(f"{REFUSAL_MARKER}\n缺少的信息：液压参数"))
    saved = _patch_log(monkeypatch)

    payload = list(pipeline.stream_answer(_plan()))[-1]["data"]

    assert payload["answer_status"] == "refused"
    assert payload["citation_check"]["unique"] == 0
    assert saved[0]["status"] == "refused"


def test_llm_error_still_returns_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型失败时也要把「本来检索到了什么」交给前端，并如实记状态。"""

    def failing(messages):  # noqa: ANN001 - 测试替身
        yield client.LlmEvent(kind="delta", text="结论：")
        yield client.LlmEvent(
            kind="error",
            error=client.LlmError(code="timeout", message="超时", attempts=3, retryable=True),
        )

    monkeypatch.setattr(client, "stream_chat", failing)
    saved = _patch_log(monkeypatch)

    payload = list(pipeline.stream_answer(_plan()))[-1]["data"]

    assert payload["answer_status"] == "error"
    assert payload["error"]["code"] == "timeout"
    assert payload["citations"], "失败也要回传本次检索到的片段"
    assert saved[0]["status"] == "error"
    assert saved[0]["error"] is not None


# ---------------- 调用日志 ----------------


def test_save_call_log_persists_usage_and_timings(tmp_path) -> None:
    """token 用量、检索耗时与生成耗时必须真的落进日志表，字段不能丢。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'log.db'}")
    Base.metadata.create_all(engine)
    plan = _plan()
    check = citations.validate_answer(f"结论 [{DOC} p.24]", plan.allowed)

    with sessionmaker(bind=engine)() as session:
        ok = pipeline.save_call_log(
            plan,
            answer=f"结论 [{DOC} p.24]",
            status="answered",
            usage=client.TokenUsage(prompt_tokens=812, completion_tokens=96, total_tokens=908),
            error=None,
            generation_ms=1234.5,
            check=check,
            session=session,
        )
        row = session.execute(select(QaLog)).scalar_one()

    assert ok is True
    assert (row.prompt_tokens, row.completion_tokens, row.total_tokens) == (812, 96, 908)
    assert row.usage_estimated is False
    assert row.retrieval_ms == 12.5
    assert row.generation_ms == 1234.5
    assert row.retrieved_chunk_ids == plan.chunk_ids
    assert row.invalid_citations == []
    assert row.citations[0]["label"] == f"[{DOC} p.24]"


def test_save_call_log_failure_does_not_raise(tmp_path) -> None:
    """日志是旁路：写库失败返回 False，绝不把已经生成的回答弄丢。"""
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")  # 故意不建表

    with sessionmaker(bind=engine)() as session:
        ok = pipeline.save_call_log(
            _plan(),
            answer="结论",
            status="answered",
            usage=None,
            error=None,
            generation_ms=1.0,
            check=None,
            session=session,
        )

    assert ok is False


# ---------------- HTTP：SSE ----------------


@pytest.fixture()
def api() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        name, data = "", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            events.append((name, json.loads(data)))
    return events


def test_chat_endpoint_streams_meta_delta_citations_done(
    monkeypatch: pytest.MonkeyPatch, api: TestClient
) -> None:
    """curl 能直接看到的分帧顺序：meta → delta… → citations → done。"""
    monkeypatch.setattr(pipeline, "prepare", lambda question, mode, top_k: _plan(question))
    monkeypatch.setattr(
        client,
        "stream_chat",
        _fake_stream(f"结论：10mm² [{DOC} p.24]"),
    )
    _patch_log(monkeypatch)

    response = api.post("/api/chat", json={"question": "4T22G 选多粗的线"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert [name for name, _ in events][0] == "meta"
    assert [name for name, _ in events][-2:] == ["citations", "done"]
    assert [name for name, _ in events].count("delta") > 0
    final = events[-2][1]
    assert final["citations"][0]["page"] == 24
    assert "snippet" in final["citations"][0]
    assert final["timings"]["retrieval_ms"] == 12.5


def test_chat_endpoint_rejects_unknown_mode(api: TestClient) -> None:
    """模式非法在流开始前就返回 422，不用让前端去解析错误事件。"""
    response = api.post("/api/chat", json={"question": "上电无显示", "mode": "bm25_only"})

    assert response.status_code == 422


def test_chat_endpoint_reports_retrieval_failure_as_event(
    monkeypatch: pytest.MonkeyPatch, api: TestClient
) -> None:
    """检索异常不能变成 500，要作为结构化 error 事件发出去。"""

    def boom(question: str, mode: str, top_k: int) -> pipeline.AnswerPlan:
        raise RuntimeError("向量模型不可用")

    monkeypatch.setattr(pipeline, "prepare", boom)
    response = api.post("/api/chat", json={"question": "上电无显示"})

    events = _sse_events(response.text)
    assert events[0][0] == "error"
    assert events[0][1]["code"] == "retrieval_failed"
    assert events[1][0] == "done"
