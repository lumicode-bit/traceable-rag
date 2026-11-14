"""检索层单元测试：融合、分词、评测集加载、指标与模式校验。

这些用例不下载模型、不连数据库——凡是需要真实模型或索引的验证都放在
`python -m app.retrieval.bench` 里跑，避免 pytest 变成一条几小时的命令。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval import metrics
from app.eval.dataset import DatasetError, load_queries
from app.retrieval import InvalidModeError, search
from app.retrieval.bm25 import tokenize
from app.retrieval.hybrid import rrf_fuse


def test_rrf_fuse_prefers_agreement() -> None:
    """两路都靠前的块，融合分必须高于只在单路靠前的块。"""
    fused = rrf_fuse({"vector": [10, 20, 30], "bm25": [30, 40]}, k=60)

    by_id = {hit.chunk_id: hit for hit in fused}
    assert fused[0].chunk_id == 30, "两路都命中的 30 应排第一"
    assert by_id[30].score > by_id[10].score
    assert by_id[30].sources == "VB"
    assert by_id[10].sources == "V"
    assert by_id[40].sources == "B"
    # 原始排名必须保留，便于分析是哪一路救回来的
    assert by_id[30].vector_rank == 3
    assert by_id[30].bm25_rank == 1


def test_rrf_fuse_is_stable_for_ties() -> None:
    """分数相同时按 chunk_id 升序，保证多次运行结果一致。"""
    fused = rrf_fuse({"vector": [7], "bm25": [3]}, k=60)
    assert [hit.chunk_id for hit in fused] == [3, 7]


def test_rrf_k_changes_scale_not_order() -> None:
    """k 只压制名次差距的量级，不改变单路内部的先后关系。"""
    small = rrf_fuse({"vector": [1, 2, 3]}, k=1)
    large = rrf_fuse({"vector": [1, 2, 3]}, k=60)
    assert [hit.chunk_id for hit in small] == [1, 2, 3]
    assert [hit.chunk_id for hit in large] == [1, 2, 3]
    assert small[0].score > large[0].score


def test_tokenize_keeps_model_and_term() -> None:
    """自定义词典要保证型号和专业术语不被切碎。"""
    tokens = tokenize("CP600-4T22G 的母线电压与载波频率")
    assert "cp600" in tokens
    assert "母线电压" in tokens, f"「母线电压」被切碎：{tokens}"
    assert "载波频率" in tokens, f"「载波频率」被切碎：{tokens}"


def test_tokenize_drops_punctuation() -> None:
    """标点与空白不进倒排。"""
    assert tokenize("报 Err23，检查端子。") == ["报", "err23", "检查", "端子"]


def test_load_queries_requires_gold(tmp_path: Path) -> None:
    """缺少 gold 的样本必须报错，不能静默跳过——缺行会让分数虚高。"""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "q1", "query": "上电无显示"}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    with pytest.raises(DatasetError):
        load_queries(path)


def test_load_queries_reads_gold_pages(tmp_path: Path) -> None:
    """正常样本应能读出文档名与页码。"""
    path = tmp_path / "ok.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "上电无显示怎么排查",
                "gold": [{"doc": "a.pdf", "page": 30}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    queries = load_queries(path)

    assert len(queries) == 1
    assert queries[0].gold_docs == {"a.pdf"}
    assert queries[0].gold_pairs == {("a.pdf", 30)}


def test_metrics_document_and_page_levels() -> None:
    """文档级命中可以只命中文档，页码级必须落到金标页。"""
    assert metrics.hit_document(["a.pdf", "b.pdf"], {"b.pdf"})
    assert not metrics.hit_document(["a.pdf"], {"b.pdf"})
    assert metrics.hit_page([("a.pdf", 30)], {("a.pdf", 30)})
    assert not metrics.hit_page([("a.pdf", 31)], {("a.pdf", 30)})
    assert metrics.recall_at_k([True, True, False, False]) == 0.5
    assert metrics.recall_at_k([]) == 0.0


def test_percentile_interpolates() -> None:
    """P95 用线性插值，避免只看均值被个别慢查询带偏。"""
    assert metrics.percentile([10, 20, 30, 40, 50], 0.5) == 30.0
    assert metrics.percentile([10], 0.95) == 10.0
    assert metrics.percentile([], 0.95) == 0.0


def test_invalid_mode_is_rejected_before_touching_model() -> None:
    """非法模式要在加载模型之前就报错，不能等到编码阶段才失败。"""
    with pytest.raises(InvalidModeError):
        search("上电无显示", top_k=3, mode="bm25_only")
