"""校验脚本自身的单元测试：假的检索 + 假的模型，验证「能抓出幻觉、能判失败」。

真实准确率必须用 `python tests/check_citations.py` 跑真实模型，
这里只保证脚本不会自己骗自己（比如把畸形引用当成不存在、或该失败时返回 0）。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.llm import pipeline, prompt
from app.llm.citations import Citation, allowed_keys

_SCRIPT = Path(__file__).with_name("check_citations.py")
_SPEC = importlib.util.spec_from_file_location("check_citations", _SCRIPT)
assert _SPEC and _SPEC.loader
check_citations = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_citations)

DOC = "CP600空压机一体机手册.pdf"
NO_ANSWER_HINTS = ("序列号", "报价")


def _fake_prepare(question: str, mode: str, top_k: int) -> pipeline.AnswerPlan:
    """真问题给两个片段，拒答题给空片段——后者如果还被作答，脚本必须判失败。"""
    passages: list[Citation] = []
    if not any(hint in question for hint in NO_ANSWER_HINTS):
        passages = [
            Citation(
                chunk_id=1,
                doc=DOC,
                page=24,
                section="4.4 主功率线缆",
                content="按表4-2 选择线径，4T22G 推荐 10mm²。",
                snippet="按表4-2 选择线径",
            ),
            Citation(
                chunk_id=2,
                doc="CP650空压机一体机手册.pdf",
                page=32,
                section="附录A",
                content="故障码 Err14 / Err23 对照表",
                snippet="故障码表",
            ),
        ]
    return pipeline.AnswerPlan(
        question=question,
        mode=mode,
        top_k=top_k,
        retrieval_ms=1.0,
        citations=passages,
        allowed=allowed_keys(passages),
        messages=prompt.build_messages(question, passages),
        chunk_ids=[item.chunk_id for item in passages],
    )


@pytest.fixture()
def queries(tmp_path: Path) -> Path:
    path = tmp_path / "queries.jsonl"
    lines = [
        {"id": "q01", "query": "上电无显示怎么排查", "gold": [{"doc": DOC, "page": 30}]},
        {"id": "q02", "query": "4T22G 主功率线缆选多粗", "gold": [{"doc": DOC, "page": 24}]},
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in lines) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """把检索与建库都挡掉，只留校验逻辑在跑。"""
    monkeypatch.setattr(check_citations, "_guard_index", lambda: 0)
    monkeypatch.setattr(check_citations, "init_db", lambda: None)
    monkeypatch.setattr(pipeline, "prepare", _fake_prepare)
    monkeypatch.setattr(pipeline, "save_call_log", lambda *args, **kwargs: True)


def test_mock_run_passes_when_citations_are_real(
    offline: None, queries: Path, tmp_path: Path
) -> None:
    """假模型复述提示词里的来源 → 引用全真、拒答题拒答 → 退出码 0。"""
    out = tmp_path / "report.json"

    code = check_citations.main(["--mock", "--queries", str(queries), "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] is True
    assert report["summary"]["invalid_citations"] == 0
    assert report["summary"]["accuracy"] == 1.0
    assert report["summary"]["unanswerable_refused"] == report["summary"]["unanswerable_total"]


def test_mock_with_hallucination_fails(offline: None, queries: Path) -> None:
    """故意编一个不存在的来源 → 必须判失败，退出码 1。"""
    code = check_citations.main(["--mock", "--hallucinate", "--queries", str(queries)])

    assert code == 1


def test_unanswerable_answering_is_failure() -> None:
    """该拒答却给了结论 → 汇总里必须出现对应问题。"""
    rows = [
        {
            "id": "q01",
            "status": "answered",
            "invalid": [],
            "unique": 1,
            "error": None,
        }
    ]
    summary = check_citations._summarize(rows, [{"question": "序列号规则", "refused": False, "reason": "未拒答"}])

    assert summary["passed"] is False
    assert any("该拒答而未拒答" in item for item in summary["problems"])


def test_answered_without_citation_is_failure() -> None:
    """有结论却一个来源都不给，无法溯源，同样算失败。"""
    rows = [{"id": "q01", "status": "answered", "invalid": [], "unique": 0, "error": None}]
    summary = check_citations._summarize(rows, [])

    assert summary["passed"] is False
    assert any("没有任何引用" in item for item in summary["problems"])


def test_refused_on_answerable_question_is_soft_failure() -> None:
    """误拒是**质量问题**，不是违约：记录、逐条列出，但不让 CI 变红。

    理由：门禁只该守「不许编」。把「该答没答」也算成红灯，CI 会因为 1/48 长期红，
    红久了就没人看——门禁等于没有。
    """
    rows = [{"id": "q01", "status": "refused", "invalid": [], "unique": 0, "error": None}]

    summary = check_citations._summarize(rows, [])

    assert summary["passed"] is True, "误拒不该阻塞门禁"
    assert summary["soft_fail"] == 1
    assert any("误拒" in item for item in summary["quality_issues"])
