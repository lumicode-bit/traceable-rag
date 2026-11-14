"""题库格式测试：拒答题必须显式声明，普通题缺金标必须报错。

这条规则的用意是防「少标一行，该题变成怎么答都算对」，把分数虚高。
"""
from __future__ import annotations

import json

import pytest

from app.eval.dataset import DatasetError, load_queries
from app.eval.run_eval import _load as load_eval_questions


def _write(tmp_path, rows: list[dict]):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_answer_question_without_gold_is_rejected(tmp_path) -> None:
    path = _write(tmp_path, [{"id": "q1", "query": "某个问题"}])

    with pytest.raises(DatasetError, match="gold"):
        load_queries(path)


def test_refusal_question_allows_empty_gold(tmp_path) -> None:
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "type": "unanswerable",
                "query": "出厂序列号规则是什么",
                "gold": [],
                "expect": "refusal",
            }
        ],
    )

    queries = load_queries(path)

    assert len(queries) == 1
    assert queries[0].expect == "refusal"
    assert queries[0].gold == []


def test_unknown_expect_value_is_rejected(tmp_path) -> None:
    path = _write(
        tmp_path,
        [{"id": "q1", "query": "问题", "gold": [{"doc": "a.pdf", "page": 1}], "expect": "maybe"}],
    )

    with pytest.raises(DatasetError, match="expect"):
        load_queries(path)


def test_run_eval_skips_questions_without_gold(tmp_path) -> None:
    """评测器只统计有金标的题；拒答题不参与召回计算。"""
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "type": "lookup",
                "query": "主功率线缆选多粗",
                "gold": [{"chunk_id": 7, "doc": "a.pdf", "page": 24}],
            },
            {
                "id": "q2",
                "type": "unanswerable",
                "query": "序列号规则",
                "gold": [],
                "expect": "refusal",
            },
        ],
    )

    questions = load_eval_questions(path)

    assert len(questions) == 2
    assert [item.id for item in questions if item.gold] == ["q1"]
    assert questions[0].gold_chunks == {7}
    assert questions[0].gold_pages == {("a.pdf", 24)}
