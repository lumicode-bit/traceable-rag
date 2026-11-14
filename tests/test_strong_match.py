"""强匹配通道的标识符抽取：只认「够长、含字母又含数字」的标识符。

这条规则的边界很关键——放太宽（把 `01`、`u1` 也算）会让通道对任何查询都触发，
等于把 RRF 排序推翻；放太窄（只认连字符）则漏掉 `CP600`、`DI2` 这类主力标识符。
"""
from __future__ import annotations

from app.retrieval.hybrid import _query_identifiers


def test_extracts_function_code_and_terminal() -> None:
    ids = _query_identifiers("我看图纸上DI2端子旁边标着正转点动，那它对应的功能参数F4-01设成几")

    assert "f4-01" in ids, "连字符功能码必须整体保留，否则被切成 f4 + 01"
    assert "di2" in ids


def test_extracts_model_and_doc_type() -> None:
    ids = _query_identifiers("CP600 的 4T22G 机型主功率线缆选多粗的线")

    assert "cp600" in ids
    assert "4t22g" in ids


def test_ignores_pure_digits_and_short_tokens() -> None:
    ids = _query_identifiers("选择 01 号端子 u1 还是 4 号")

    assert "01" not in ids, "纯数字命中面太广，不能进强匹配"
    assert "u1" not in ids, "两字符的端子号太短，留给定长检索"


def test_plain_sentence_has_no_identifier() -> None:
    """没有标识符的查询不触发强匹配——q035 就是这种，靠语义检索。"""
    assert _query_identifiers("机器断电以后，得等多久才能去拆上盖板") == []
