"""拒答判定：必须是「结论段只有未找到 + 全篇没有出处」，不能用全文包含。

这条判定直接决定「作答率」这个指标。早先用 `REFUSAL_MARKER in answer`，模型
「先答、再补一句片段里没有 X」的诚实回答会被算成拒答——实测 q008、q045 都是这样
被误计的，导致同一份代码跑出 45/46/47 的假波动。
"""
from __future__ import annotations

from app.llm.citations import is_refusal

PURE_REFUSAL = """结论
资料中未找到相关内容
缺少的信息：CP600 对应的输入电压与频率规格

排查步骤
无

来源
无"""

ANSWER_THEN_GAP = """结论
通电前应确认电源电压为 AC380V~480V、50/60Hz [CP320变频器用户手册.pdf p.24]。

排查步骤
1. 用万用表测量进线电压 [CP320变频器用户手册.pdf p.24]。
2. 片段中未给出 CP600 对应的规格，需另查 CP600 手册。

来源
[CP320变频器用户手册.pdf p.24]"""


def test_pure_refusal_is_refusal() -> None:
    assert is_refusal(PURE_REFUSAL) is True


def test_answer_then_gap_is_not_refusal() -> None:
    """先作答、后补边界，是诚实而不是拒答——这是最容易被误判的形态。"""
    assert is_refusal(ANSWER_THEN_GAP) is False


def test_markdown_heading_still_works() -> None:
    answer = "## 结论\n资料中未找到相关内容\n\n## 排查步骤\n无"
    assert is_refusal(answer) is True


def test_marker_after_conclusion_is_not_refusal() -> None:
    """标记出现在结论之后（比如排查步骤里），结论段本身是有效回答。"""
    answer = "结论\n应按 C 型曲线设置 [CP600空压机一体机手册.pdf p.24]。\n\n排查步骤\n1. 资料中未找到相关内容的说明见附录。"
    assert is_refusal(answer) is False


def test_empty_answer_is_not_refusal() -> None:
    """空回答是「没答」，属于失败，不该被算成「正确地拒答」。"""
    assert is_refusal("") is False
