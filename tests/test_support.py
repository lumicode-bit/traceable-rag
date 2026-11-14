"""引用支撑性校验（规则版）的单元测试。

这套规则的职责边界要说清楚：它只判断**结论里的数值/型号/故障码有没有出现在被引用的块里**，
不判断语义是否等价。因此测试里既验证它能抓住「数字串台」，也验证它**不硬判**纯定性结论。
"""
from __future__ import annotations

from app.llm.support import check_answer, detect_tokens

BODY_A = "表4-2 4T22G 机型推荐 16mm2 铜芯线，母线电压 380V 以下停机。"
BODY_B = "用绝缘测试仪测量电机和输出线的绝缘，Err23 表示对地短路。"


def _resolve(label: str) -> str | None:
    return {
        "[CP600空压机一体机手册.pdf p.24]": BODY_A,
        "[CP650空压机一体机手册.pdf p.35]": BODY_B,
    }.get(label)


def test_supported_claim_passes() -> None:
    answer = "结论\n4T22G 推荐 16mm² 铜芯线 [CP600空压机一体机手册.pdf p.24]。\n"

    result = check_answer(answer, _resolve)

    assert result.checked == 1
    assert result.supported == 1
    assert result.unsupported == 0
    assert result.ratio == 1.0


def test_fabricated_error_code_is_caught() -> None:
    """引用真实存在，但该块里没有 Err99——这正是 citations.py 抓不到的那类错。"""
    answer = "排查步骤\n1. 更换 Err99 保险 [CP650空压机一体机手册.pdf p.35]。\n"

    result = check_answer(answer, _resolve)

    assert result.unsupported == 1
    assert result.claims[0].missing == ["Err99"]


def test_qualitative_claim_is_not_judged() -> None:
    """没有数值/型号可比的定性结论记 unknown，不能算它错，也不能算它对。"""
    answer = "排查步骤\n1. 先断电再拆盖板 [CP650空压机一体机手册.pdf p.35]。\n"

    result = check_answer(answer, _resolve)

    assert result.unknown == 1
    assert result.checked == 0
    assert result.ratio is None


def test_citation_label_is_not_treated_as_fact() -> None:
    """`[CP600…p.24]` 里的型号不能被当结论事实，否则每句都会被判不支持。"""
    answer = "结论\n推荐 16mm² 铜芯线 [CP600空压机一体机手册.pdf p.24]。\n"

    result = check_answer(answer, _resolve)

    assert result.supported == 1


def test_source_section_is_skipped() -> None:
    """「来源」小节只是引用清单，不该被当成结论去核对。"""
    answer = (
        "结论\n推荐 16mm² [CP600空压机一体机手册.pdf p.24]。\n\n"
        "来源\n[CP600空压机一体机手册.pdf p.24]\n"
    )

    result = check_answer(answer, _resolve)

    assert result.claims == 1 or len(result.claims) == 1


def test_detect_tokens_covers_units_and_codes() -> None:
    tokens = detect_tokens("低于 380V 时报 Err23，参数 F4-01 设为 4，端子 DI2 接 U1")

    assert "380V" in tokens
    assert "Err23" in tokens
    assert "F4-01" in tokens
    assert "DI2" in tokens


def test_model_name_may_come_from_context() -> None:
    """对比句只引用其中一块是合理的：型号允许在本次上下文里找，数值仍必须在被引块里。"""
    answer = (
        "结论\nCP320 的端子是 U、V、W，与 CP650 不同 "
        "[CP650空压机一体机手册.pdf p.35]。\n"
    )

    result = check_answer(answer, _resolve, context=BODY_A + BODY_B + " CP320变频器用户手册.pdf")

    assert "CP320".lower() not in [token.lower() for token in result.claims[0].missing]


def test_number_still_requires_cited_block() -> None:
    """数值不能靠上下文蒙混：上下文里有 380V，但被引块里没有，仍算不被支撑。"""
    answer = "结论\n母线电压低于 380V 时停机 [CP650空压机一体机手册.pdf p.35]。\n"

    result = check_answer(answer, _resolve, context=BODY_A)

    assert result.unsupported == 1
    assert "380V" in result.claims[0].missing


def test_ratio_is_none_when_nothing_checkable() -> None:
    answer = "结论\n应当先断电 [CP650空压机一体机手册.pdf p.35]。\n"

    result = check_answer(answer, _resolve)

    assert result.ratio is None, "没有可判定句时必须返回 None，不能假装 100%"
