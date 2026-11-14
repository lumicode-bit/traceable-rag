"""引用支撑性校验（规则版）：被引用的块里，到底有没有结论句里的关键事实。

**这一层补的是 `citations.py` 管不到的洞**：那边只证明「[手册 p.35] 这个来源存在」，
但答非所引（引用了真实存在、却不含该数值的段落）照样能过。这里做一件很土但有效的事：

1. 把回答切成「带引用的结论句」；
2. 从每句里抽出**可机械比对的事实**——数值+单位、故障码、型号、功能码、端子号；
3. 逐个到它引用的那几个块里找原文；找不到就记为「不被支撑」；
4. 句子里抽不出任何事实的（纯定性描述），记为「无法判定」，不硬判。

为什么先做规则版而不是直接上模型：
- 规则可解释、零成本、不会引入新的幻觉，适合当**第一道闸**；
- 真值就在块正文里，字符串包含判断已经能抓住「数字串台」这一类最危险的错误；
- 模型判蕴含可以作为第二道闸叠加，但要接受它自己也会错，不能拿它单独定案。

已知局限（写在这里，免得被当成万能）：
- 同义改写（「推荐 10 平方」vs 原文「10mm²」）会被判无法判定而不是被支撑；
- 单位换算（0.75kW vs 750W）不认；
- 定性结论（「应断电后操作」）没有可抽的 token，只能靠人工。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# 数值 + 单位：10mm²、400V、3.5kW、60s、50Hz、0.15MPa、25℃、80%
# 单位后面必须是「非字母数字」，否则 `4T22G` 会被拆成 `4T`（吨）与 `22G`（克）两个假 token，
# 而它们当然不在原文里，于是整句被判「不被支撑」。同理不收 `t`/`g` 这类歧义单位。
_UNIT_END = r"(?![A-Za-z0-9])"
_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm²|mm2|mm|m²|kV|kW|MW|kVA|VA|mA|A|V|kHz|Hz|MPa|kPa|Pa|bar|"
    r"kg|ms|min|℃|°C|%|MΩ|Ω|r/min|rpm|N·m|N\.m)"
    + _UNIT_END,
    re.IGNORECASE,
)
# 故障码 / 型号 / 功能码 / 端子号：Err23、CP600-4T22G、F4-01、A8-58、DI2、U1
#
# 型号尾巴必须限定 ASCII：`\w` 在 Python 里匹配中文，`CP\d{3}[-\w]*` 会把
# 「CP320系列变频器主要用于三相交流同步电机」整句吞成一个 token，导致每条结论都判成不支持。
_ID_LIKE = re.compile(
    r"(?:Err|Er|AL|E)\.?\d{1,3}"
    r"|[A-Z]{1,2}\d{1,2}-\d{1,3}"                    # F4-01 / A8-58
    r"|CP\d{3}[A-Za-z0-9._-]*"                       # CP600 / CP650-4T37-H
    r"|(?:DI|DO|AI|AO|T)\d{1,2}"                     # DI2 / AO1
    r"|[UVW][123]"                                   # U1 / V1 / W1
)
_CITATION = re.compile(r"\[[^\[\]]{2,120}\]")
_HEADINGS = {"结论", "排查步骤", "来源"}
# 只匹配列表序号，不碰内容里的数字
_LIST_MARKER = re.compile(r"^(?:[（(]?\d{1,2}[).、）]|[-*+])\s*")


@dataclass
class ClaimSupport:
    """一条结论句的支撑性判定。"""

    sentence: str
    labels: list[str]
    tokens: list[str]
    missing: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.tokens:
            return "unknown"  # 抽不出可比对的事实，不硬判
        return "unsupported" if self.missing else "supported"


@dataclass
class SupportCheck:
    """整段回答的支撑性汇总。"""

    claims: list[ClaimSupport] = field(default_factory=list)

    @property
    def checked(self) -> int:
        """可判定的结论句数（不含 unknown）。"""
        return sum(1 for claim in self.claims if claim.verdict != "unknown")

    @property
    def supported(self) -> int:
        return sum(1 for claim in self.claims if claim.verdict == "supported")

    @property
    def unsupported(self) -> int:
        return sum(1 for claim in self.claims if claim.verdict == "unsupported")

    @property
    def unknown(self) -> int:
        return sum(1 for claim in self.claims if claim.verdict == "unknown")

    @property
    def ratio(self) -> float | None:
        """在可判定的句子里，被支撑的比例；无可判定句时为 None（不假装 100%）。"""
        return self.supported / self.checked if self.checked else None

    def as_dict(self) -> dict:
        return {
            "claims": len(self.claims),
            "checked": self.checked,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "unknown": self.unknown,
            "ratio": self.ratio,
            "failures": [
                {
                    "sentence": claim.sentence[:160],
                    "labels": claim.labels,
                    "missing": claim.missing,
                }
                for claim in self.claims
                if claim.verdict == "unsupported"
            ],
        }


def check_answer(
    answer: str,
    resolve: Callable[[str], str | None],
    *,
    context: str = "",
) -> SupportCheck:
    """按引用标签逐句核对；`resolve(label) -> 块正文 or None` 由调用方提供。

    `context` 是**本次检索到的全部片段**（不只是被引用的那几块）。
    为什么需要它：像「CP320 图纸标注的对应关系与 CP650 不同」这种对比句，
    只引用 CP650 那块是合理的，型号 CP320 出现在另一块里。若要求型号也必须落在
    **被引用的**块里，这类正确引用会被判成「不被支撑」——实测 2 例全是这种误报。

    因此两类 token 的严格度不同：
    - **数值/单位**（`16mm²`、`380V`、`50Hz`）必须出现在被引用的块里——这才是「有据」；
    - **型号/器件名**（`CP320`、`CP650`）只要在本次上下文里出现即可——它更像「说的是哪台设备」，
      不是被断言的事实。
    """
    result = SupportCheck()
    context_corpus = _normalize(context) if context else ""
    for sentence in _split_claims(answer):
        labels = [match.group(0) for match in _CITATION.finditer(sentence)]
        if not labels:
            continue
        # 先把引用标签从句子里抠掉再抽事实：否则 `[CP600…p.24]` 里的型号会被当成
        # 「结论里的事实」，而它当然不会出现在块正文里，于是每句都判成不支持
        stripped = _CITATION.sub("", sentence)
        tokens = detect_tokens(stripped)
        numbers = _numeric_tokens(stripped)
        bodies = [body for body in (resolve(label) for label in labels) if body]
        # 引用的块一个都取不到，说明校验环节本身有问题，交给 citations 那一层报错
        if not bodies and tokens:
            result.claims.append(
                ClaimSupport(sentence=sentence, labels=labels, tokens=tokens, missing=tokens)
            )
            continue
        corpus = _normalize(" ".join(bodies))
        missing: list[str] = []
        for token in tokens:
            hit = _token_in(token, corpus)
            if not hit and token not in numbers and context_corpus:
                # 型号类只在全局上下文里找（见 docstring）
                hit = _token_in(token, context_corpus)
            if not hit:
                missing.append(token)
        result.claims.append(
            ClaimSupport(sentence=sentence, labels=labels, tokens=tokens, missing=missing)
        )
    return result


def _numeric_tokens(sentence: str) -> set[str]:
    """哪些 token 属于「数值+单位」——这类必须落在被引用的块里，不能靠上下文蒙混。"""
    return {match.group(0).strip() for match in _NUM_UNIT.finditer(sentence)}


def detect_tokens(sentence: str) -> list[str]:
    """抽出一句里可机械比对的事实 token（去重、保序）。"""
    tokens: list[str] = []
    for match in list(_NUM_UNIT.finditer(sentence)) + list(_ID_LIKE.finditer(sentence)):
        token = match.group(0).strip()
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _split_claims(answer: str) -> list[str]:
    """切成结论句：按中文句末标点与换行切，去掉标题行与「来源」小节。"""
    sentences: list[str] = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = stripped.strip("#* 　").rstrip("：:")
        if heading in _HEADINGS:
            # 「来源」之后只是引用清单，没有需要核对的结论
            if heading == "来源":
                break
            continue
        if stripped.startswith("来源"):
            break
        for piece in re.split(r"(?<=[。！？；])", stripped):
            # 只剥列表序号（`1. ` `2、` `（3）`），不能动数字本身，否则 `4T22G` 会变成 `T22G`
            piece = _LIST_MARKER.sub("", piece.strip()).strip()
            if piece and not piece.startswith("|") and "[" in piece:
                sentences.append(piece)
    return sentences


def _token_in(token: str, corpus: str) -> bool:
    """token 是否出现在正文里。

    两级判定，宁可放过不可错杀：
    1. 先比「归一化后的完整 token」（`16 mm²` → `16mm2`）；
    2. 再不中，只比数字部分——表格里常写成 `| 16 | mm2 |` 或 `59.0A` / `59 A` 混用，
       只差格式不该判成「不支持」。这会放过「单位对不上但数字对」的情况，
       属于已知的宽松处，写进文档而不是假装没这回事。
    """
    if corpus.find(_normalize(token)) != -1:
        return True
    match = re.match(r"\d+(?:\.\d+)?", token)
    if match is None:
        return False
    return _normalize(match.group(0)) in corpus


def _normalize(text: str) -> str:
    """比对前的归一化：只保留字母、数字、中文与 `.`/`%`。

    表格里的事实通常被竖线和空格切碎（`| 16 | mm2 |`），不把这些符号剥掉，
    `16mm2` 就永远匹配不上，所有表格类结论都会被误判成「不被支撑」。
    """
    cleaned = (
        text.lower()
        .replace("²", "2")
        .replace("·", ".")
        .replace("　", "")
    )
    return re.sub(r"[^0-9a-z\u4e00-\u9fff.%]", "", cleaned)
