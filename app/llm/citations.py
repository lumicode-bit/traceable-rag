"""引用的领域对象与校验规则：本次检索到的片段是唯一合法的来源集合。

这里只做纯文本处理，不查库、不调模型，因此校验脚本与单元测试都能直接复用。

校验口径（回答里的引用是否幻觉）：

- 合法：`[文档名 p.页码]` 的 (文档名, 页码) 命中本次检索到的某个片段；
- 幻觉：文档名或页码对不上任何片段；
- 畸形：看起来想写引用（含 `p.`、`第N页`）但不符合格式——同样算无效，
  否则模型写成 `[手册 p.24-25]` 会因为「没被识别」而被算成 100% 正确。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 回答里的引用：[文档名 p.页码]，容许模型在括号内多写空格、大小写混用
_CITATION_RE = re.compile(r"^\s*(?P<doc>.+?)\s*[pP]\.\s*(?P<page>\d+)\s*$")
# 所有中括号块；非引用块（如 [片段1]）直接忽略
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
# 只给文档名、不给页码的引用（DOCX 片段没有页码时才合法）
_DOC_ONLY_RE = re.compile(r"^\s*(?P<doc>.+?\.(?:pdf|docx|doc|png|jpg|jpeg))\s*$", re.IGNORECASE)
# 想写引用却没写成标准格式的特征
_MALFORMED_HINT_RE = re.compile(r"[pP]\s*\.\s*\d|第\s*\d+\s*页|\d+\s*页")

# 模型在片段不足时必须回这句话，用于判定「拒答」而不是「编了一个答案」
REFUSAL_MARKER = "资料中未找到相关内容"


@dataclass(frozen=True)
class Citation:
    """一个可被引用的片段：来源信息 + 完整正文 + 正文摘要。

    两个正文字段不能混用，混用会直接毁掉回答质量：

    - `content` 是**完整块正文**，只送给模型。表格类块（故障码表 1800+ 字）如果只给
      前 120 字，模型看到的全是表头，只能拒答——这一条是被真实评测抓出来的；
    - `snippet` 是**给前端看的摘要**，进响应体与 `qa_logs`，不参与提示词。
    """

    chunk_id: int
    doc: str
    page: int | None
    section: str | None
    content: str
    snippet: str

    @property
    def label(self) -> str:
        """唯一合法的来源写法，与系统提示词要求的格式完全一致。"""
        if self.page is None:
            # DOCX 拿不到页码，只能给文档名，不能编一个页码填上
            return f"[{self.doc}]"
        return f"[{self.doc} p.{self.page}]"

    def as_dict(self) -> dict[str, Any]:
        """对外（SSE / 日志）只给摘要：完整正文不进响应体，一次问答不回传上万字。"""
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc,
            "page": self.page,
            "section": self.section,
            "snippet": self.snippet,
            "chars": len(self.content),
            "label": self.label,
        }


@dataclass(frozen=True)
class CitationIssue:
    """一条无效引用及其原因。"""

    label: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "reason": self.reason}


@dataclass
class CitationCheck:
    """一次回答的引用校验结果。"""

    occurrences: int = 0
    valid: list[str] = field(default_factory=list)
    invalid: list[CitationIssue] = field(default_factory=list)

    @property
    def unique(self) -> int:
        return len({_normalize_label(label) for label in self.valid}) + len(
            {_normalize_label(issue.label) for issue in self.invalid}
        )

    @property
    def accuracy(self) -> float | None:
        """有效引用 / 去重后的全部引用；没有引用时返回 None（拒答是合法的 0 引用）。"""
        total = self.unique
        if total == 0:
            return None
        return (total - self._unique_invalid) / total

    @property
    def _unique_invalid(self) -> int:
        return len({_normalize_label(issue.label) for issue in self.invalid})

    def as_dict(self) -> dict[str, Any]:
        return {
            "occurrences": self.occurrences,
            "unique": self.unique,
            "valid": list(self.valid),
            "invalid": [issue.as_dict() for issue in self.invalid],
            "accuracy": self.accuracy,
        }


def allowed_keys(citations: list[Citation]) -> set[tuple[str, int | None]]:
    """本次检索允许引用的来源集合；校验与提示词用的是同一份数据。"""
    return {(citation.doc, citation.page) for citation in citations}


def parse_citations(answer: str) -> list[str]:
    """按出现顺序取出回答里所有像引用的中括号块（含畸形块）。"""
    found: list[str] = []
    for match in _BRACKET_RE.finditer(answer):
        inner = match.group(1).strip()
        if not inner:
            continue
        if (
            _CITATION_RE.match(inner)
            or _DOC_ONLY_RE.match(inner)
            or _MALFORMED_HINT_RE.search(inner)
        ):
            found.append(match.group(0))
    return found


def validate_answer(answer: str, allowed: set[tuple[str, int | None]]) -> CitationCheck:
    """逐条核对引用是否落在本次检索的片段上；不允许用别的知识凑来源。

    `occurrences` 记出现次数，`valid` / `invalid` 按去重后的来源各记一条：
    同一来源写三遍不会把准确率刷成好看的数字。
    """
    check = CitationCheck()
    docs = {_normalize_doc(doc): (doc, page) for doc, page in allowed}
    seen: set[str] = set()
    for label in parse_citations(answer):
        check.occurrences += 1
        key = _normalize_label(label)
        if key in seen:
            continue
        seen.add(key)
        issue = _check_one(label, allowed, docs)
        if issue is None:
            check.valid.append(label)
        else:
            check.invalid.append(issue)
    return check


# 「结论」段与「排查步骤」段的分界；模型有时写成 `## 排查步骤`
_STEP_SPLIT = re.compile(r"\n\s*#*\s*排查步骤")
# 任意方括号引用（不校验真伪，只判断「有没有给出处」）
_ANY_CITATION = re.compile(r"\[[^\[\]]{2,120}\]")


def is_refusal(answer: str) -> bool:
    """回答是否**就是**拒答：结论段只有「未找到」，且全篇没有给任何出处。

    不能用 `REFUSAL_MARKER in answer`。模型经常「先完整作答，再补一句边界」，
    例如 q008：

        结论
        通电前应确认电源电压为 AC380V~480V、50/60Hz [CP320…p.24]。
        …
        排查步骤
        （补一句）片段中未给出 CP600 对应的规格。

    这类回答是**诚实**（答了能答的、说明了不能答的），用全文包含判定会把它们
    错算成拒答——实测 q008 与 q045 都这样被误计，导致「作答 45~47 波动」的假象。

    真正的拒答长这样（结论段第一句就是标记，且没有任何引用）：

        结论
        资料中未找到相关内容
        缺少的信息：…
    """
    text = answer or ""
    head = _STEP_SPLIT.split(text)[0]
    conclusion = head.split("结论", 1)[-1].strip().lstrip("#* 　").strip()
    return conclusion.startswith(REFUSAL_MARKER) and not _ANY_CITATION.search(text)


def _check_one(
    label: str,
    allowed: set[tuple[str, int | None]],
    docs: dict[str, tuple[str, int | None]],
) -> CitationIssue | None:
    """返回 None 表示合法，否则返回问题原因。"""
    inner = label.strip()[1:-1].strip()
    match = _CITATION_RE.match(inner)
    if match is None:
        return _check_doc_only(label, inner, allowed, docs)

    doc_text = match.group("doc").strip()
    page = int(match.group("page"))
    normalized = _normalize_doc(doc_text)
    if normalized not in docs:
        return CitationIssue(label=label, reason=f"文档不在本次检索结果中：{doc_text}")

    # 同一文档可能有多个片段，页码必须落在其中一个片段的页上
    pages = {item_page for item_doc, item_page in allowed if _normalize_doc(item_doc) == normalized}
    if page not in pages:
        available = "、".join(str(item) for item in sorted(p for p in pages if p is not None))
        return CitationIssue(
            label=label,
            reason=f"页码 {page} 不在本次检索到的片段中（该文档可用页码：{available or '无'}）",
        )
    return None


def _check_doc_only(
    label: str,
    inner: str,
    allowed: set[tuple[str, int | None]],
    docs: dict[str, tuple[str, int | None]],
) -> CitationIssue | None:
    """处理没有页码的引用：DOCX 片段确实没有页码可给，PDF 片段则必须写出页码。

    带文件后缀的中括号不再当成「不认识的括号」放过，否则编一个不存在的文件名
    （`[不存在的手册.pdf]`）会完全不计入分母，把准确率刷成 100%。
    """
    match = _DOC_ONLY_RE.match(inner)
    if match is None:
        return CitationIssue(label=label, reason="格式不是 [文档名 p.页码]")

    doc_text = match.group("doc").strip()
    normalized = _normalize_doc(doc_text)
    if normalized not in docs:
        return CitationIssue(label=label, reason=f"文档不在本次检索结果中：{doc_text}")

    pages = {
        item_page for item_doc, item_page in allowed if _normalize_doc(item_doc) == normalized
    }
    if pages != {None}:
        available = "、".join(str(item) for item in sorted(p for p in pages if p is not None))
        return CitationIssue(
            label=label,
            reason=f"缺少页码；该文档的片段有页码：{available}",
        )
    return None


def _normalize_doc(doc: str) -> str:
    """比对文档名时忽略空白与路径，容忍模型把文件名写成 `手册.pdf` 或 `手册`。"""
    text = "".join(doc.split())
    for prefix in ("./", "data/raw/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.lower().endswith((".pdf", ".docx", ".doc")):
        text = text.rsplit(".", 1)[0]
    return text


def _normalize_label(label: str) -> str:
    return "".join(label.split()).lower()
