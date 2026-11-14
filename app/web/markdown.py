"""极简 Markdown 渲染：只做评测报告需要的那几种语法。

为什么自己写：`requirements.txt` 里每一个包都要为「能不能在这台机器上跑起来」负责，
为了渲染两份报告引入 markdown 库不值得。这里只支持标题、表格、列表、代码块、
引用块、分隔线与行内 **粗体** / `代码`，其余一律按普通段落处理。

安全前提：先做 HTML 转义再拼标签，因此报告里的 `<script>` 只会显示成文本。
"""
from __future__ import annotations

import html
import re

_INLINE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def render(markdown: str) -> str:
    """把 Markdown 转成 HTML 片段（不含 <html> 外壳）。"""
    lines = (markdown or "").replace("\r\n", "\n").split("\n")
    blocks: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        # 代码块：原样保留（转义后），不做行内处理
        if stripped.startswith("```"):
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            blocks.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        # 表格：当前行以 | 开头且下一行是分隔行
        if stripped.startswith("|") and index + 1 < len(lines) and _TABLE_SEPARATOR.match(
            lines[index + 1].strip()
        ):
            table, index = _render_table(lines, index)
            blocks.append(table)
            continue

        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            text = stripped[level:].strip()
            blocks.append(f"<h{level}>{_inline(text)}</h{level}>")
            index += 1
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            blocks.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue

        if re.match(r"^[-*+]\s+", stripped):
            items: list[str] = []
            while index < len(lines) and re.match(r"^[-*+]\s+", lines[index].strip()):
                items.append(f"<li>{_inline(lines[index].strip()[2:].strip())}</li>")
                index += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue

        if re.match(r"^-{3,}$", stripped):
            blocks.append("<hr>")
            index += 1
            continue

        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if (
                not current
                or current.startswith(("#", ">", "```", "|"))
                or re.match(r"^[-*+]\s+", current)
            ):
                break
            paragraph.append(current)
            index += 1
        blocks.append(f"<p>{_inline(' '.join(paragraph))}</p>")

    return "\n".join(blocks)


def _render_table(lines: list[str], start: int) -> tuple[str, int]:
    """把 | a | b | 形式的表格转成 <table>；首行当表头。"""
    header_cells = _split_row(lines[start])
    index = start + 2  # 跳过表头与分隔行
    body: list[list[str]] = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        body.append(_split_row(lines[index]))
        index += 1

    columns = len(header_cells)
    head = "".join(f"<th>{_inline(cell)}</th>" for cell in header_cells)
    rows = []
    for cells in body:
        padded = cells + [""] * max(columns - len(cells), 0)
        rows.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in padded) + "</tr>")

    html_table = (
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return html_table, index


def _split_row(line: str) -> list[str]:
    """切分表格行：去掉首尾竖线后按 | 拆，单元格内不做转义以外处理。"""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _inline(text: str) -> str:
    """行内语法：先转义再替换，保证报告内容里的标签不会被执行。"""
    escaped = html.escape(text)
    escaped = _INLINE_BOLD.sub(r"<strong>\1</strong>", escaped)
    return _INLINE_CODE.sub(r"<code>\1</code>", escaped)
