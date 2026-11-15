"""切块：优先按标题层级切，段落边界控制 400–800 字，表格整体保留。

职责边界：只做「解析结果 → 检索块」，不读文件、不写数据库。
每个块都带 page_no 与 section 路径；同一小节被拆开时，块首重复标题并保留 100 字重叠。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.ingest.loader import ParsedDocument, TextBlock

MIN_CHARS = 400
MAX_CHARS = 800
OVERLAP_CHARS = 100
MAX_HEADING_CHARS = 40
SECTION_MAX_CHARS = 255

# "1 概述"、"2.2.2控制端口分布"；分隔符可缺省，用于区分标题与列表项
_NUMBERING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})(\s+|、|．|\.)?(\S.*)$")
_TABLE_CAPTION_RE = re.compile(r"^[图表]\s*\d+(?:[-.]\d+)*")
# 目录行：标题后跟点线引导符与页码，如 "1.5.2 系统外围电气元件......13"
_TOC_LINE_RE = re.compile(r"[.．·…]{3,}\s*\d+\s*$")
# 编号后面紧跟单位时说明是数值而非标题，如 "5.5 kW"
_UNIT_START_RE = re.compile(
    r"^(?:k?[Ww]|mm|cm|m|MPa|kPa|Pa|V|A|Hz|rpm|kg|t|h|min|s|L|dB|%|℃|°C|N·m|r/min)"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass
class ChunkDraft:
    """待入库的检索块。"""

    page_no: int | None
    section: str | None
    content: str


def split(parsed: ParsedDocument) -> list[ChunkDraft]:
    """把整份文档切成检索块。"""
    blocks = parsed.iter_blocks()
    body_size = _body_size(blocks)
    size_levels = _size_level_map(blocks, body_size)
    repeated_labels = _repeated_labels(blocks)

    drafts: list[ChunkDraft] = []
    stack: list[tuple[int, str]] = []
    units: list[TextBlock] = []
    title = ""
    section = ""

    for block in blocks:
        text = block.text.strip()
        if _TOC_LINE_RE.search(text):
            continue  # 目录行不参与切块，避免污染检索

        level = _heading_level(block, body_size, size_levels, repeated_labels)
        if level is None:
            if text:
                units.append(block)
            continue

        if units:
            drafts.extend(_pack(title, section, units))
            units = []

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, _clean_heading(block.text)))
        title = stack[-1][1]
        section = " > ".join(item for _, item in stack)

    if units:
        drafts.extend(_pack(title, section, units))
    return drafts


def _size_key(size: float) -> float:
    """字号归一到 0.5 的粒度，抵消 PDF 里的浮点抖动。"""
    return round(size * 2) / 2


def _repeated_labels(blocks: list[TextBlock]) -> set[str]:
    """全文反复出现的超短标签（危险/警告/注意）是提示框，不是章节标题。"""
    counter: dict[str, int] = {}
    for block in blocks:
        if block.kind != "paragraph":
            continue
        text = block.text.strip()
        if 0 < len(text) <= 6:
            counter[text] = counter.get(text, 0) + 1
    return {text for text, count in counter.items() if count >= 4}


def _body_size(blocks: list[TextBlock]) -> float:
    """按段落条数求正文字号。

    只统计 20 字以上的段落，避免把表格里的小字号单元格当成正文基准。
    """
    counter: dict[float, int] = {}
    for block in blocks:
        if block.kind != "paragraph" or len(block.text.strip()) < 20:
            continue
        key = round(block.font_size, 1)
        counter[key] = counter.get(key, 0) + 1
    if not counter:
        return 0.0
    return max(counter.items(), key=lambda item: (item[1], item[0]))[0]


def _size_level_map(blocks: list[TextBlock], body_size: float) -> dict[float, int]:
    """把明显大于正文的字号从大到小映射为 1..N 级标题。"""
    if body_size <= 0:
        return {}
    sizes = sorted(
        {
            _size_key(block.font_size)
            for block in blocks
            if block.kind == "paragraph" and block.font_size >= body_size + 1.0
        },
        reverse=True,
    )
    return {size: min(index + 1, 4) for index, size in enumerate(sizes)}


def _heading_level(
    block: TextBlock,
    body_size: float,
    size_levels: dict[float, int],
    repeated_labels: set[str],
) -> int | None:
    """判定块是否为标题，并给出层级；返回 None 表示正文。"""
    if block.kind != "paragraph":
        return None
    text = block.text.strip()
    if not text or len(text) > MAX_HEADING_CHARS:
        return None
    if text[-1] in "。；，、,;：:":
        return None
    if _TABLE_CAPTION_RE.match(text):
        return None
    if text in repeated_labels:
        return None

    size = _size_key(block.font_size)
    bigger = body_size > 0 and block.font_size >= body_size + 0.4

    if block.heading_level:
        return block.heading_level

    numbered = _numeric_heading_level(text)
    if numbered is not None:
        level, separator = numbered
        # 「1、检查油位」这类列表项通常与正文同字号，不作为标题
        if separator == "、" and not bigger:
            return None
        if bigger or block.bold or level >= 2:
            return level
        return None

    # 纯字号判定：要求是中文短标题，排除「②」「Min」这类符号/英文碎片
    if not bigger or len(text) < 3 or not _CJK_RE.search(text):
        return None
    return size_levels.get(size)


def _numeric_heading_level(text: str) -> tuple[int, str] | None:
    """识别 1 / 1.1 / 1.1.1 形式的标题编号。"""
    match = _NUMBERING_RE.match(text)
    if not match:
        return None

    number, separator, title = match.groups()
    title = title.strip()
    if len(title) < 2 or _UNIT_START_RE.match(title):
        return None
    # 标题里至少有一个汉字，避免把 "1.5 2.0" 这类数值行当标题
    if not _CJK_RE.search(title):
        return None
    return number.count(".") + 1, separator or ""


def _clean_heading(text: str) -> str:
    """去掉装饰性前缀（§、#），保留编号，便于引用定位。"""
    return re.sub(r"^[§#\s\u3000]+", "", text).strip()


def _pack(title: str, section: str, units: list[TextBlock]) -> list[ChunkDraft]:
    """把同一小节下的段落/表格装进 400–800 字的块。"""
    prefix = f"{title}\n" if title else ""
    prefix_len = len(prefix)

    bodies: list[tuple[str, int | None]] = []
    buffer = ""
    buffer_page: int | None = None

    for unit in units:
        text = unit.text.strip()
        if not text:
            continue

        if unit.kind == "table":
            # 短表保持原子性（能带上下文就带）；长表按行拆、每块复制表头。
            # TABLE_SPLIT_CHARS=0（默认）时不拆——实测拆完更差，见 DESIGN 的对照记录。
            pieces = (
                split_table(text, settings.table_split_chars)
                if settings.table_split_chars > 0
                else [text]
            )
            if len(pieces) > 1:
                if buffer:
                    bodies.append((buffer, buffer_page))
                    buffer = ""
                    buffer_page = None
                for piece in pieces:
                    bodies.append((piece, unit.page_no))
                continue

            # 表格是原子单位：能放进当前块就带上下文，放不下就单独成块，绝不拆开。
            # 当前块还很短（例如只有标题和表题）时优先跟随表格，避免标题被孤立。
            fits = bool(buffer) and len(buffer) + len(text) + 1 + prefix_len <= MAX_CHARS
            if fits or (buffer and len(buffer) < MIN_CHARS):
                buffer = f"{buffer}\n{text}" if buffer else text
            else:
                if buffer:
                    bodies.append((buffer, buffer_page))
                buffer = text
                buffer_page = unit.page_no
            continue

        # 超长段落按句子边界先拆开，避免单块远超上限
        for piece in _split_long_text(text, max(MIN_CHARS, MAX_CHARS - prefix_len)):
            if not buffer:
                buffer = piece
                buffer_page = unit.page_no
                continue

            if len(buffer) + len(piece) + 1 + prefix_len > MAX_CHARS:
                bodies.append((buffer, buffer_page))
                overlap = _tail(buffer, OVERLAP_CHARS)
                if overlap and len(overlap) + len(piece) + 1 + prefix_len > MAX_CHARS:
                    overlap = ""
                buffer = f"{overlap}\n{piece}" if overlap else piece
                buffer_page = unit.page_no
            else:
                buffer = f"{buffer}\n{piece}"

    if buffer:
        bodies.append((buffer, buffer_page))

    bodies = _merge_short(bodies)

    drafts: list[ChunkDraft] = []
    for body, page_no in bodies:
        content = f"{prefix}{body}" if prefix else body
        drafts.append(
            ChunkDraft(
                page_no=page_no,
                section=section[:SECTION_MAX_CHARS] if section else None,
                content=content,
            )
        )
    return drafts


def split_table(text: str, limit: int) -> list[str]:
    """把长表格按行拆成多块，**每块都复制表头**。

    为什么不再坚持「表格原子性」：一张 2900 字的参数表整块入库，向量是被几十行
    平均出来的方向，语义检索几乎排不到它前面；而用户问的往往只是其中一行。
    拆开、每块带上表头之后，「某一行」的语义才可能被命中，同时模型看到的仍是
    「表头 + 若干行」，不会出现有数据没表头的情况。

    - 表头 = 第一行 + 紧随其后的分隔行（`| --- |`）；
    - 若表头本身是两行（合并单元格被拍平成的第二行），一并复制；
    - 表格之外的说明文字（表题、脚注）跟第一块走，不重复。
    返回 [text] 表示不需要拆。
    """
    if len(text) <= limit:
        return [text]

    lines = text.splitlines()
    table_start = next((i for i, line in enumerate(lines) if line.strip().startswith("|")), None)
    if table_start is None:
        return [text]

    preamble = lines[:table_start]
    table_lines = [line for line in lines[table_start:] if line.strip()]
    if len(table_lines) < 4:
        return [text]  # 去掉表头与分隔行后没几行数据，拆了只剩碎片

    header: list[str] = [table_lines[0]]
    body_start = 1
    if len(table_lines) > 1 and set(table_lines[1].strip()) <= set("|-: "):
        header.append(table_lines[1])
        body_start = 2
    # 第二行表头（合并单元格拍平后常见）：数据行前若还有一行几乎全是空单元格，算作表头
    if body_start < len(table_lines) - 1:
        candidate = table_lines[body_start]
        cells = [cell.strip() for cell in candidate.strip("|").split("|")]
        if cells and sum(1 for cell in cells if not cell) >= max(1, len(cells) // 2):
            header.append(candidate)
            body_start += 1

    rows = table_lines[body_start:]
    header_text = "\n".join(header)
    header_len = len(header_text) + 1

    pieces: list[list[str]] = []
    current: list[str] = []
    current_len = header_len
    for row in rows:
        if current and current_len + len(row) + 1 > limit:
            pieces.append(current)
            current = []
            current_len = header_len
        current.append(row)
        current_len += len(row) + 1
    if current:
        pieces.append(current)

    if len(pieces) <= 1:
        return [text]

    result: list[str] = []
    for index, piece in enumerate(pieces):
        body = f"{header_text}\n" + "\n".join(piece)
        if index == 0 and preamble:
            body = "\n".join(preamble) + "\n" + body
        result.append(body)
    return result


def _split_long_text(text: str, limit: int) -> list[str]:
    """把超长段落按句号、分号等边界拆成不超过 limit 的片段。"""
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[。！？；])", text):
        if not sentence:
            continue
        if len(current) + len(sentence) <= limit:
            current += sentence
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)

    # 极端情况：单句仍超长，只能硬切
    result: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            result.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            result.append(piece)
    return result


def _merge_short(bodies: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    """同一小节内把不足 400 字的块并入相邻块，减少碎片块。

    只做拼接，不拆表格，所以不会破坏「表格整体保留」。
    """
    merged: list[tuple[str, int | None]] = []
    for body, page_no in bodies:
        if merged:
            prev_body, prev_page = merged[-1]
            too_short = len(body) < MIN_CHARS or len(prev_body) < MIN_CHARS
            if too_short and len(prev_body) + len(body) + 1 <= MAX_CHARS:
                merged[-1] = (f"{prev_body}\n{body}", prev_page)
                continue
        merged.append((body, page_no))
    return merged


def _tail(text: str, size: int) -> str:
    """取块尾做重叠，尽量从句子边界开始，避免截断半句。"""
    if len(text) <= size:
        return text.strip()
    segment = text[-size:]
    for mark in ("。", "；", "！", "？", "\n"):
        index = segment.find(mark)
        if -1 < index < len(segment) - 10:
            return segment[index + 1 :].strip()
    return segment.strip()
