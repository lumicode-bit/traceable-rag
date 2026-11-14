"""文档加载：把 PDF / 图片 / DOCX 解析成「页 + 段落/表格块」。

职责边界：只负责解析，不做切块、不写数据库。
PDF 先取文本层；单页字符数不足 30 且含图片时视为扫描页，渲染成图交给 OCR。
"""
from __future__ import annotations

import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from app.ingest import ocr

# 单页文本层字符数低于该值即判定为扫描页
SCANNED_PAGE_MAX_CHARS = 30
# 扫描页渲染分辨率，兼顾中文小字识别率与速度
OCR_RENDER_DPI = 220
SUPPORTED_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".docx")

_CJK_RE = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
# 页码行，如 "- 12 -"、"12"、"第 12 页"
_PAGE_NUMBER_RE = re.compile(r"^[-—–_\s]*(?:第\s*)?[0-9ivxIVX]{1,6}\s*(?:页)?[-—–_\s]*$")
# 项目符号字符，PDF 常把同一个符号画两遍，形成「◆ ◆」纯符号行
_BULLET_CHARS = "◆◇●○■□▪▫•◦‣∙·※★☆▶►▸▹"
_BULLET_TRIM_CHARS = _BULLET_CHARS + " \u3000\t"
# 同一行内两个 span 的水平间距超过该值即视为两段独立内容（页眉左右栏、多栏排版）
LINE_GAP_PT = 36.0
# 页眉页脚判定：字号比正文大出该值即视为标题/图表文字，不参与剔除
RUNNING_SIZE_SLACK = 3.0
# 页眉页脚重复阈值：>=3 次无条件判定；==2 次需文本较短（交替页眉常见形态）
RUNNING_REPEAT_HARD = 3
RUNNING_REPEAT_SOFT = 2
RUNNING_SHORT_TEXT = 24
# 表格题注，如 "表2-1 主回路端子说明"
_TABLE_CAPTION_RE = re.compile(r"^[图表]\s*\d+(?:[-.]\d+)*")


class UnsupportedFormatError(RuntimeError):
    """文件扩展名不在支持范围内。"""


class EmptyDocumentError(RuntimeError):
    """整份文档没有解析出任何可用文本。"""


@dataclass
class TextBlock:
    """切块的原子单位：一个自然段落，或一张完整的表格。"""

    text: str
    page_no: int | None
    font_size: float = 0.0
    bold: bool = False
    kind: str = "paragraph"  # paragraph | table
    heading_level: int | None = None  # DOCX 样式已给出层级时直接使用
    low_confidence: bool = False  # OCR 置信度低于阈值
    confidence: float | None = None


@dataclass
class ParsedPage:
    """一页的解析结果。"""

    page_no: int | None
    used_ocr: bool = False
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """整份文档的解析结果。"""

    path: Path
    doc_type: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def ocr_page_count(self) -> int:
        return sum(1 for page in self.pages if page.used_ocr)

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for page in self.pages for block in page.blocks)

    def iter_blocks(self) -> list[TextBlock]:
        """按阅读顺序展开全部块。"""
        return [block for page in self.pages for block in page.blocks]


@dataclass
class _Line:
    """页面里的一行文本，仅用于重建段落，不对外暴露。"""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, self.size, 1.0)


def load(path: str | Path) -> ParsedDocument:
    """按扩展名分派解析器，返回统一结构。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        document = _load_pdf(file_path)
    elif suffix in (".png", ".jpg", ".jpeg"):
        document = _load_image(file_path)
    elif suffix == ".docx":
        document = _load_docx(file_path)
    else:
        raise UnsupportedFormatError(f"暂不支持的文件格式：{suffix}")

    if document.char_count < 10:
        raise EmptyDocumentError("未能从文档中解析出任何文本")
    return document


def _is_cjk(char: str) -> bool:
    """判断字符是否属于中日韩文字或全角标点。"""
    return bool(char) and bool(_CJK_RE.match(char))


def _join_text(left: str, right: str) -> str:
    """中文之间直接拼接，西文之间补空格，避免把词粘死。"""
    if not left:
        return right
    if not right:
        return left
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    return f"{left} {right}"


def _is_inside(rect: fitz.Rect, box: fitz.Rect) -> bool:
    """线段中心点是否落在表格区域内。"""
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    return box.x0 <= cx <= box.x1 and box.y0 <= cy <= box.y1


# ---------------------------------------------------------------- PDF


def _load_pdf(path: Path) -> ParsedDocument:
    """逐页解析 PDF：文本页走文本层，扫描页走 OCR。"""
    pages: list[ParsedPage] = []
    with fitz.open(path) as pdf:
        page_lines: list[list[_Line]] = []
        page_tables: list[list[tuple[float, str]]] = []
        heights: list[float] = []

        for page in pdf:
            lines, tables = _scan_page(page)
            page_lines.append(lines)
            page_tables.append(tables)
            heights.append(page.rect.height)

        _drop_running_lines(page_lines, heights)

        for index, page in enumerate(pdf):
            page_no = index + 1
            blocks = _page_blocks(page_no, page_lines[index], page_tables[index])
            used_ocr = False

            raw_chars = len(page.get_text("text").strip())
            if raw_chars < SCANNED_PAGE_MAX_CHARS and page.get_images(full=True):
                ocr_paragraphs = _ocr_page(page)
                if ocr_paragraphs:
                    blocks = [
                        TextBlock(
                            text=paragraph.text,
                            page_no=page_no,
                            font_size=0.0,
                            kind="paragraph",
                            low_confidence=paragraph.low_confidence,
                            confidence=paragraph.confidence,
                        )
                        for paragraph in ocr_paragraphs
                    ]
                used_ocr = True

            pages.append(ParsedPage(page_no=page_no, used_ocr=used_ocr, blocks=blocks))

    return ParsedDocument(path=path, doc_type="pdf", pages=pages)


def _ocr_page(page: fitz.Page) -> list[ocr.OcrParagraph]:
    """把扫描页渲染成 PNG 后交给 OCR；临时文件用完即删。"""
    pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "page.png"
        pixmap.save(str(image_path))
        return ocr.recognize(image_path)


def _scan_page(page: fitz.Page) -> tuple[list[_Line], list[tuple[float, str]]]:
    """提取页面文本行与表格。

    只有通过结构校验的表格才按表格处理并转成 markdown；被误判的「警告框」等
    排版块会让位给普通文本行，避免整页文本被吞掉。
    """
    tables: list[tuple[float, str, fitz.Rect]] = []
    for table in page.find_tables().tables:
        rows = table.extract()
        if not _looks_like_table(rows):
            continue
        markdown = _table_to_markdown(rows)
        if markdown:
            box = fitz.Rect(table.bbox)
            tables.append((box.y0, markdown, box))

    table_boxes = [box for _, _, box in tables]
    lines: list[_Line] = []

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            # 页眉左右栏常被 PyMuPDF 并成一行，先按大间距切成独立片段，
            # 否则「章节名 + 手册名」会拼成一个每页都不同的字符串，逃过重复判定
            for segment in _split_segments(spans):
                text = ""
                for span in segment:
                    text = _join_text(text, span["text"].strip())
                # 项目符号常被 PDF 重复绘制，先剥掉前导符号；整行都是符号则丢弃
                text = text.lstrip(_BULLET_TRIM_CHARS)
                if not text:
                    continue

                x0 = min(span["bbox"][0] for span in segment)
                y0 = min(span["bbox"][1] for span in segment)
                x1 = max(span["bbox"][2] for span in segment)
                y1 = max(span["bbox"][3] for span in segment)
                rect = fitz.Rect(x0, y0, x1, y1)
                if any(_is_inside(rect, box) for box in table_boxes):
                    continue

                lines.append(
                    _Line(
                        text=text,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        size=max(span["size"] for span in segment),
                        bold=any(_span_is_bold(span) for span in segment),
                    )
                )
    return lines, [(top, markdown) for top, markdown, _ in tables]


def _split_segments(spans: list[dict]) -> list[list[dict]]:
    """按水平间距把一行拆成若干独立片段，间隔超过 LINE_GAP_PT 视为不相干。"""
    ordered = sorted(spans, key=lambda span: span["bbox"][0])
    segments: list[list[dict]] = [[ordered[0]]]
    for span, previous in zip(ordered[1:], ordered):
        gap = span["bbox"][0] - previous["bbox"][2]
        if gap > LINE_GAP_PT:
            segments.append([span])
        else:
            segments[-1].append(span)
    return segments


def _span_is_bold(span: dict) -> bool:
    """PyMuPDF 的 flag 第 5 位表示粗体；部分中文字体改用字体名标记。"""
    font_name = str(span.get("font", "")).lower()
    return bool(span.get("flags", 0) & 16) or "bold" in font_name or "heavy" in font_name


def _drop_running_lines(pages: list[list[_Line]], heights: list[float]) -> None:
    """删除页眉、页脚与页码：位置在页面上下边缘且跨页重复出现。

    两个必须避开的坑：
    ① 章标题有时就排在页面顶部的页眉带里（如 CP600 的「2 产品信息」排在 y=37.6），
       且与页眉文字相同，只按「重复」判定会把真标题一起删掉 → 用字号豁免；
    ② 手册常奇偶页交替印「章节名 / 手册名」，每个变体全书只出现 2 次，
       高阈值会让它逃过剔除、混进正文块 → 短文本允许 2 次即判定。
    """
    body_size = _body_size([line for lines in pages for line in lines])

    counter: Counter[str] = Counter()
    for lines, height in zip(pages, heights):
        for line in lines:
            if _in_margin(line, height) and not _is_large_text(line, body_size):
                counter[_normalize_margin_text(line.text)] += 1

    repeated = {
        text
        for text, count in counter.items()
        if text
        and (
            count >= RUNNING_REPEAT_HARD
            or (count >= RUNNING_REPEAT_SOFT and len(text) <= RUNNING_SHORT_TEXT)
        )
    }

    for lines, height in zip(pages, heights):
        lines[:] = [
            line
            for line in lines
            if not _is_running_line(line, height, repeated, body_size)
        ]


def _body_size(lines: list[_Line]) -> float:
    """统计正文字号：取长度 >=20 字的行里出现次数最多的字号。"""
    counter: Counter[float] = Counter()
    for line in lines:
        if len(line.text) >= 20:
            counter[round(line.size * 2) / 2] += 1
    if not counter:
        return 0.0
    return counter.most_common(1)[0][0]


def _is_large_text(line: _Line, body_size: float) -> bool:
    """字号明显大于正文的行视为标题或图表文字，不参与页眉页脚剔除。"""
    if body_size <= 0:
        return False
    return line.size >= body_size + RUNNING_SIZE_SLACK


def _in_margin(line: _Line, page_height: float) -> bool:
    """行是否位于页面上下 8% 的边缘区域。"""
    return line.y0 < page_height * 0.08 or line.y1 > page_height * 0.92


def _normalize_margin_text(text: str) -> str:
    """边缘文本归一化：去空白、去数字，页码差异不影响重复判定。"""
    compact = re.sub(r"\s+", "", text)
    return re.sub(r"\d+", "#", compact)


def _is_running_line(
    line: _Line, page_height: float, repeated: set[str], body_size: float
) -> bool:
    if not _in_margin(line, page_height):
        return False
    text = line.text.strip()
    # 页码优先判断：页码常有 11pt 等较大字号，不能被字号豁免放过去
    if _PAGE_NUMBER_RE.match(text):
        return True
    if _is_large_text(line, body_size):
        return False
    return _normalize_margin_text(text) in repeated


def _page_blocks(
    page_no: int, lines: list[_Line], tables: list[tuple[float, str]]
) -> list[TextBlock]:
    """把行合并成段落，并在正确位置插入整张表格的 markdown。"""
    elements: list[tuple[float, int, TextBlock]] = []

    for top, block in _merge_paragraphs(lines):
        block.page_no = page_no
        elements.append((top, 0, block))

    for top, markdown in tables:
        elements.append((top, 1, TextBlock(text=markdown, page_no=page_no, kind="table")))

    elements.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in elements]


def _merge_paragraphs(lines: list[_Line]) -> list[tuple[float, TextBlock]]:
    """按行距、字号、缩进把行合并成自然段落。"""
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (round(line.y0, 1), line.x0))
    groups: list[list[_Line]] = []
    current: list[_Line] = []

    for line in ordered:
        if not current:
            current = [line]
            continue

        previous = current[-1]
        gap = line.y0 - previous.y1
        tolerance = max(previous.height, line.size) * 0.9
        starts_new = (
            gap > tolerance
            or abs(line.size - previous.size) > 0.7
            or previous.text.endswith(("。", "！", "？", "；", "."))
            or abs(line.x0 - current[0].x0) > line.size * 1.5
        )
        if starts_new:
            groups.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        groups.append(current)

    blocks: list[tuple[float, TextBlock]] = []
    for group in groups:
        text = ""
        for line in group:
            text = _join_text(text, line.text.strip())
        if not text:
            continue
        blocks.append(
            (
                group[0].y0,
                TextBlock(
                    text=text,
                    page_no=None,
                    font_size=max(line.size for line in group),
                    bold=all(line.bold for line in group),
                    kind="paragraph",
                ),
            )
        )
    return blocks


def _looks_like_table(rows: list[list[str | None]]) -> bool:
    """结构校验：至少两列有内容、单元格以短文本为主。

    设备手册里常见的「警告/危险」提示框也会被 find_tables 识别为表格，
    但实际只有一列有字、且单元格是整段说明，这里据此排除。
    """
    if len(rows) < 2:
        return False

    width = max(len(row) for row in rows)
    if width < 2:
        return False

    cleaned = [[(cell or "").strip() for cell in row] for row in rows]

    filled_columns = 0
    for column in range(width):
        values = [row[column] for row in cleaned if column < len(row)]
        non_empty = [value for value in values if value]
        if non_empty and len(non_empty) >= max(1, len(values) * 0.4):
            filled_columns += 1
    if filled_columns < 2:
        return False

    lengths = sorted(len(value) for row in cleaned for value in row if value)
    if not lengths:
        return False
    return lengths[len(lengths) // 2] <= 40


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """把表格整体转成 markdown，保证不拆散。"""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [re.sub(r"\s*\n\s*", " ", (cell or "")).strip().replace("|", "\\|") for cell in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < 2:
        return ""

    width = max(len(row) for row in cleaned)
    if width < 2:
        return ""
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
    return "\n".join([header, separator, *body])


# ---------------------------------------------------------------- 图片


def _load_image(path: Path) -> ParsedDocument:
    """整张图片走 OCR，视为一页。"""
    paragraphs = ocr.recognize(path)
    blocks = [
        TextBlock(
            text=paragraph.text,
            page_no=1,
            font_size=0.0,
            low_confidence=paragraph.low_confidence,
            confidence=paragraph.confidence,
        )
        for paragraph in paragraphs
    ]
    return ParsedDocument(
        path=path, doc_type="image", pages=[ParsedPage(page_no=1, used_ocr=True, blocks=blocks)]
    )


# ---------------------------------------------------------------- DOCX


def _load_docx(path: Path) -> ParsedDocument:
    """按正文顺序读取 DOCX 的段落与表格；页号无法获得，置空。"""
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    docx = DocxDocument(str(path))
    blocks: list[TextBlock] = []

    for child in docx.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, docx)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph.style.name if paragraph.style else "")
            blocks.append(
                TextBlock(
                    text=text,
                    page_no=None,
                    font_size=16.0 - level * 2 if level else 0.0,
                    heading_level=level,
                )
            )
        elif tag == "tbl":
            markdown = _docx_table_to_markdown(Table(child, docx))
            if markdown:
                blocks.append(TextBlock(text=markdown, page_no=None, kind="table"))

    return ParsedDocument(
        path=path, doc_type="docx", pages=[ParsedPage(page_no=None, blocks=blocks)]
    )


def _docx_heading_level(style_name: str) -> int | None:
    """从样式名识别标题层级，兼容中文 Word 的「标题 1」。"""
    match = re.search(r"(?:heading|标题)\s*(\d)", style_name, re.IGNORECASE)
    if match:
        return max(1, min(4, int(match.group(1))))
    return None


def _docx_table_to_markdown(table: object) -> str:
    """python-docx 表格转 markdown。"""
    rows: list[list[str | None]] = []
    for row in table.rows:  # type: ignore[attr-defined]
        rows.append([cell.text for cell in row.cells])
    return _table_to_markdown(rows)
