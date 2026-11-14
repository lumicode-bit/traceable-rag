"""OCR 封装：RapidOCR（onnxruntime）中文识别，输出文本与置信度，低置信度段落打标记。

为什么不用 PaddleOCR：paddlepaddle 依赖几百 MB 且与 torch/onnxruntime 常打架。
RapidOCR 后端是 onnxruntime，纯 CPU、约 80MB、Windows/macOS/Linux 一致，
识别质量对「扫描版手册正文」这个场景够用。

职责边界：只做「图片 → 文本 + 置信度」，不关心文件是 PDF 还是图片，
也不做切块与入库。引擎按需加载，未安装时抛出明确异常，绝不静默返回空文本。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 低于该置信度的段落会在正文前加标记，便于下游人工复核
LOW_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_MARK = "【OCR低置信度 {score:.2f}】"

_CJK_RANGES = ((0x3400, 0x9FFF), (0x3000, 0x303F), (0xFF00, 0xFFEF))


class OcrUnavailableError(RuntimeError):
    """RapidOCR 未安装或引擎加载失败。"""


@dataclass
class OcrParagraph:
    """OCR 识别出的一个段落。"""

    text: str
    confidence: float
    low_confidence: bool


@dataclass
class _OcrLine:
    """OCR 识别出的一行，用于按行距重组段落。"""

    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


_engine: Any = None
_engine_lock = threading.Lock()


def _get_engine() -> Any:
    """惰性创建 RapidOCR 实例；首次调用会加载内置的检测/识别 onnx 模型。"""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # 依赖缺失必须显式失败
            raise OcrUnavailableError(
                "未安装 rapidocr-onnxruntime，无法识别扫描页；"
                "请执行 pip install rapidocr-onnxruntime"
            ) from exc
        try:
            _engine = RapidOCR()
        except Exception as exc:  # 模型文件损坏、onnxruntime 版本不兼容都走这里
            raise OcrUnavailableError(f"加载 RapidOCR 失败：{type(exc).__name__}: {exc}") from exc
        return _engine


def recognize(image_path: str | Path) -> list[OcrParagraph]:
    """识别一张图片，返回按段落聚合的文本与置信度。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"待识别图片不存在：{path}")

    engine = _get_engine()
    lines = _extract_lines(_predict(engine, path))
    if not lines:
        return []
    return _group_paragraphs(lines)


def _predict(engine: Any, path: Path) -> Any:
    """调用 RapidOCR：返回 (结果列表, 耗时列表)，结果可能是 None。"""
    output = engine(str(path))
    if isinstance(output, tuple):
        return output[0]
    return output


def _extract_lines(result: Any) -> list[_OcrLine]:
    """把 RapidOCR 的 `[[box, text, score], ...]` 统一成 _OcrLine 列表。"""
    lines: list[_OcrLine] = []
    for entry in result or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        box, text, score = entry[0], entry[1], entry[2]
        content = str(text).strip()
        if not content:
            continue
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            confidence = 0.0
        lines.append(_OcrLine(content, confidence, *_normalize_box(_flatten(box))))
    return lines


def _flatten(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for element in value:
            result.extend(_flatten(element))
        return result
    return []


def _normalize_box(coords: list[float]) -> tuple[float, float, float, float]:
    """支持 [x0, y0, x1, y1] 与四点多边形两种形态。"""
    if len(coords) == 4:
        x0, y0, x1, y1 = coords
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if len(coords) >= 8:
        xs = coords[0::2]
        ys = coords[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    return (0.0, 0.0, 0.0, 0.0)


def _group_paragraphs(lines: list[_OcrLine]) -> list[OcrParagraph]:
    """按行距把 OCR 行聚成段落，并按平均置信度打低置信度标记。"""
    ordered = sorted(lines, key=lambda line: (round(line.y0, 1), line.x0))
    groups: list[list[_OcrLine]] = []
    current: list[_OcrLine] = []

    for line in ordered:
        if current:
            previous = current[-1]
            gap = line.y0 - previous.y1
            if gap > max(6.0, previous.height * 0.8):
                groups.append(current)
                current = [line]
                continue
        current.append(line)
    if current:
        groups.append(current)

    paragraphs: list[OcrParagraph] = []
    for group in groups:
        text = ""
        for line in group:
            text = _join(text, line.text)
        text = text.strip()
        if not text:
            continue
        confidence = sum(line.confidence for line in group) / len(group)
        low = confidence < LOW_CONFIDENCE_THRESHOLD
        if low:
            text = LOW_CONFIDENCE_MARK.format(score=confidence) + text
        paragraphs.append(OcrParagraph(text=text, confidence=confidence, low_confidence=low))
    return paragraphs


def _join(left: str, right: str) -> str:
    """中文直接拼接，西文补空格。"""
    if not left:
        return right
    if not right:
        return left
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    return f"{left} {right}"


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)
