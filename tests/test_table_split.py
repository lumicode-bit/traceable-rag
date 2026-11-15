"""长表格按行拆块：每块必须带表头，且不丢行。

背景：这份语料 203 块里 147 块含表格，最长一块 2922 字是一整张参数表。
整块入库时向量被几十行「平均」，语义检索排不上来；拆开并复制表头后，
「某一行」才有机会被命中，同时模型看到的仍是「表头 + 若干行」。
"""
from __future__ import annotations

from app.ingest.chunker import split_table

HEADER = "| 型号 | 额定电流(A) | 线径(mm2) |"
SEPARATOR = "| --- | --- | --- |"


def _table(rows: int) -> str:
    body = "\n".join(f"| CP600-4T{i}G | {i * 2}.0 | {i * 2 + 1} |" for i in range(1, rows + 1))
    return f"表4-2 选型指导\n{HEADER}\n{SEPARATOR}\n{body}\n注：仅示例"


def test_short_table_is_not_split() -> None:
    text = _table(3)

    assert split_table(text, limit=1200) == [text]


def test_long_table_is_split_with_header_repeated() -> None:
    text = _table(120)

    pieces = split_table(text, limit=600)

    assert len(pieces) > 1
    for index, piece in enumerate(pieces):
        assert HEADER in piece, f"第 {index} 段丢了表头"
        assert SEPARATOR in piece, f"第 {index} 段丢了分隔行"


def test_no_row_is_lost() -> None:
    text = _table(120)
    rows = [line for line in text.splitlines() if line.startswith("| CP600")]
    pieces = split_table(text, limit=600)

    collected = [
        line for piece in pieces for line in piece.splitlines() if line.startswith("| CP600")
    ]
    assert collected == rows, "拆块时丢了行或改了顺序"


def test_caption_stays_on_first_piece_only() -> None:
    text = _table(120)

    pieces = split_table(text, limit=600)

    assert "表4-2 选型指导" in pieces[0]
    assert all("表4-2 选型指导" not in piece for piece in pieces[1:])


def test_piece_stays_within_limit_plus_header() -> None:
    text = _table(120)
    pieces = split_table(text, limit=600)

    header_len = len(HEADER) + len(SEPARATOR) + 2
    for piece in pieces:
        # 每段上限 = limit（拆行时的预算已含表头长度），留一点余量给换行
        assert len(piece) <= 600 + header_len + 40
