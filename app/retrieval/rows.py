"""表格行级索引：解决「整块是几十行的平均方向」的问题。

背景：203 块里 147 块含表格，最长一块 2922 字是一整张参数表。用户问「U0-74 的通信地址」，
整块与问题的相似度低于任何一段散文——因为向量是几十行的均值，那一行的信号被稀释掉了。
拆块不是解法（实测更差，见 DESIGN），剩下的是「行级检索、块级返回」：

- 把每个表格块里的每一行当成一个可检索单元（`| U0-74 | 电机实际输出转矩 | 0.1% | 7046H |`）；
- 行级 BM25 命中后，把它所属的**整块**作为一个新来源加进 RRF 融合；
- 最终返回给模型、切块界面上展示的仍然是整块——模型看到一个完整的表头和上下文，
  不会出现「有数据没表头」。

只做 BM25，不做行级向量：行很短，向量对短文本的区分度差；而参数名/功能码这类
恰好是词法匹配最擅长的。代价是一张常驻内存的小索引（约 3000 行，几 MB）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.retrieval.bm25 import tokenize


@dataclass
class RowIndex:
    """表格行的内存索引。`chunk_ids[r]` 是该行所属的块，`texts[r]` 是归一化后的行原文。"""

    chunk_ids: list[int]
    texts: list[str]
    bm25: BM25Okapi
    row_count: int
    chunk_count: int
    built_at: float = 0.0


_index: RowIndex | None = None
_lock = threading.Lock()


def reset() -> None:
    """测试与重建索引用。"""
    global _index
    with _lock:
        _index = None


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    # 分隔行（| --- | --- |）没有内容，不参与匹配
    return stripped.startswith("|") and not set(stripped) <= set("|-: 　")


def _build(session: Session) -> RowIndex:
    rows: list[list[str]] = []
    owners: list[int] = []
    texts: list[str] = []
    recorded = session.execute(select(func.count()).select_from(Chunk)).scalar_one()

    for chunk_id, content in session.execute(select(Chunk.id, Chunk.content)).all():
        lines = [line for line in content.splitlines() if _is_table_row(line)]
        if len(lines) < 3:  # 太短的不当表格（一两行的行内表格没有检索价值）
            continue
        header = lines[0]
        for line in lines[1:]:
            # 表头**不**拼进每一行：拼进去会让同一张表的所有行共享「功能码/名称/通讯地址」
            # 这些词，任何问参数的查询都会命中整张表的每一行，行级信号被摊平（实测反而更差）。
            tokens = tokenize(line)
            if tokens:
                rows.append(tokens)
                owners.append(chunk_id)
                texts.append("".join(line.split()).lower())

    bm25 = BM25Okapi(rows) if rows else BM25Okapi([["空"]])
    return RowIndex(
        chunk_ids=owners,
        texts=texts,
        bm25=bm25,
        row_count=len(rows),
        chunk_count=recorded,
        built_at=time.time(),
    )


def search(
    session: Session, query: str, depth: int, *, must_contain: list[str] | None = None
) -> list[tuple[int, float]]:
    """返回 (chunk_id, 行级最高分) 列表，按分数降序、按块去重。

    `must_contain` 是查询里的确切标识符（`U0-74`、`F4-01`）。给了就**只保留含其中至少一个的行**——
    这是把「行级匹配」从「词面相似」收紧成「确切的标识符对上」。不加这个约束时，
    任何问参数的查询都会命中整张表的每一行（表头与通用词太多），实测把召回从 0.938 拉到 0.729。
    """
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = _build(session)

    tokens = tokenize(query)
    if not tokens or not _index.row_count:
        return []

    needles = [item.lower() for item in (must_contain or [])]
    scores = _index.bm25.get_scores(tokens)
    best: dict[int, float] = {}
    for index, score in enumerate(scores):
        if score <= 0:
            continue
        if needles and not any(needle in _index.texts[index] for needle in needles):
            continue
        chunk_id = _index.chunk_ids[index]
        if score > best.get(chunk_id, 0.0):
            best[chunk_id] = float(score)
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
    return ranked[:depth]
