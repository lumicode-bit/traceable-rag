"""检索层对外唯一入口。

对外只有两个函数：
- `search(query, top_k, mode) -> list[Chunk]`：规格要求的统一接口；
- `search_with_ranks(...) -> list[SearchHit]`：带两路原始排名的版本，供 bench / 前端调试用。

底层模块各管一件事：`vector.py` 语义召回、`bm25.py` 词法召回、`hybrid.py` 融合；
重排在 `app/llm/rerank.py`（它是一次远端模型调用，与生成层同属「模型接入」）。
本模块只做编排与结果装配，不写检索算法。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.llm import rerank
from app.models import Chunk
from app.retrieval import bm25, hybrid, vector

# 可选模式；顺序即 bench 的输出顺序
MODES: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank")

__all__ = [
    "MODES",
    "InvalidModeError",
    "SearchHit",
    "search",
    "search_with_ranks",
    "rebuild_indexes",
]


class InvalidModeError(ValueError):
    """传入了不支持的检索模式。"""


@dataclass
class SearchHit:
    """一条检索结果：块本体 + 各阶段得分与排名。"""

    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_id(self) -> int:
        return int(self.chunk.id)

    @property
    def sources(self) -> str:
        """来源标记：V=仅向量命中，B=仅 BM25 命中，VB=两路都命中。"""
        mark = ""
        if self.vector_rank is not None:
            mark += "V"
        if self.bm25_rank is not None:
            mark += "B"
        return mark or "—"

    @property
    def citation(self) -> str:
        """引用串：文档名 + 页码 + 章节路径，回答时必须带上它。"""
        name = self.chunk.document.filename if self.chunk.document else "未知文档"
        page = f" 第 {self.chunk.page_no} 页" if self.chunk.page_no else ""
        section = f" {self.chunk.section}" if self.chunk.section else ""
        return f"{name}{page}{section}".strip()


def search(query: str, top_k: int = 5, mode: str = "hybrid") -> list[Chunk]:
    """规格要求的统一接口：返回排好序的块列表。

    自己开短会话，调用方不需要关心数据库；返回的块已预加载所属文档，会话关闭后仍可读。
    """
    hits = search_with_ranks(query, top_k=top_k, mode=mode)
    return [hit.chunk for hit in hits]


def search_with_ranks(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
    *,
    session: Session | None = None,
    candidate_k: int | None = None,
) -> list[SearchHit]:
    """与 `search` 同义，但保留向量排名、BM25 排名、融合排名，用于分析与调参。"""
    if mode not in MODES:
        raise InvalidModeError(
            f"不支持的检索模式：{mode}（可选：{'、'.join(MODES)}）"
        )
    if top_k <= 0:
        return []
    if session is not None:
        return _execute(session, query, top_k, mode, candidate_k)
    with SessionLocal() as own_session:
        return _execute(own_session, query, top_k, mode, candidate_k)


def _execute(
    session: Session, query: str, top_k: int, mode: str, candidate_k: int | None
) -> list[SearchHit]:
    """按模式分派；所有模式最终都装配成 SearchHit 列表。"""
    if mode == "vector":
        hits = vector.search(session, query, top_k)
        chunks = _load_chunks(session, [chunk_id for chunk_id, _ in hits])
        return [
            SearchHit(
                chunk=chunks[chunk_id],
                score=score,
                vector_rank=rank,
                vector_score=score,
            )
            for rank, (chunk_id, score) in enumerate(hits, start=1)
            if chunk_id in chunks
        ]

    depth = candidate_k or settings.retrieval_candidates
    fused = hybrid.search(session, query, depth, candidate_k=depth)
    chunks = _load_chunks(session, [hit.chunk_id for hit in fused])

    ranked: list[SearchHit] = [
        SearchHit(
            chunk=chunks[item.chunk_id],
            score=item.score,
            vector_rank=item.vector_rank,
            bm25_rank=item.bm25_rank,
            fused_rank=rank,
            vector_score=item.vector_score,
            bm25_score=item.bm25_score,
        )
        for rank, item in enumerate(fused, start=1)
        if item.chunk_id in chunks
    ]

    if mode == "hybrid":
        return ranked[:top_k]

    # hybrid_rerank：把融合后的前 N 个候选交给远端重排，再截 top_k
    candidates = ranked[: rerank.candidate_limit()]
    outcome = rerank.rerank(
        query, [(hit.chunk_id, hit.chunk.content) for hit in candidates], top_k
    )
    by_id = {hit.chunk_id: hit for hit in candidates}
    results: list[SearchHit] = []
    for chunk_id, rerank_score in outcome.items:
        hit = by_id[chunk_id]
        hit.rerank_score = rerank_score
        # 降级（未配置 / 接口失败 / 关闭）时分数为 None，排序仍是融合原序，
        # 把原因留在结果里，避免 bench 把「没重排」当成「重排无效」
        hit.extra["rerank"] = 0.0 if outcome.degraded else 1.0
        if outcome.reason and outcome.degraded:
            hit.extra["rerank_degraded"] = 1.0
        if rerank_score is not None:
            hit.score = float(rerank_score)
        results.append(hit)
    return results


def _load_chunks(session: Session, chunk_ids: list[int]) -> dict[int, Chunk]:
    """按 ID 批量取块，返回 id → Chunk 的映射，保持调用方给定的顺序。"""
    if not chunk_ids:
        return {}
    rows = session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids))).scalars().all()
    return {int(chunk.id): chunk for chunk in rows}


def rebuild_indexes(session: Session | None = None, *, force_embedding: bool = False) -> dict[str, int]:
    """重建两路索引：补齐缺失向量 + 重建 BM25，返回各自的处理条数。"""
    if session is not None:
        return _rebuild(session, force_embedding)
    with SessionLocal() as own_session:
        return _rebuild(own_session, force_embedding)


def _rebuild(session: Session, force_embedding: bool) -> dict[str, int]:
    embedded = vector.index_chunks(session, force=force_embedding)
    index = bm25.rebuild(session)
    return {"embedded": embedded, "bm25_chunks": index.chunk_count}
