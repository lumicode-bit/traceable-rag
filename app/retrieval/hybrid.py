"""混合检索：用 RRF（Reciprocal Rank Fusion）融合向量与 BM25 两路结果。

为什么用 RRF 而不是加权分数相加：两路的分数尺度完全不同（余弦相似度落在 0~1，
BM25 可以到几十甚至上百），直接相加会让某一路天然主导。RRF 只看排名不看分数，
`score = Σ 1 / (k + rank)`，k 取经验值 60，对尺度不敏感，也不需要调权重。

融合结果**保留每一路的原始排名**，用于分析「这题是向量救回来的还是 BM25 救回来的」。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document
from app.retrieval import bm25, vector

# 查询里的「确切标识符」：f4-01 / cp600-4t22g / err23 / di2 / a01
# 与 bm25 的分词规则保持同族，但这里要的是「整串出现」，不是切词
_IDENTIFIER = re.compile(r"[a-z]{1,3}\d{1,3}(?:-\d{1,3})+|[a-z]{2,4}\d{2,4}[a-z0-9-]*")


def _vector_side(query: str, depth: int) -> list[tuple[int, float]]:
    """向量那一路：自己开 Session，避免跨线程共享 Session。"""
    with SessionLocal() as own_session:
        return vector.search(own_session, query, depth)


def _bm25_side(query: str, depth: int) -> list[tuple[int, float]]:
    """BM25 那一路：同上。纯内存计算，通常几十微秒就返回。"""
    with SessionLocal() as own_session:
        return bm25.search(own_session, query, depth)


@dataclass
class FusedHit:
    """融合后的一个候选，携带两路原始排名。"""

    chunk_id: int
    score: float
    vector_rank: int | None
    bm25_rank: int | None
    vector_score: float | None = None
    bm25_score: float | None = None
    # 调试用：strong_match=1 表示这块是靠「确切标识符全命中」提上来的
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> str:
        """来源标记：V=向量命中，B=BM25 命中，VB=两路都命中。"""
        mark = ""
        if self.vector_rank is not None:
            mark += "V"
        if self.bm25_rank is not None:
            mark += "B"
        return mark


def rrf_fuse(
    rankings: dict[str, list[int]], *, k: int | None = None
) -> list[FusedHit]:
    """把多路排名融合成一条有序列表。

    rankings 的 key 就是路名（如 "vector" / "bm25"），value 是按相关性降序的 chunk_id。
    """
    smooth = settings.rrf_k if k is None else k
    scores: dict[int, float] = {}
    detail: dict[int, dict[str, int]] = {}

    for route, ranked in rankings.items():
        for position, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (smooth + position)
            detail.setdefault(chunk_id, {})[route] = position

    hits = [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            vector_rank=detail[chunk_id].get("vector"),
            bm25_rank=detail[chunk_id].get("bm25"),
        )
        for chunk_id, score in scores.items()
    ]
    # 分数相同时按 chunk_id 升序，保证结果稳定可复现
    hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return hits


def search(
    session: Session, query: str, top_k: int, *, candidate_k: int | None = None
) -> list[FusedHit]:
    """两路各取 candidate_k 个候选，RRF 融合后返回前 top_k。

    两路**并发**取：向量那一路要等一次远端 embedding 往返（约 350ms），BM25 是纯内存计算。
    串行时总耗时是 sum，并发后是 max——稳态下省不了几毫秒（BM25 本来就快），
    但冷启动和向量接口抖动时，串行会把两者的等待叠起来，这正是 P95 被放大的原因之一。

    每个线程开自己的 Session：SQLAlchemy 的 Session 不是线程安全的，共享会出难查的错。
    两边读的是同一个库文件 / 同一个连接池，短事务并发读在 SQLite 的 WAL 下没问题。
    """
    depth = candidate_k or settings.retrieval_candidates
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as pool:
        vector_job = pool.submit(_vector_side, query, depth)
        bm25_job = pool.submit(_bm25_side, query, depth)
        # 先取向量结果：它最慢，早一点抛异常能省掉等另一路的时间
        vector_hits = vector_job.result()
        bm25_hits = bm25_job.result()

    fused = rrf_fuse(
        {
            "vector": [chunk_id for chunk_id, _ in vector_hits],
            "bm25": [chunk_id for chunk_id, _ in bm25_hits],
        }
    )

    vector_scores = dict(vector_hits)
    bm25_scores = dict(bm25_hits)
    for hit in fused:
        hit.vector_score = vector_scores.get(hit.chunk_id)
        hit.bm25_score = bm25_scores.get(hit.chunk_id)

    fused = _promote_strong_matches(session, query, fused)
    return fused[:top_k]


def _promote_strong_matches(
    session: Session, query: str, fused: list[FusedHit]
) -> list[FusedHit]:
    """强匹配通道：查询里带着确切标识符（F4-01 / CP600-4T22G / DI2）时，
    同时包含这些标识符的块提到最前。

    为什么需要它：RRF 只看名次。q029 的金标块在 BM25 里排第 3、向量里排到几十名之外，
    融合后两个「两路都中等靠前」的参数表块就把它挤到第 6。可它明明**同时含有 F4-01 与 DI2**
    ——这种「确切标识符全部命中」的证据，比名次更硬。

    两个约束，避免这一招变成噪声源：
    - 只在命中的块**不超过 3 个**时生效（命中一大片说明这个标识符没有区分度，例如 A01）；
    - 只在两路候选里找（不扫全库），代价可忽略。
    """
    identifiers = _query_identifiers(query)
    if len(identifiers) < 2 or len(fused) < 2:
        return fused

    chunk_ids = [hit.chunk_id for hit in fused]
    rows = session.execute(
        select(Chunk.id, Chunk.content, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    # 文档名也算进可匹配文本：问「CP600 手册里 A01 是哪年改的」时，
    # 页码块正文里往往只有「A01」，型号只出现在文档名上。
    texts = {chunk_id: f"{filename} {content}".lower() for chunk_id, content, filename in rows}

    strong: list[FusedHit] = []
    for hit in fused:
        text = texts.get(hit.chunk_id) or ""
        if text and all(identifier in text for identifier in identifiers):
            strong.append(hit)
    if not strong or len(strong) > 3 or len(strong) == len(fused):
        return fused

    rest = [hit for hit in fused if hit not in strong]
    for hit in strong:
        hit.extra["strong_match"] = 1.0
    return strong + rest


def _query_identifiers(query: str) -> list[str]:
    """抽出查询里**够长、够独特**的标识符，用于强匹配。

    直接复用 BM25 的分词器，保证「查询里认得的标识符」与「库里能匹配上的 token」
    是同一套规则——两边各写一套正则，迟早对不上。
    只留同时含字母与数字、且长度 ≥3 的：`f4-01`、`cp600`、`di2`、`a01` 留下；
    `u1`、`01`、`的` 这类要么太短、要么纯数字，命中一大片反而没用。
    """
    found: list[str] = []
    for token in bm25.tokenize(query):
        if len(token) >= 3 and any(ch.isdigit() for ch in token) and any(
            ch.isalpha() for ch in token
        ):
            if token not in found:
                found.append(token)
    return found
