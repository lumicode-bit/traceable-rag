"""向量检索：远端 embedding 编码 + numpy 全量点积。

规模账（决定了不需要 pgvector）：3 万块 × 512 维 float32 只有 60MB，
一次全量点积在一台普通笔记本上是 10~30 毫秒，而装一套 Postgres + pgvector
在这个量级上的收益是零。超过 `VECTOR_WARN_CHUNKS`（默认 5 万块）会打印警告，
提示该换索引了——那时再换，检索入口仍然只有 `search()` 一个函数。

存储与检索的两点约定：

- `chunks.embedding` 存**归一化后的 float32 原始字节**（`np.float32.tobytes()`），
  SQLite BLOB 与 Postgres BYTEA 都能直接放下，不需要任何扩展类型；
- 矩阵常驻内存，按「有向量的块数」失效；`index_chunks()` 写完会主动清缓存，
  避免刚补完向量却检索到旧矩阵。

`chunks.embed_model` 记录生成向量的模型。换模型后旧向量不会被混进新矩阵
（维度相同也不混），而是被跳过并提示重建——这是最容易静默出错的地方。
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.llm import embed
from app.llm.embed import EmbeddingUnavailableError  # 对外仍从这个模块导出，方便调用方捕获
from app.models import Chunk

__all__ = [
    "EmbeddingUnavailableError",
    "index_chunks",
    "load_index",
    "reset_index",
    "search",
]

logger = logging.getLogger(__name__)


@dataclass
class _VectorIndex:
    """内存里的向量索引快照。"""

    ids: np.ndarray
    matrix: np.ndarray
    count: int
    skipped: int = 0
    extra: dict[str, int] = field(default_factory=dict)


_index: _VectorIndex | None = None
_index_lock = threading.Lock()
_warned_size = False


def load_index(session: Session) -> _VectorIndex:
    """取当前索引；未构建或块数变化时重建（与 BM25 索引同一套策略）。"""
    global _index
    current = _index
    if current is not None and current.count == _embedded_count(session):
        return current
    with _index_lock:
        if _index is not None and _index.count == _embedded_count(session):
            return _index
        _index = _build(session)
        return _index


def reset_index() -> None:
    """丢弃内存索引，下一次检索时重建。"""
    global _index
    with _index_lock:
        _index = None


def search(session: Session, query: str, top_k: int) -> list[tuple[int, float]]:
    """向量召回，返回 [(chunk_id, 余弦相似度)]，按相似度降序。"""
    index = load_index(session)
    if index.ids.size == 0:
        return []
    vector = embed.encode_query(query)
    if vector.shape[0] != index.matrix.shape[1]:
        raise EmbeddingUnavailableError(
            f"查询向量维度（{vector.shape[0]}）与库内向量维度（{index.matrix.shape[1]}）不一致；"
            "换模型后需要重建索引"
        )
    scores = index.matrix @ vector
    order = np.argsort(-scores)[:top_k]
    return [(int(index.ids[i]), float(scores[i])) for i in order]


def index_chunks(session: Session, *, force: bool = False) -> int:
    """为缺失（或模型不一致）的块生成向量，返回本次写入的块数。

    `force=True` 忽略已有向量全部重算，用于换模型或换切块策略后的重建。
    """
    model = embed.model_name()
    statement = select(Chunk.id, Chunk.section, Chunk.content)
    if not force:
        statement = statement.where(
            or_(
                Chunk.embedding.is_(None),
                Chunk.embed_model.is_(None),
                Chunk.embed_model != model,
            )
        )
    rows = session.execute(statement.order_by(Chunk.id)).all()
    if not rows:
        return 0

    # 编码时把章节路径拼进正文，让「所属小节」也参与语义匹配
    texts = [_document_text(section, content) for _, section, content in rows]
    vectors = embed.encode_documents(texts, session=session)

    table = Chunk.__table__
    for (chunk_id, _, _), vector in zip(rows, vectors):
        session.execute(
            table.update()
            .where(table.c.id == chunk_id)
            .values(embedding=vector.astype(np.float32).tobytes(), embed_model=model)
        )
    session.commit()
    reset_index()
    return len(rows)


def _embedded_count(session: Session) -> int:
    return int(
        session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
        ).scalar_one()
    )


def _build(session: Session) -> _VectorIndex:
    """把全部向量读进内存拼成矩阵；模型不一致或维度不一致的行跳过并计数。"""
    global _warned_size

    model = embed.model_name()
    rows = session.execute(
        select(Chunk.id, Chunk.embedding, Chunk.embed_model).where(Chunk.embedding.isnot(None))
    ).all()

    ids: list[int] = []
    vectors: list[np.ndarray] = []
    skipped_model = 0
    skipped_dim = 0
    dim: int | None = None

    for chunk_id, blob, row_model in rows:
        if row_model != model:
            skipped_model += 1
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if dim is None:
            dim = int(vector.shape[0])
        if vector.shape[0] != dim:
            skipped_dim += 1
            continue
        ids.append(int(chunk_id))
        vectors.append(vector)

    if skipped_model:
        logger.warning(
            "%d 个块的向量由其它模型生成（当前 %s），已跳过；请执行 "
            "python -m app.retrieval.bench --rebuild 重建索引",
            skipped_model,
            model,
        )
    if skipped_dim:
        logger.warning("%d 个块的向量维度与当前索引不一致，已跳过；请重建索引", skipped_dim)

    if not vectors:
        return _VectorIndex(
            ids=np.zeros(0, dtype=np.int64),
            matrix=np.zeros((0, max(1, embed.dim())), dtype=np.float32),
            count=_embedded_count(session),
            skipped=skipped_model + skipped_dim,
        )

    matrix = np.vstack(vectors).astype(np.float32)
    if matrix.shape[0] > settings.vector_warn_chunks and not _warned_size:
        _warned_size = True
        logger.warning(
            "向量索引已达 %d 块，超过 %d 的建议上限：全量点积开始变慢，"
            "该考虑 pgvector / FAISS 了",
            matrix.shape[0],
            settings.vector_warn_chunks,
        )

    return _VectorIndex(
        ids=np.asarray(ids, dtype=np.int64),
        matrix=matrix,
        count=_embedded_count(session),
        skipped=skipped_model + skipped_dim,
        extra={"model_skipped": skipped_model, "dim_skipped": skipped_dim},
    )


def _document_text(section: str | None, content: str) -> str:
    """文档侧的编码文本：章节路径在前、正文在后；正文已含标题时不重复拼接。"""
    if not section:
        return content
    if content.startswith(section):
        return content
    return f"{section}\n{content}"
