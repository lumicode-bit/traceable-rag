"""ORM 模型：documents / chunks / embedding_cache / eval_runs / eval_results / qa_logs，不含业务逻辑。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Document(Base):
    """设备手册、维保规范、故障案例等原始文档登记。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    doc_type: Mapped[str] = mapped_column(String(32), nullable=False, default="pdf")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ocr_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    failed_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Chunk(Base):
    """文档切分后的检索单元；向量以 float32 原始字节存在 `embedding`，由检索层写入。"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 归一化后的 float32 向量字节（np.float32.tobytes()）。不用 pgvector：
    # 2 万个块 × 512 维只有 40MB，numpy 全量点积 10~30ms，装 Postgres 的收益是零。
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # 生成该向量的模型名；换模型后旧向量必须重建，否则不同模型的向量会被混在一个矩阵里
    embed_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # 检索结果需要回溯到文档名做引用；预加载避免会话关闭后访问触发 N+1 或 Detached 报错
    document: Mapped["Document"] = relationship(lazy="joined", innerjoin=True)


class EmbeddingCache(Base):
    """文本 → 向量的本地缓存，key 为 sha256(model + text)。

    存在的唯一理由：入库重跑、评测重跑、同一段 query 反复检索时，不要重复付费。
    命中缓存就不发网络请求，这一点由 `app/llm/embed.py` 保证。
    """

    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    text_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EvalRun(Base):
    """一次评测运行的记录，配置以 JSON 保存。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalResult(Base):
    """评测运行中单个问题的检索与生成结果，以及逐条指标。"""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_chunk_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    retrieved_chunk_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class QaLog(Base):
    """一次问答调用的日志：用量、耗时、引用与错误，用于事后核算与排查幻觉。

    只记录「发生过什么」，不做业务判定；写入失败不得影响回答返回。
    """

    __tablename__ = "qa_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # answered / refused / error / partial（partial：客户端中途断开）
    answer_status: Mapped[str] = mapped_column(String(32), nullable=False, default="answered")
    reference_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_citations: Mapped[Any] = mapped_column(JSON, nullable=True)
    citations: Mapped[Any] = mapped_column(JSON, nullable=True)
    retrieved_chunk_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 服务端未返回 usage 时按字数估算，必须标记来源，不能把估算值当实测值
    usage_estimated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieval_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    llm_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
