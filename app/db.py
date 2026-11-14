"""数据库基础设施：引擎、声明基类、会话工厂、建表与连通性探测。

默认 SQLite（`data/kb.db`），不要求本机有 Postgres。两点适配是必须的：

1. `check_same_thread=False`：FastAPI 把阻塞代码丢进线程池，会话可能在不同线程里
   使用同一个连接池，SQLite 默认的线程检查会直接报错；
2. `journal_mode=WAL`：读多写少（检索远多于入库）时 WAL 允许读写并发，
   比默认的 rollback journal 少一批「database is locked」。

换 pgvector 时只改 `DATABASE_URL`，检索层已经封装在 `search()` 一个函数里。
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, make_url, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """所有 ORM 模型共用的声明基类。"""


def _is_sqlite(url: str) -> bool:
    return make_url(url).get_backend_name() == "sqlite"


def _prepare_sqlite(url: str) -> dict:
    """SQLite 需要先建目录，并放开线程检查；其它方言不需要这些。"""
    database = make_url(url).database
    if database and database != ":memory:":
        Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    return {"connect_args": {"check_same_thread": False, "timeout": 30}}


# create_engine 是惰性的，模块导入时不会建立真实连接
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    **(_prepare_sqlite(settings.database_url) if _is_sqlite(settings.database_url) else {}),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if _is_sqlite(settings.database_url):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001 - SQLAlchemy 钩子
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            # SQLite 默认**不**执行外键约束，不开这一项 `ON DELETE CASCADE` 形同虚设：
            # 删文档会留下一批孤儿分块，它们仍参与检索、引用会指向已不存在的文档。
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


def get_session() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个会话，请求结束后自动关闭。"""
    with SessionLocal() as session:
        yield session


def init_db() -> None:
    """按 ORM 模型建表（幂等，不含任何数据写入）。"""
    from app import models  # noqa: F401  局部导入，确保模型已注册到 Base.metadata

    Base.metadata.create_all(bind=engine)
    _warn_schema_drift()


def _warn_schema_drift() -> None:
    """本项目没有迁移工具：旧结构的库不会被 `create_all` 改掉，只能明确提示重建。"""
    try:
        inspector = inspect(engine)
        if "chunks" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("chunks")}
    except Exception:  # 探测失败不影响建表，交给后续查询暴露问题
        return
    if "embed_model" not in columns:
        logger.warning(
            "检测到旧版库结构（chunks 缺少 embed_model，embedding 类型也不是 BLOB）。"
            "本阶段把向量存成 float32 字节且不再依赖 pgvector，请删除 %s 后重新入库。",
            settings.database_url,
        )


def check_db() -> None:
    """执行最小查询探测连通性；不通时抛出异常由调用方转换为 503。"""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
