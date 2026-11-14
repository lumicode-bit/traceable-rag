"""文档管理接口：上传、列表、删除、重新解析。

四条设计决定：

1. **上传不在请求里解析**。一份 163 页的手册（含 OCR）要几十秒，同步做会被 nginx
   超时切断、手机上只看到一个转圈。上传只负责落盘 + 登记，解析丢给后台任务，
   页面轮询状态。
2. **解析串行**。SQLite 单写者，两个文件同时解析只会互相等锁；用一把模块级锁把
   入库串起来，比在 SQL 层处理锁冲突简单得多。
3. **解析完顺手补索引**。`ingest_file` 只写块、不写向量；不补索引，新文档在检索里
   等于不存在。补索引失败不改变「已入库」的状态，但要在日志里说清楚。
4. **删除要连文件一起删**，否则重新扫描 `data/raw` 会把删掉的文档又捡回来。
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, get_session
from app.ingest.pipeline import STATUS_INDEXED, STATUS_PARSING, STATUS_PENDING, ingest_file
from app.models import Chunk, Document
from app.retrieval import bm25, rebuild_indexes
from app.retrieval.vector import reset_index

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 与 loader.SUPPORTED_SUFFIXES 保持一致；单独列一份是为了给出准确的 415 提示
ALLOWED_SUFFIXES = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
_UNSAFE_NAME = re.compile(r"[^\w\u4e00-\u9fff.\-]+")

# 串行化入库：SQLite 只允许一个写者，抢锁只会让两边都变慢
_ingest_lock = threading.Lock()


def _raw_dir() -> Path:
    path = Path(settings.raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@router.get("")
def list_documents(session: Session = Depends(get_session)) -> dict:
    """文档列表：状态、页数、块数、已向量化块数。"""
    chunk_total = (
        select(Chunk.document_id, func.count().label("total"))
        .group_by(Chunk.document_id)
        .subquery()
    )
    chunk_vectorized = (
        select(Chunk.document_id, func.count().label("vectorized"))
        .where(Chunk.embedding.isnot(None))
        .group_by(Chunk.document_id)
        .subquery()
    )
    rows = session.execute(
        select(
            Document,
            func.coalesce(chunk_total.c.total, 0),
            func.coalesce(chunk_vectorized.c.vectorized, 0),
        )
        .outerjoin(chunk_total, chunk_total.c.document_id == Document.id)
        .outerjoin(chunk_vectorized, chunk_vectorized.c.document_id == Document.id)
        .order_by(Document.id.desc())
    ).all()

    items = [
        {
            "id": document.id,
            "filename": document.filename,
            "status": document.status,
            "pages": document.page_count,
            "ocr_pages": document.ocr_pages,
            "chunks": int(total),
            "chunks_vectorized": int(vectorized),
            "failed_reason": document.failed_reason,
            "created_at": document.created_at.isoformat() if document.created_at else None,
        }
        for document, total, vectorized in rows
    ]
    return {"documents": items, "total": len(items)}


@router.post("", status_code=201)
async def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    """接收文件 → 落盘 → 后台解析。

    这里只做「能不能收」的判断：后缀、大小、文件名。解析失败由后台任务写进
    `failed_reason`，页面能看到原因，而不是一个静默消失的上传。
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型：{suffix or '无后缀'}（允许 {'、'.join(sorted(ALLOWED_SUFFIXES))}）",
        )

    raw_dir = _raw_dir()
    target = _unique_path(raw_dir, _safe_name(file.filename or f"upload{suffix}"))

    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {settings.max_upload_mb}MB 上限",
                    )
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"写入文件失败：{exc}") from exc
    finally:
        await file.close()

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="上传内容为空")

    # 同 sha256 且已入库时，ingest_file 会跳过；这里先查一下，好给前端一句明确的话
    duplicate = _find_duplicate(session, target)
    background.add_task(_ingest_document, target)

    return {
        "filename": target.name,
        "size": written,
        "duplicate_of": duplicate,
        "message": "文件已上传，正在后台解析" if not duplicate else "该文件已入库，本次不会重复解析",
    }


@router.delete("/{document_id}")
def delete_document(document_id: int, session: Session = Depends(get_session)) -> dict:
    """删除文档及其分块；文件也一并删掉，否则下次全量扫描会把它捡回来。"""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    path = _raw_dir() / document.source_path
    # 显式删分块，不只依赖 ON DELETE CASCADE：SQLite 默认不校验外键（见 db.py 的
    # `PRAGMA foreign_keys=ON`），漏开时这里会留下孤儿块，而孤儿块照样参与检索。
    session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    session.delete(document)
    session.commit()

    removed_file = False
    try:
        if path.is_file():
            path.unlink()
            removed_file = True
    except OSError as exc:  # 文件删不掉不该让接口失败，但要让人知道
        logger.warning("删除文件失败 %s：%s", path, exc)

    _invalidate_indexes()
    return {"deleted": document_id, "file_removed": removed_file}


@router.post("/{document_id}/ingest", status_code=202)
def reingest_document(
    document_id: int,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    """重新解析：把状态改回 pending，交给 ingest_file 重建记录与分块。"""
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    path = _raw_dir() / document.source_path
    if not path.is_file():
        raise HTTPException(status_code=410, detail=f"源文件已不存在：{document.source_path}")

    document.status = STATUS_PENDING
    document.failed_reason = None
    session.commit()

    background.add_task(_ingest_document, path)
    return {"document_id": document_id, "message": "已排入解析队列"}


def _ingest_document(path: Path, *, build_index: bool = True) -> None:
    """后台任务：解析 + 补索引。任何异常只记日志，不再抛给已经返回的请求。"""
    raw_dir = _raw_dir()
    with _ingest_lock:
        try:
            with SessionLocal() as session:
                result = ingest_file(session, path, raw_dir)
                if result.status == "failed":
                    logger.warning("解析失败 %s：%s", path.name, result.reason)
                    return
                if result.status == "skipped":
                    return
                if build_index:
                    _build_indexes(session, path.name)
        except Exception:
            logger.exception("后台解析异常 %s", path.name)


def _build_indexes(session: Session, name: str) -> None:
    """补向量与 BM25；失败只告警——文档已入库，没向量只是暂时搜不到。"""
    try:
        stats = rebuild_indexes(session)
        logger.info("索引已更新（%s）：%s", name, stats)
    except Exception as exc:
        logger.warning(
            "文档 %s 已入库，但补齐向量索引失败（检索暂时搜不到它）：%s", name, exc
        )


def _find_duplicate(session: Session, path: Path) -> int | None:
    from app.ingest.pipeline import sha256_of

    digest = sha256_of(path)
    row = session.execute(
        select(Document.id).where(Document.sha256 == digest, Document.status == STATUS_INDEXED)
    ).scalar_one_or_none()
    return int(row) if row is not None else None


def _safe_name(name: str) -> str:
    """只留文件名本身，去掉路径分隔符与控制字符，防止写到 data/raw 之外。"""
    base = Path(name).name
    cleaned = _UNSAFE_NAME.sub("_", base).strip("._")
    return cleaned or "upload.bin"


def _unique_path(directory: Path, name: str) -> Path:
    """重名不覆盖：`手册.pdf` 已存在时写 `手册-1.pdf`。"""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 1000):
        alternative = directory / f"{stem}-{index}{suffix}"
        if not alternative.exists():
            return alternative
    raise HTTPException(status_code=409, detail=f"同名文件过多：{name}")


def _invalidate_indexes() -> None:
    """删文档后立即让内存索引失效，避免检索到已经不存在的块。"""
    try:
        bm25.invalidate()
        reset_index()
    except Exception:  # 索引是缓存，失效失败不该让删除接口报错
        logger.warning("索引失效处理失败", exc_info=True)


__all__ = ["router", "STATUS_PARSING"]
