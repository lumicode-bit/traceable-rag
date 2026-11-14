"""块内容接口：引用卡片点开时取完整原文。

为什么单独一个接口：一次问答的响应体里只放 120 字摘要（`snippet`），
5 个片段如果全带回原文就是上万字，手机上根本滚不过来。完整正文按需拉取。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Chunk

router = APIRouter(prefix="/api/chunks", tags=["chunks"])


@router.get("/{chunk_id}")
def get_chunk(chunk_id: int, session: Session = Depends(get_session)) -> dict:
    """返回一个块的完整正文与来源信息；不存在就是 404，不返回空字符串。"""
    chunk = session.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="块不存在（可能已随文档被删除）")

    document = chunk.document
    return {
        "chunk_id": chunk.id,
        "doc": document.filename if document else "未知文档",
        "page": chunk.page_no,
        "section": chunk.section,
        "content": chunk.content,
        "chars": len(chunk.content),
    }
