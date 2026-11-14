"""FastAPI 应用入口：装配应用、挂载路由与静态资源、/health 健康检查。"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from app.api.chat import router as chat_router
from app.api.chunks import router as chunks_router
from app.api.documents import router as documents_router
from app.db import check_db, init_db
from app.web.pages import STATIC_DIR
from app.web.pages import router as pages_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时建表；数据库不可用时仍启动服务，让 /health 如实返回 503。"""
    try:
        init_db()
    except Exception as exc:  # 建表失败不应掩盖健康检查的职责
        logger.warning("数据库初始化失败，/health 将持续返回 503：%s", exc)
    yield


app = FastAPI(title="工业设备运维知识库", version="0.7.0", lifespan=lifespan)

# 接口在前、页面在后：页面路由带 "/"，放最后更不容易误挡
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(chunks_router)
app.include_router(pages_router)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health() -> Response:
    """返回数据库连通状态：可用 200 {"db":"ok"}，不可用 503 {"db":"error"}。"""
    try:
        check_db()
    except Exception:
        return JSONResponse(status_code=503, content={"db": "error"})
    return JSONResponse(status_code=200, content={"db": "ok"})
