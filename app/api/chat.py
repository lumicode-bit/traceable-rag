"""问答接口：POST /api/chat，SSE 流式返回带引用的回答。

响应是一个 SSE 流，事件顺序固定：

```
event: meta      检索结果摘要（耗时、片段数、引用列表、是否已配置模型）
event: delta     增量文本，可能有很多条
event: citations 最后的引用列表（文档名、页码、块内容摘要）+ 引用校验 + 用量 + 耗时
event: done      结束标记
```

出错时在 `citations`/`done` 之前插入一条 `event: error`，负载是
`{code, message, retryable, attempts}`——前端拿到的是结构化错误，不是异常栈。

检索与生成都是同步阻塞代码（向量编码是 CPU、httpx 是同步流），因此必须放进
线程池执行，否则会把 FastAPI 的事件循环堵住，其它请求一起排队。
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.config import settings
from app.llm import pipeline
from app.retrieval import MODES, InvalidModeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    """单轮问答请求；不做多轮记忆，没有 history 字段。"""

    question: str = Field(..., min_length=1, max_length=500, description="故障现象或问题")
    mode: str = Field(default=settings.chat_mode, description="检索模式：vector / hybrid / hybrid_rerank")
    top_k: int = Field(default=settings.chat_top_k, ge=1, le=20, description="送入提示词的片段数")


@router.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    """接收问题，流式返回回答；最后一个事件带完整引用列表。"""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question 不能为空")
    if payload.mode not in MODES:
        raise HTTPException(
            status_code=422,
            detail=f"不支持的检索模式：{payload.mode}（可选：{'、'.join(MODES)}）",
        )
    return StreamingResponse(
        _event_stream(question, payload.mode, payload.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉 nginx 之类反代的缓冲，否则流式会攒成一整块再发
            "X-Accel-Buffering": "no",
        },
    )


async def _event_stream(question: str, mode: str, top_k: int) -> AsyncIterator[str]:
    """把同步的检索 / 生成链路桥接成异步事件流。"""
    try:
        plan = await run_in_threadpool(pipeline.prepare, question, mode, top_k)
    except InvalidModeError as exc:
        yield _sse("error", {"code": "invalid_mode", "message": str(exc), "retryable": False})
        yield _sse("done", {})
        return
    except Exception as exc:  # 检索失败（模型缺失、库不可用）也要给前端一个可读的原因
        logger.exception("检索失败")
        yield _sse(
            "error",
            {
                "code": "retrieval_failed",
                "message": f"检索失败：{type(exc).__name__} {exc}",
                "retryable": True,
                "attempts": 1,
            },
        )
        yield _sse("done", {})
        return

    yield _sse("meta", plan.meta())
    async for item in iterate_in_threadpool(pipeline.stream_answer(plan)):
        yield _sse(item["event"], item["data"])
    yield _sse("done", {})


def _sse(event: str, data: dict[str, Any]) -> str:
    """SSE 帧：JSON 单行，禁止换行，否则会被浏览器当成新的 data 行切开。"""
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"
