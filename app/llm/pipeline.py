"""生成层编排：检索 → 组织提示词 → 流式生成 → 校验引用 → 写调用日志。

分工边界：
- `app/retrieval` 决定「给模型看哪些片段」；
- `app/llm/prompt.py` 决定「怎么把这些片段和约束说给模型」；
- `app/llm/client.py` 只管「怎么把请求发出去、失败怎么表达」；
- 本模块只做编排与记账，不实现任何检索算法或 HTTP 细节。

两段耗时分开记：`retrieval_ms`（含向量编码与融合）与 `generation_ms`（模型首个 token 到结束）。
日志写入失败只告警，绝不影响已经流给用户的回答。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.llm import citations, client, prompt
from app.llm.citations import Citation, CitationCheck
from app.llm.client import LlmError, TokenUsage
from app.llm.support import SupportCheck
from app.llm.support import check_answer as support_check
from app.models import QaLog
from app.retrieval import SearchHit, search_with_ranks

logger = logging.getLogger(__name__)

# 返回给前端的「块内容摘要」长度；完整正文不进响应体，避免一次问答回传几千字
SNIPPET_CHARS = 120


@dataclass
class AnswerPlan:
    """一次问答的检索结果与提示词快照：生成阶段只读，不再动数据库。"""

    question: str
    mode: str
    top_k: int
    retrieval_ms: float
    citations: list[Citation]
    allowed: set[tuple[str, int | None]]
    messages: list[prompt.Message]
    chunk_ids: list[int]

    def meta(self) -> dict[str, Any]:
        """流开始前推给前端的事件：让页面能立刻显示「检索到什么、要花多久」。"""
        return {
            "question": self.question,
            "mode": self.mode,
            "top_k": self.top_k,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "passage_count": len(self.citations),
            "citations": [citation.as_dict() for citation in self.citations],
            "llm_configured": client.is_available(),
            "model": _model_label(),
            "mock_mode": settings.mock_mode,
        }


def prepare(question: str, mode: str, top_k: int) -> AnswerPlan:
    """检索并组装提示词。模式非法时由 `search_with_ranks` 抛 `InvalidModeError`。"""
    started = time.perf_counter()
    hits = search_with_ranks(question, top_k=top_k, mode=mode)
    retrieval_ms = (time.perf_counter() - started) * 1000

    passages = [_to_citation(hit) for hit in hits]
    return AnswerPlan(
        question=question,
        mode=mode,
        top_k=top_k,
        retrieval_ms=retrieval_ms,
        citations=passages,
        allowed=citations.allowed_keys(passages),
        messages=prompt.build_messages(question, passages),
        chunk_ids=[hit.chunk_id for hit in hits],
    )


def stream_answer(plan: AnswerPlan) -> Iterator[dict[str, Any]]:
    """流式产出事件：`{"event": "delta"|"citations", "data": {...}}`。

    生成结束后一定先发 `citations`（引用列表 + 校验结果 + 用量 + 耗时），再结束。
    客户端中途断开时走 `finally`，把已生成的部分如实记成 `partial`。
    """
    started = time.perf_counter()
    parts: list[str] = []
    usage: TokenUsage | None = None
    error: LlmError | None = None
    finish_reason: str | None = None
    logged = False
    try:
        for event in client.stream_chat(plan.messages):
            if event.kind == "delta":
                parts.append(event.text)
                yield {"event": "delta", "data": {"text": event.text}}
            elif event.kind == "done":
                usage = event.usage
                finish_reason = event.finish_reason
            else:
                error = event.error

        answer = "".join(parts)
        generation_ms = (time.perf_counter() - started) * 1000
        if finish_reason == "length":
            # 被 max_tokens 截断：回答尾部可能刚好切在引用中间，校验结果要打折扣看
            logger.warning("生成长度触顶被截断（max_tokens=%s），回答可能不完整", settings.llm_max_tokens)
        check = citations.validate_answer(answer, plan.allowed)
        # 支撑性校验要按「标签 → 可搜索文本」查，不能把 citations 列表当函数传进去。
        #
        # 两个坑都在真实数据上踩过：
        # 1. 文本要带上文档名——结论常写「CP320 的端子是 U、V、W」，而块正文是接线图表，
        #    型号只出现在文档名里；
        # 2. 一个标签可能对应**多个块**（同一页有 2 个块，标签只带页码），
        #    用 dict 直接覆盖会拿错块，必须把所有同标签块的正文拼起来。
        bodies: dict[str, list[str]] = {}
        for citation in plan.citations:
            bodies.setdefault(citation.label, []).append(f"{citation.doc} {citation.content}")
        support = support_check(
            answer,
            lambda label: " ".join(bodies.get(label, [])) or None,
            # 型号对比句只引用其中一块是合理的，所以型号允许在全局上下文里找
            context=" ".join(text for texts in bodies.values() for text in texts),
        )
        status = _status_of(answer, error)
        yield {
            "event": "citations",
            "data": {
                "answer_status": status,
                "citations": [citation.as_dict() for citation in plan.citations],
                "citation_check": check.as_dict(),
                "support_check": support.as_dict(),
                "usage": usage.as_dict() if usage else None,
                "timings": {
                    "retrieval_ms": round(plan.retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "total_ms": round(plan.retrieval_ms + generation_ms, 1),
                },
                "model": _model_label(),
                "finish_reason": finish_reason,
                "error": error.as_dict() if error else None,
            },
        }
        save_call_log(
            plan,
            answer=answer,
            status=status,
            usage=usage,
            error=error,
            generation_ms=generation_ms,
            check=check,
        )
        logged = True
    finally:
        if not logged:
            save_call_log(
                plan,
                answer="".join(parts),
                status="partial",
                usage=usage,
                error=error
                or LlmError(code="interrupted", message="客户端断开，生成未完成", retryable=True),
                generation_ms=(time.perf_counter() - started) * 1000,
                check=None,
            )


def answer_once(question: str, mode: str, top_k: int) -> dict[str, Any]:
    """一次性收集完整回答，供校验脚本与离线评估使用；与接口走同一条链路。"""
    plan = prepare(question, mode, top_k)
    answer_parts: list[str] = []
    payload: dict[str, Any] = {}
    for item in stream_answer(plan):
        if item["event"] == "delta":
            answer_parts.append(item["data"]["text"])
        elif item["event"] == "citations":
            payload = item["data"]
    return {
        "question": question,
        "answer": "".join(answer_parts),
        "citations": payload.get("citations", []),
        "citation_check": payload.get("citation_check", {}),
        "support_check": payload.get("support_check", {}),
        "usage": payload.get("usage"),
        "timings": payload.get("timings", {}),
        "error": payload.get("error"),
        "answer_status": payload.get("answer_status", "error"),
        "finish_reason": payload.get("finish_reason"),
        "retrieval_ms": round(plan.retrieval_ms, 1),
    }


def save_call_log(
    plan: AnswerPlan,
    *,
    answer: str,
    status: str,
    usage: TokenUsage | None,
    error: LlmError | None,
    generation_ms: float,
    check: CitationCheck | None,
    session: Session | None = None,
) -> bool:
    """把本次调用写进 `qa_logs`；返回是否写入成功。写日志失败不影响回答。"""
    row = QaLog(
        question=plan.question,
        mode=plan.mode,
        top_k=plan.top_k,
        answer=answer or None,
        answer_status=status,
        reference_count=check.occurrences if check else 0,
        invalid_citations=[issue.as_dict() for issue in check.invalid] if check else None,
        citations=[citation.as_dict() for citation in plan.citations],
        retrieved_chunk_ids=list(plan.chunk_ids),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        total_tokens=usage.total_tokens if usage else None,
        usage_estimated=bool(usage.estimated) if usage else False,
        retrieval_ms=round(plan.retrieval_ms, 1),
        generation_ms=round(generation_ms, 1),
        llm_model=_model_label(),
        llm_error=error.as_text() if error else None,
    )
    try:
        if session is not None:
            session.add(row)
            session.commit()
        else:
            with SessionLocal() as own_session:
                own_session.add(row)
                own_session.commit()
    except Exception as exc:  # 日志是旁路，不能把已经生成好的回答弄丢
        logger.warning("写入 qa_logs 失败：%s", exc)
        return False
    return True


def _to_citation(hit: SearchHit) -> Citation:
    """把检索命中转成引用对象：完整正文给模型，摘要给前端。

    两者必须分开：故障码表这类块有 1800+ 字，只把前 120 字给模型，
    等于让它对着表头回答，结果只能是拒答。
    """
    chunk = hit.chunk
    document = chunk.document
    return Citation(
        chunk_id=hit.chunk_id,
        doc=document.filename if document else "未知文档",
        page=chunk.page_no,
        section=chunk.section,
        content=chunk.content,
        snippet=_summarize(chunk.content),
    )


def _summarize(text: str, limit: int = SNIPPET_CHARS) -> str:
    body = " ".join(text.split())
    if len(body) <= limit:
        return body
    return body[:limit] + "…"


def _model_label() -> str | None:
    """日志与响应里显示的模型名；MOCK 必须显式标出来，不能冒充真实模型。"""
    if settings.mock_mode:
        return "stub-llm(mock)"
    return settings.llm_model or None


def _status_of(answer: str, error: LlmError | None) -> str:
    if error is not None:
        return "error"
    if citations.is_refusal(answer):
        return "refused"
    return "answered"
