"""OpenAI 兼容大模型客户端：直连 `/chat/completions`，不引入额外 SDK。

三条硬约束，都是接口稳定性问题而不是「最佳实践」：

1. **超时**：连接、读取、整体截止共用一个 `LLM_TIMEOUT_SECONDS`（默认 30s）。
   流式响应里两个 token 之间也可能卡死，因此除了 httpx 的读超时，还按整体截止时间兜底。
2. **重试**：最多 2 次，指数退避（1s、2s），只重试可恢复的错误（超时 / 连接失败 /
   408、409、429、5xx）。**已经吐出过 token 就不重试**——重试会从头再生成一遍，
   前端会看到重复文本，且浪费额度。
3. **结构化错误**：任何失败都返回 `LlmError`，不向上抛异常。上层（`api/chat.py`）
   把它转成 SSE 的 `error` 事件，前端拿到的是 `{code, message, retryable, attempts}`。

用量统计优先取服务端返回的 `usage`；服务端不回（部分兼容实现不支持
`stream_options.include_usage`）时按字数估算并标记 `estimated=True`，估算值不能当实测值用。
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

Message = dict[str, str]

# 可重试的 HTTP 状态码：限流、并发冲突、网关与服务端错误
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

# 中文文本的经验折算系数：约 1.5 个字 ≈ 1 个 token。只用于服务端不返回 usage 时估算
_CHARS_PER_TOKEN = 1.5

# 进程内开关：某些兼容实现不认 stream_options，被拒一次后本进程内不再发送
_include_usage = settings.llm_include_usage


class LlmConfigError(RuntimeError):
    """LLM 配置缺失。仅在「需要调用但没配置」时抛出，由 `is_configured()` 提前拦截。"""


@dataclass(frozen=True)
class TokenUsage:
    """一次调用的 token 用量；`estimated` 为 True 表示是本地估算而非服务端实测。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class LlmError:
    """结构化错误：前端与日志读同一份字段，不需要解析异常栈。"""

    # config_missing / timeout / network / http_status / protocol / interrupted
    code: str
    message: str
    attempts: int = 1
    retryable: bool = False
    status_code: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "status_code": self.status_code,
        }

    def as_text(self) -> str:
        """写进 `qa_logs.llm_error` 的单行文本。"""
        return f"{self.code}(status={self.status_code}, attempts={self.attempts}): {self.message}"


@dataclass(frozen=True)
class LlmEvent:
    """流式事件：`delta` 增量文本、`done` 结束（带用量）、`error` 结构化失败。"""

    kind: Literal["delta", "done", "error"]
    text: str = ""
    usage: TokenUsage | None = None
    error: LlmError | None = None
    # 结束原因：stop / length（被 max_tokens 截断）/ None（服务端没给）
    finish_reason: str | None = None


@dataclass(frozen=True)
class Completion:
    """非流式调用（校验脚本、离线评估）的返回体。"""

    text: str
    usage: TokenUsage | None
    error: LlmError | None
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage.as_dict() if self.usage else None,
            "error": self.error.as_dict() if self.error else None,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class _LlmCallFailed(Exception):
    """内部信号：把一次尝试的失败包成结构化错误，交给外层决定要不要重试。"""

    def __init__(self, error: LlmError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class _StreamState:
    """一次尝试的可变状态；生成器无法返回值，用这个容器带出用量与累计文本。"""

    text: str = ""
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    deadline: float = 0.0


def is_configured() -> bool:
    """三项配置齐全才算可用；缺项时接口直接返回 config_missing，不发起 HTTP 请求。"""
    return bool(
        settings.llm_base_url.strip()
        and settings.llm_api_key.strip()
        and settings.llm_model.strip()
    )


def is_available() -> bool:
    """能产出回答：要么配好了真实接口，要么显式开了 MOCK_MODE。"""
    return settings.mock_mode or is_configured()


def missing_config() -> list[str]:
    """列出缺失的配置项名，用于错误提示。"""
    return [
        name
        for name, value in (
            ("LLM_BASE_URL", settings.llm_base_url),
            ("LLM_API_KEY", settings.llm_api_key),
            ("LLM_MODEL", settings.llm_model),
        )
        if not value.strip()
    ]


def stream_chat(messages: Sequence[Message]) -> Iterator[LlmEvent]:
    """流式调用并逐个 token 产出；失败时以 `error` 事件收尾，不抛异常。

    `MOCK_MODE=1` 时整条替换为本地桩模型：这里就是「模型边界」，
    上层的提示词、SSE、引用校验全都感知不到差别。
    """
    if settings.mock_mode:
        from app.llm import mock  # 局部导入：mock 需要本模块的事件类型

        yield from mock.stream_chat(list(messages))
        return

    try:
        extra_body = _extra_body()
    except LlmConfigError as exc:
        yield LlmEvent(
            kind="error",
            error=LlmError(code="config_invalid", message=str(exc), retryable=False),
        )
        return

    if not is_configured():
        yield LlmEvent(
            kind="error",
            error=LlmError(
                code="config_missing",
                message="大模型未配置，请先在 .env 填写 " + "、".join(missing_config()),
                retryable=False,
            ),
        )
        return

    attempts = max(1, settings.llm_max_retries + 1)
    for attempt in range(1, attempts + 1):
        state = _StreamState(deadline=time.monotonic() + settings.llm_timeout_seconds)
        emitted = False
        try:
            for piece in _stream_once(messages, state, extra_body):
                emitted = True
                yield LlmEvent(kind="delta", text=piece)
        except _LlmCallFailed as exc:
            error = replace(exc.error, attempts=attempt)
            # 已经吐过 token 就不再重试：重来一遍会让前端看到重复内容
            if emitted or not error.retryable or attempt >= attempts:
                logger.warning("大模型调用失败：%s", error.as_text())
                yield LlmEvent(kind="error", error=error)
                return
            backoff = settings.llm_retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning("%s，%.1fs 后重试（第 %d 次）", error.as_text(), backoff, attempt)
            time.sleep(backoff)
            continue

        usage = state.usage or _estimate_usage(messages, state.text)
        yield LlmEvent(
            kind="done", text=state.text, usage=usage, finish_reason=state.finish_reason
        )
        return


def complete(messages: Sequence[Message]) -> Completion:
    """非流式封装：收集全部增量，返回文本 + 用量 + 结构化错误。"""
    started = time.perf_counter()
    parts: list[str] = []
    usage: TokenUsage | None = None
    error: LlmError | None = None
    for event in stream_chat(messages):
        if event.kind == "delta":
            parts.append(event.text)
        elif event.kind == "done":
            usage = event.usage
        else:
            error = event.error
    elapsed_ms = (time.perf_counter() - started) * 1000
    return Completion(text="".join(parts), usage=usage, error=error, elapsed_ms=elapsed_ms)


def _stream_once(
    messages: Sequence[Message], state: _StreamState, extra_body: dict[str, Any] | None = None
) -> Iterator[str]:
    """一次 HTTP 尝试：打开 SSE 流并按 OpenAI 兼容格式解析增量。"""
    global _include_usage

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": list(messages),
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
    }
    # 厂商特有字段最后合并，允许覆盖上面的默认值
    if extra_body:
        payload.update(extra_body)
    if _include_usage:
        payload["stream_options"] = {"include_usage": True}

    try:
        with httpx.Client(timeout=httpx.Timeout(settings.llm_timeout_seconds)) as client:
            with client.stream(
                "POST", _endpoint(), json=payload, headers=_headers()
            ) as response:
                if response.status_code != 200:
                    body = response.read().decode("utf-8", "replace")
                    # 少数兼容实现不认 stream_options：本进程内关掉用量统计并重试一次，
                    # 让它退回「按字数估算」，而不是整条问答失败
                    if (
                        _include_usage
                        and response.status_code == 400
                        and "stream_options" in body
                    ):
                        _include_usage = False
                        logger.warning("服务端不认 stream_options，改用字数估算用量")
                        raise _LlmCallFailed(
                            LlmError(
                                code="http_status",
                                message="服务端不接受 stream_options，已关闭用量统计并重试",
                                retryable=True,
                                status_code=400,
                            )
                        )
                    raise _LlmCallFailed(_http_error(response.status_code, body))
                yield from _iter_sse(response, state)
    except httpx.TimeoutException as exc:
        raise _LlmCallFailed(
            LlmError(
                code="timeout",
                message=f"模型接口超时（{settings.llm_timeout_seconds:.0f}s）",
                retryable=True,
            )
        ) from exc
    except httpx.HTTPError as exc:
        raise _LlmCallFailed(
            LlmError(
                code="network",
                message=f"模型接口连接失败：{type(exc).__name__} {exc}",
                retryable=True,
            )
        ) from exc


def _iter_sse(response: httpx.Response, state: _StreamState) -> Iterator[str]:
    """逐行解析 SSE：只认 `data:` 负载，忽略注释与心跳，`[DONE]` 结束。"""
    for raw in response.iter_lines():
        if time.monotonic() > state.deadline:
            raise _LlmCallFailed(
                LlmError(
                    code="timeout",
                    message=f"模型接口超过整体截止时间 {settings.llm_timeout_seconds:.0f}s 仍未结束",
                    retryable=True,
                )
            )
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            continue
        if line == "[DONE]":
            return
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            # 个别实现会在流里插入非 JSON 的心跳文本，跳过而不是整条失败
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise _LlmCallFailed(_payload_error(chunk["error"]))

        choices = chunk.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                state.text += piece
                yield piece
            if choice.get("finish_reason"):
                state.finish_reason = str(choice["finish_reason"])
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            state.usage = _parse_usage(usage)


def _extra_body() -> dict[str, Any]:
    """解析 `LLM_EXTRA_BODY`：厂商特有字段（如关掉思考模式）原样并进请求体。

    不同兼容实现的扩展字段各不相同，硬编码任何一个都会绑死在某一家上。
    放开一个 JSON 口子，比在代码里为每家写 if 更可控，也更好排错。
    """
    raw = settings.llm_extra_body.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LlmConfigError(f"LLM_EXTRA_BODY 不是合法 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise LlmConfigError("LLM_EXTRA_BODY 必须是 JSON 对象，例如 {\"thinking\": {\"type\": \"disabled\"}}")
    return parsed


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def _endpoint() -> str:
    """base_url 兼容两种写法：只给到 `/v1`，或直接给到 `/chat/completions`。"""
    base = settings.llm_base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _parse_usage(payload: dict[str, Any]) -> TokenUsage:
    prompt = _as_int(payload.get("prompt_tokens"))
    completion = _as_int(payload.get("completion_tokens"))
    total = _as_int(payload.get("total_tokens")) or prompt + completion
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        estimated=False,
    )


def _http_error(status: int, body: str) -> LlmError:
    """HTTP 非 200：截断响应体，避免把整页 HTML 塞进日志与前端。"""
    snippet = " ".join(body.split())[:300] or "（空响应体）"
    return LlmError(
        code="http_status",
        message=f"模型接口返回 HTTP {status}：{snippet}",
        retryable=status in _RETRYABLE_STATUS or status >= 500,
        status_code=status,
    )


def _payload_error(payload: Any) -> LlmError:
    """HTTP 200 但流里冒出 error 字段（部分兼容实现的错误约定）。"""
    message = payload.get("message") if isinstance(payload, dict) else str(payload)
    return LlmError(
        code="protocol",
        message=f"模型接口在流中返回错误：{str(message)[:300]}",
        retryable=False,
    )


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _estimate_usage(messages: Sequence[Message], answer: str) -> TokenUsage:
    """按字数估算用量；标记 estimated=True，避免把估算值当成服务端实测值。"""
    prompt_chars = sum(len(message.get("content", "")) for message in messages)
    # 每条消息的角色与分隔符约 4 个 token 的额外开销
    prompt = int(prompt_chars / _CHARS_PER_TOKEN) + 4 * len(messages)
    completion = int(len(answer) / _CHARS_PER_TOKEN)
    return TokenUsage(
        prompt_tokens=max(prompt, 1),
        completion_tokens=completion,
        total_tokens=max(prompt, 1) + completion,
        estimated=True,
    )
