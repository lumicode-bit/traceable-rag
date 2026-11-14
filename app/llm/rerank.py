"""远端重排：调 `/rerank` 接口，失败一律降级为原序并标记 `degraded`。

为什么是「降级」而不是「报错」：重排只是给已召回的候选重新排序，原序本身可用。
因为一次网络抖动就让整条问答失败，收益与代价不成比例。但降级必须**可见**：
返回原因写日志、`RerankOutcome.degraded` 传给上层，bench 与日志里能看到它不是真重排。

`MOCK_MODE=1` 时用本地词面重合度打分，纯为了让 hybrid_rerank 这条链路在无 key
的机器上也能被跑到，不代表真实重排效果。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
DEFAULT_RERANK_CANDIDATES = 10


@dataclass
class RerankOutcome:
    """重排结果：items 为 [(chunk_id, 分数或 None)]，degraded 表明这次是不是真的重排过。"""

    items: list[tuple[int, float | None]] = field(default_factory=list)
    degraded: bool = False
    reason: str = ""
    mode: str = "off"  # off / http / mock / degraded


def is_enabled() -> bool:
    """重排开关，唯一判据是 `RERANK_ENABLED`。"""
    return bool(settings.rerank_enabled)


def candidate_limit() -> int:
    """实际参与重排的候选数。"""
    return max(1, int(settings.rerank_candidates or DEFAULT_RERANK_CANDIDATES))


def is_configured() -> bool:
    return bool(
        settings.rerank_base_url.strip()
        and settings.rerank_api_key.strip()
        and settings.rerank_model.strip()
    )


def rerank(
    query: str, candidates: list[tuple[int, str]], top_k: int
) -> RerankOutcome:
    """对候选重排；关闭、未配置或调用失败都返回可用结果，不抛异常。"""
    if not candidates:
        return RerankOutcome(items=[], mode="off")
    if not is_enabled():
        return RerankOutcome(items=[(chunk_id, None) for chunk_id, _ in candidates[:top_k]])

    if settings.mock_mode:
        from app.llm import mock  # 局部导入，避免 mock 与其它模块互相引用

        scores = mock.stub_rerank_scores(query, [text for _, text in candidates])
        ranked = _rank(candidates, scores)
        return RerankOutcome(items=ranked[:top_k], mode="mock")

    if not is_configured():
        reason = "重排已开启但未配置 RERANK_BASE_URL / RERANK_API_KEY / RERANK_MODEL，已降级为原序"
        logger.warning(reason)
        return _degraded(candidates, top_k, reason)

    try:
        scores = _request(query, [text for _, text in candidates])
    except Exception as exc:  # 网络、协议、鉴权问题都降级，不让整条链路失败
        reason = f"重排接口调用失败，已降级为原序：{type(exc).__name__}: {exc}"
        logger.warning(reason)
        return _degraded(candidates, top_k, reason)

    return RerankOutcome(items=_rank(candidates, scores)[:top_k], mode="http")


def _degraded(candidates: list[tuple[int, str]], top_k: int, reason: str) -> RerankOutcome:
    return RerankOutcome(
        items=[(chunk_id, None) for chunk_id, _ in candidates[:top_k]],
        degraded=True,
        reason=reason,
        mode="degraded",
    )


def _rank(
    candidates: list[tuple[int, str]], scores: list[float]
) -> list[tuple[int, float | None]]:
    """按分数降序；分数相同保持融合原序（sorted 是稳定排序）。"""
    paired = list(zip((chunk_id for chunk_id, _ in candidates), scores))
    return sorted(paired, key=lambda item: -item[1])


def _request(query: str, documents: list[str]) -> list[float]:
    """一次 HTTP 调用；重排是可选优化，这里只重试一次，失败就降级。"""
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    last_error = ""
    for attempt in (1, 2):
        try:
            with httpx.Client(timeout=httpx.Timeout(settings.rerank_timeout_seconds)) as client:
                response = client.post(
                    _endpoint(),
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {settings.rerank_api_key}",
                        "Content-Type": "application/json",
                    },
                )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}：{' '.join(response.text.split())[:200]}"
                if response.status_code not in _RETRYABLE_STATUS and response.status_code < 500:
                    raise RuntimeError(last_error)
                continue
            return _parse(response.json(), len(documents))
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last_error or "重排接口调用失败")


def _parse(payload: object, expected: int) -> list[float]:
    """兼容三种常见返回：`results[].relevance_score`、`results[].score`、`scores[]`。"""
    if not isinstance(payload, dict):
        raise RuntimeError("重排接口返回的不是 JSON 对象")

    if isinstance(payload.get("scores"), list) and len(payload["scores"]) == expected:
        return [float(value) for value in payload["scores"]]

    results = payload.get("results") or payload.get("data")
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"重排接口返回异常：{str(payload)[:200]}")

    scores = [0.0] * expected
    for position, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", position))
        if 0 <= index < expected:
            value = item.get("relevance_score", item.get("score", 0.0))
            scores[index] = float(value)
    return scores


def _endpoint() -> str:
    base = settings.rerank_base_url.strip().rstrip("/")
    if base.endswith("/rerank"):
        return base
    return f"{base}/rerank"
