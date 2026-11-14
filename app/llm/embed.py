"""远端向量模型客户端：OpenAI 兼容 `/embeddings` + 本地缓存。

四条硬约束，都是围绕「本机不装 torch」定的：

1. **只依赖 httpx**。不 import torch / sentence-transformers / transformers，
   连 `try: import` 都不写——一旦写下去，CI 和答辩机器就得背 5GB 依赖。
2. **批大小 32、超时 30s、失败重试 2 次**（指数退避 1s、2s），只重试超时、
   连接失败与 408/429/5xx。
3. **同一文本第二次不许再发网络请求**：key = `sha256(model + text)`，命中
   `embedding_cache` 就直接读本地字节。入库重跑、评测重跑、同一句 query 反复检索
   都不再产生费用。
4. **向量一律 L2 归一化后再存**。检索侧只做点积，等价于余弦相似度，
   不需要在每次查询时重复算模长。

未配置接口时**不静默退化**：抛 `EmbeddingUnavailableError`，让调用方记失败原因；
想在没有 key 的机器上跑通链路，请显式设 `MOCK_MODE=1`。
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence

import httpx
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import EmbeddingCache

logger = logging.getLogger(__name__)

# 单次 SQL 里 IN 的参数个数上限留余量（SQLite 默认 999）
_KEY_QUERY_CHUNK = 500
# 缓存写入按批提交：一次 203 行的多值 INSERT 会撞上 SQLite 的变量数上限
_CACHE_WRITE_CHUNK = 100
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class EmbeddingUnavailableError(RuntimeError):
    """向量接口不可用（未配置、鉴权失败、超时、维度不符）；不返回零向量冒充成功。"""


def is_configured() -> bool:
    """三项齐全才算可用；缺项时给出可操作的错误，而不是退化成随机向量。"""
    return bool(
        settings.embed_base_url.strip()
        and settings.embed_api_key.strip()
        and settings.embed_model.strip()
    )


def missing_config() -> list[str]:
    return [
        name
        for name, value in (
            ("EMBED_BASE_URL", settings.embed_base_url),
            ("EMBED_API_KEY", settings.embed_api_key),
            ("EMBED_MODEL", settings.embed_model),
        )
        if not value.strip()
    ]


def model_name() -> str:
    """当前向量模型的名字；写进 `chunks.embed_model`，换模型时能识别出旧向量。"""
    if settings.mock_mode:
        return f"stub-hash-{settings.embed_dim}"
    return settings.embed_model or "unconfigured"


def dim() -> int:
    return int(settings.embed_dim)


def encode_documents(texts: Sequence[str], *, session: Session | None = None) -> np.ndarray:
    """把一批文本编码成归一化向量，返回 shape=(n, dim) 的 float32 数组。

    缓存读写用独立的短会话：调用方可能正握着一个写事务，共用会话会把它的半成品
    一起提交；SQLite 在 WAL 模式下读不阻塞写，独立会话是安全的。
    """
    items = [text or "" for text in texts]
    if not items:
        return np.zeros((0, dim()), dtype=np.float32)

    if settings.mock_mode:
        from app.llm import mock  # 局部导入：mock 反过来要用 client 的事件类型

        return mock.stub_embed(items)

    if not is_configured():
        raise EmbeddingUnavailableError(
            "未配置向量接口：请在 .env 填 " + "、".join(missing_config())
            + "；没有 key 时设 MOCK_MODE=1 用本地桩模型跑通链路"
        )

    keys = [_cache_key(text) for text in items]
    cached = _read_cache(keys)
    missing = [index for index, key in enumerate(keys) if key not in cached]
    if missing:
        fresh = _request([items[index] for index in missing])
        _write_cache([(keys[index], items[index], fresh[row]) for row, index in enumerate(missing)])
        for row, index in enumerate(missing):
            cached[keys[index]] = fresh[row]

    matrix = np.vstack([cached[key] for key in keys]).astype(np.float32)
    return _normalize(matrix)


def encode_query(query: str) -> np.ndarray:
    """编码查询文本；检索指令前缀只在 query 侧加。"""
    return encode_documents([settings.embed_query_instruction + query])[0]


def cache_size(session: Session | None = None) -> int:
    """缓存条目数，供 bench / 状态展示用。"""
    from sqlalchemy import func

    statement = select(func.count()).select_from(EmbeddingCache)
    if session is not None:
        return int(session.execute(statement).scalar_one())
    with SessionLocal() as own_session:
        return int(own_session.execute(statement).scalar_one())


def _cache_key(text: str) -> str:
    """sha256(model + text)：换模型后同一段文本会重新算，缓存不会串味。"""
    payload = f"{model_name()}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_cache(keys: list[str]) -> dict[str, np.ndarray]:
    """批量读缓存；未启用缓存时返回空字典。"""
    if not settings.embed_cache_enabled:
        return {}
    found: dict[str, np.ndarray] = {}
    with SessionLocal() as session:
        for start in range(0, len(keys), _KEY_QUERY_CHUNK):
            batch = keys[start : start + _KEY_QUERY_CHUNK]
            rows = session.execute(
                select(EmbeddingCache.key, EmbeddingCache.vector).where(
                    EmbeddingCache.key.in_(batch)
                )
            ).all()
            for key, blob in rows:
                found[key] = _from_blob(blob)
    return found


def _write_cache(items: list[tuple[str, str, np.ndarray]]) -> None:
    """写缓存（先删后插，跨 SQLite / PostgreSQL 都成立）。

    两个必须处理的细节，都是真实语料逼出来的：

    - **按 key 去重**：手册里重复段落（页眉残留、同一张表出现在两章）很常见，
      同一批里出现两个相同 key 会撞 `UNIQUE constraint failed: embedding_cache.key`；
    - **分批提交**：203 行 × 5 列的多值 INSERT 会超过 SQLite 的变量数上限，
      整批失败会让缓存白写（向量仍然返回，但下次还要再付费）。
    """
    if not settings.embed_cache_enabled or not items:
        return
    model = model_name()
    rows_by_key = {
        key: {
            "key": key,
            "model": model,
            "dim": int(vector.shape[0]),
            "vector": _to_blob(vector),
            "text_chars": len(text),
        }
        for key, text, vector in items
    }
    rows = list(rows_by_key.values())
    try:
        with SessionLocal() as session:
            for start in range(0, len(rows), _CACHE_WRITE_CHUNK):
                batch = rows[start : start + _CACHE_WRITE_CHUNK]
                session.execute(
                    delete(EmbeddingCache).where(
                        EmbeddingCache.key.in_([row["key"] for row in batch])
                    )
                )
                session.execute(EmbeddingCache.__table__.insert(), batch)
            session.commit()
    except Exception as exc:  # 缓存是加速手段，写不进去不影响本次结果
        logger.warning("写入 embedding_cache 失败（本次仍返回向量）：%s", exc)


def _request(texts: list[str]) -> np.ndarray:
    """按批调用接口；任何一批失败都整体失败，不返回半成品矩阵。"""
    batches = [
        texts[start : start + max(1, settings.embed_batch_size)]
        for start in range(0, len(texts), max(1, settings.embed_batch_size))
    ]
    vectors: list[np.ndarray] = []
    for index, batch in enumerate(batches, start=1):
        logger.info("向量请求 %d/%d：%d 条文本", index, len(batches), len(batch))
        vectors.append(_post_batch(batch))
    return np.vstack(vectors).astype(np.float32)


def _post_batch(batch: list[str]) -> np.ndarray:
    """一次 HTTP 请求（含重试），返回该批的 (n, dim) 数组。"""
    payload = {"model": settings.embed_model, "input": batch, "encoding_format": "float"}
    attempts = max(1, settings.embed_max_retries + 1)
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(settings.embed_timeout_seconds)) as client:
                response = client.post(
                    _endpoint(),
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {settings.embed_api_key}",
                        "Content-Type": "application/json",
                    },
                )
            if response.status_code != 200:
                snippet = " ".join(response.text.split())[:300]
                last_error = f"HTTP {response.status_code}：{snippet}"
                if response.status_code not in _RETRYABLE_STATUS and response.status_code < 500:
                    raise EmbeddingUnavailableError(f"向量接口拒绝请求（{last_error}）")
                raise httpx.HTTPStatusError(last_error, request=response.request, response=response)
            return _parse_batch(response.json(), len(batch))
        except EmbeddingUnavailableError:
            raise
        except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= attempts:
                break
            backoff = 2 ** (attempt - 1)  # 1s、2s
            logger.warning("向量接口失败（%s），%ds 后重试（第 %d 次）", last_error, backoff, attempt)
            time.sleep(backoff)
    raise EmbeddingUnavailableError(
        f"向量接口调用失败（{settings.embed_model}，已重试 {attempts - 1} 次）：{last_error}"
    )


def _parse_batch(payload: object, expected: int) -> np.ndarray:
    """解析 OpenAI 兼容返回：`{"data": [{"embedding": [...], "index": 0}, ...]}`。"""
    if not isinstance(payload, dict):
        raise EmbeddingUnavailableError("向量接口返回的不是 JSON 对象")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        message = payload.get("error") or payload
        raise EmbeddingUnavailableError(f"向量接口返回异常：{str(message)[:300]}")

    ordered = sorted(
        enumerate(data),
        key=lambda item: int(item[1].get("index", item[0])) if isinstance(item[1], dict) else item[0],
    )
    vectors: list[list[float]] = []
    for _, item in ordered:
        if not isinstance(item, dict) or "embedding" not in item:
            raise EmbeddingUnavailableError("向量接口返回项缺少 embedding 字段")
        vectors.append([float(value) for value in item["embedding"]])

    if len(vectors) != expected:
        raise EmbeddingUnavailableError(
            f"向量接口返回条数不符：期望 {expected}，实际 {len(vectors)}"
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    if settings.embed_dim and matrix.shape[1] != settings.embed_dim:
        raise EmbeddingUnavailableError(
            f"向量维度不符：接口返回 {matrix.shape[1]} 维，EMBED_DIM 配置为 {settings.embed_dim}；"
            "换模型后必须同步改 EMBED_DIM 并重建索引"
        )
    return matrix


def _endpoint() -> str:
    """base_url 写到 `/v1` 或直接写到 `/embeddings` 都能拼对。"""
    base = settings.embed_base_url.strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量保持为零（避免除零产生 NaN）。"""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _to_blob(vector: np.ndarray) -> bytes:
    """float32 原始字节，读写都不做 JSON 序列化。"""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float32)
