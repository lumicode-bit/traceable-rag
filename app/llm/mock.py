"""不联网的桩模型：StubEmbedder + StubRerank + StubLLM，由 `MOCK_MODE=1` 启用。

存在的理由：没有 key 的机器（CI、答辩备用机、临时换环境）也要能跑通
「入库 → 检索 → 生成 → 引用校验」整条链路，证明管道与模型是解耦的。

三条自我约束：

1. **输出里明确标注 MOCK**。桩模型只是形状正确的假货，不能让人误以为看到了
   真实模型效果，回答末尾会写一行说明。
2. **完全确定**。向量由字符 n-gram 的 blake2b 哈希累加而来，不用 `hash()`
   （受 PYTHONHASHSEED 影响），进程重启、换机器都得到同一向量。
3. **引用规则与真实模型完全一致**：只引用提示词里给出的 `<cite>` 标签，
   没有片段时拒答。`--hallucinate` 之外不编造任何来源，因此用桩模型跑
   `tests/check_citations.py` 得到的 100% 只说明校验器与管道正确。

它不参与任何真实评测，也不该被写进「模型效果」的数字里。
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

import numpy as np

from app.config import settings
from app.llm.citations import REFUSAL_MARKER
from app.llm.client import LlmEvent, TokenUsage

MOCK_NOTE = "（MOCK 模式：本回答由本地桩模型生成，仅用于验证链路，不代表真实模型效果）"

_CITE_RE = re.compile(r"<cite>(?P<label>.*?)</cite>")
_PASSAGE_SPLIT_RE = re.compile(r"\[片段\s*\d+\]")
_BODY_RE = re.compile(r"正文：\s*\n?(?P<body>.*)", re.DOTALL)

# 中文 2~3 字组合足以区分「油压偏低」与「母线电压」这类短句
_NGRAM_SIZES = (2, 3)


def stub_embed(texts: list[str]) -> np.ndarray:
    """字符 n-gram 哈希向量（L2 归一化），用于在没有向量接口时跑通检索。

    共享 n-gram 越多的文本，向量内积越大——语义远远谈不上，但「词面相近的块排在前面」
    足以让链路验证有效。
    """
    dim = max(8, int(settings.embed_dim))
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        normalized = "".join(text.split())
        for size in _NGRAM_SIZES:
            for start in range(max(len(normalized) - size + 1, 0)):
                gram = normalized[start : start + size]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def stub_rerank_scores(query: str, documents: list[str]) -> list[float]:
    """按字符 n-gram 重合率打分，只为让 hybrid_rerank 这条链路能被执行到。"""
    query_grams = _grams(query)
    if not query_grams:
        return [0.0] * len(documents)
    scores: list[float] = []
    for document in documents:
        grams = _grams(document)
        overlap = len(query_grams & grams)
        scores.append(overlap / max(len(query_grams), 1))
    return scores


def stream_chat(messages: list[dict[str, str]]) -> Iterator[LlmEvent]:
    """生产入口：与 `client.stream_chat` 同签名、同事件类型，内部不联网。"""
    return stream_stub(messages)


def stream_stub(
    messages: list[dict[str, str]], *, hallucinate: bool = False
) -> Iterator[LlmEvent]:
    """按固定模板产出带引用的回答；`hallucinate=True` 时故意编一个不存在的来源。

    逐块吐出 `LlmEvent`，形状与真实客户端一致，因此上层（pipeline / SSE / 校验脚本）
    完全感知不到差别。
    """
    question = _last_user_content(messages)
    passages = _passages(question)
    answer = _compose(passages, hallucinate=hallucinate)

    for start in range(0, len(answer), 24):
        yield LlmEvent(kind="delta", text=answer[start : start + 24])
    yield LlmEvent(
        kind="done",
        text=answer,
        usage=TokenUsage(
            prompt_tokens=max(len("".join(m.get("content", "") for m in messages)) // 2, 1),
            completion_tokens=max(len(answer) // 2, 1),
            total_tokens=max(len(answer) // 2, 1) * 2,
            estimated=True,
        ),
        finish_reason="stop",
    )


def _compose(passages: list[tuple[str, str]], *, hallucinate: bool) -> str:
    """模板回答：结论 → 排查步骤 → 来源，与系统提示词要求的格式完全一致。"""
    if not passages:
        return f"{REFUSAL_MARKER}\n缺少的信息：本次检索没有返回任何片段"

    label, body = passages[0]
    excerpt = body.strip().split("\n")[0][:60] or "片段给出了相关描述"
    steps = [f"1. 按 {label} 核对故障现象与告警码。"]
    if len(passages) > 1:
        second = passages[1][0]
        steps.append(f"2. 按 {second} 复测并记录结果。")
    else:
        steps.append("2. 按上条来源逐项复测并记录结果。")

    sources = [label]
    if len(passages) > 1:
        sources.append(passages[1][0])
    if hallucinate:
        sources.append("[这份文档根本不存在.pdf p.999]")

    return (
        f"结论：{excerpt} {label}\n"
        "排查步骤\n"
        + "\n".join(steps)
        + "\n来源\n"
        + "\n".join(sources)
        + "\n\n"
        + MOCK_NOTE
    )


def _passages(user_message: str) -> list[tuple[str, str]]:
    """从提示词里还原 (cite 标签, 正文首行) 列表。"""
    passages: list[tuple[str, str]] = []
    for block in _PASSAGE_SPLIT_RE.split(user_message)[1:]:
        label_match = _CITE_RE.search(block)
        if label_match is None:
            continue
        body_match = _BODY_RE.search(block)
        body = body_match.group("body") if body_match else ""
        passages.append((label_match.group("label").strip(), body))
    return passages


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _grams(text: str) -> set[str]:
    normalized = "".join(text.split())
    grams: set[str] = set()
    for size in _NGRAM_SIZES:
        for start in range(max(len(normalized) - size + 1, 0)):
            grams.add(normalized[start : start + size])
    return grams
