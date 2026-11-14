"""检索评测指标：Recall@K、分位数耗时。

只计算能被验证的东西。这里定义两级命中：

- **文档级**：top_k 里存在一个块，其文档名在金标文档集合内。要求最松，
  三份手册同源、内容重叠时，文档级指标会偏高，不能单独用来判断检索质量。
- **页码级**：top_k 里存在一个块，其 (文档名, 页码) 命中金标页。这才是真指标。

两级一起报，避免用一个宽松数字自我安慰。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_document(filenames: Sequence[str], gold_docs: set[str]) -> bool:
    """top_k 的文档名里是否包含任一金标文档。"""
    return any(name in gold_docs for name in filenames)


def hit_page(pairs: Sequence[tuple[str, int | None]], gold_pairs: set[tuple[str, int]]) -> bool:
    """top_k 的 (文档名, 页码) 是否命中金标页。"""
    return any(
        pair[1] is not None and (pair[0], int(pair[1])) in gold_pairs for pair in pairs
    )


def recall_at_k(hit_flags: Iterable[bool]) -> float:
    """Recall@K：把每条查询的 0/1 命中取平均。空集合返回 0，不返回 1。"""
    flags = list(hit_flags)
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def percentile(values: Sequence[float], ratio: float) -> float:
    """线性插值分位数；用于报 P95 耗时，避免只看平均值被个别慢查询带偏。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(1.0, ratio))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)
