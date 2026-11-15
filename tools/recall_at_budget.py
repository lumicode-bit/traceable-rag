"""等上下文预算下的召回：解决「固定 top-k 比较对小块不公平」的问题。

背景：表格按行拆块之后，块变短了，于是 `top-5` 在两种方案下代表的**上下文总量**差很多
（整块方案 5×700≈3500 字，拆块方案 5×150≈750 字）。在固定 k 下比较，等于让拆块方案
只带五分之一的材料上场，结论会偏向整块。

本脚本改为**固定字数预算**：按名次累加块长，到达预算为止，看金标块是否在其中。
这样「拆块」可以取到 top-20~25，和整块方案的 3500 字是同一个信息量。

用法：

    python tools/recall_at_budget.py --db data/kb.db --queries data/eval/questions.jsonl
    python tools/recall_at_budget.py --db data/kb_split.db --queries data/eval/questions_split.jsonl
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUDGETS = (1500, 2500, 3500, 5000)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(args.db).as_posix()}"

    from app.db import SessionLocal, init_db
    from app.eval.dataset import load_queries
    from app.retrieval import search_with_ranks

    init_db()
    queries = load_queries(args.queries)
    answerable = [item for item in queries if item.gold]

    stats = {budget: 0 for budget in BUDGETS}
    ranks: list[int] = []
    depths: list[int] = []

    with SessionLocal() as session:
        for item in answerable:
            hits = search_with_ranks(item.query, top_k=args.depth, mode=args.mode, session=session)
            gold_pairs = item.gold_pairs
            gold_docs = item.gold_docs

            cumulative = 0
            hit_rank = None
            hit_depth = None
            # 每题对每个预算只记一次：多金标题（跨文档、被拆开的表）不能重复计数，
            # 否则 Recall 会算出 1.104 这种大于 1 的值
            per_budget = {budget: False for budget in BUDGETS}
            for index, hit in enumerate(hits, start=1):
                cumulative += len(hit.chunk.content)
                inside = (
                    (hit.chunk.document.filename, hit.chunk.page_no) in gold_pairs
                    if hit.chunk.document
                    else False
                )
                if inside:
                    if hit_rank is None:
                        hit_rank = index
                        hit_depth = cumulative
                    for budget in BUDGETS:
                        if cumulative <= budget:
                            per_budget[budget] = True
            for budget, matched in per_budget.items():
                if matched:
                    stats[budget] += 1
            if hit_rank is not None:
                ranks.append(hit_rank)
                depths.append(hit_depth or 0)

    total = len(answerable)
    print(f"库：{args.db}　题库：{args.queries}　可评分 {total} 题　mode={args.mode}")
    print("-" * 66)
    print(f"{'上下文预算':<12}{'命中题数':>8}{'Recall':>10}{'约相当于 rank':>14}")
    for budget in BUDGETS:
        hit = stats[budget]
        print(f"{str(budget) + ' 字':<12}{hit:>8}{hit / total:>10.3f}")
    if ranks:
        avg_chunk = sum(depths) / len(depths) / max(1, len(depths) and 1)
        print("-" * 66)
        print(f"金标首次出现的平均名次：{sum(ranks) / len(ranks):.1f}")
        print(f"金标首次出现时已读入的平均字数：{sum(depths) / len(depths):.0f}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python tools/recall_at_budget.py",
        description="等上下文预算下的 Recall（固定总字数，而不是固定 top-k）",
    )
    parser.add_argument("--db", required=True, help="SQLite 库文件路径")
    parser.add_argument("--queries", required=True, help="题库 jsonl")
    parser.add_argument("--mode", default="hybrid", help="检索模式，默认 hybrid")
    parser.add_argument("--depth", type=int, default=40, help="最多看多少个候选，默认 40")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
