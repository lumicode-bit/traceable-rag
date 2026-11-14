"""检索评测命令行：跑评测集，输出三种模式的 Recall@5 与平均耗时。

用法::

    python -m app.retrieval.bench                          # 默认评测集与全部模式
    python -m app.retrieval.bench --top-k 10 --verbose      # 改 K 并逐题打印
    python -m app.retrieval.bench --trace q02               # 打印某题的融合过程
    python -m app.retrieval.bench --rebuild                 # 先重建向量与 BM25 索引
    python -m app.retrieval.bench --save                    # 结果写入 eval_runs / eval_results

两条纪律：
1. 评测集只读不写，绝不在跑完之后回头改金标；
2. 耗时统计排除首次加载（先跑一轮 warmup 丢弃），否则会把模型冷启动算进单条延迟。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, init_db
from app.eval import metrics
from app.eval.dataset import EvalQuery, load_queries
from app.llm import embed, rerank
from app.models import Chunk, EvalResult, EvalRun
from app.retrieval import MODES, bm25, rebuild_indexes, search_with_ranks, vector

DEFAULT_QUERIES = "data/eval/smoke_queries.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        queries = load_queries(args.queries)
    except (ValueError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    init_db()
    with SessionLocal() as session:
        if args.rebuild:
            stats = rebuild_indexes(session, force_embedding=True)
            print(f"[索引] 重建完成：向量 {stats['embedded']} 条，BM25 {stats['bm25_chunks']} 块")
        else:
            missing = session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
            ).scalar_one()
            if missing:
                print(f"[索引] 发现 {missing} 个块缺少向量，先补建…")
                stats = rebuild_indexes(session)
                print(f"[索引] 向量 {stats['embedded']} 条，BM25 {stats['bm25_chunks']} 块")

        status = _index_status(session)
        _print_header(args, queries, status)

        reports = [_run_mode(session, queries, mode, args.top_k) for mode in args.modes]
        _print_summary(reports, args.top_k)
        if args.verbose:
            _print_per_query(queries, reports)

        if args.trace:
            _print_trace(session, queries, args.trace, args.top_k)

        if args.save:
            run_id = _save_run(session, queries, reports, args)
            print(f"\n[记录] 已写入 eval_runs id={run_id}")

    _print_verdict(reports)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.bench", description="检索模式对比评测"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="评测集 jsonl 路径")
    parser.add_argument("--top-k", type=int, default=5, help="Recall@K 的 K，默认 5")
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help=f"要评测的模式，逗号分隔，默认全部：{','.join(MODES)}",
    )
    parser.add_argument("--verbose", action="store_true", help="逐题打印命中情况")
    parser.add_argument("--trace", default="", help="打印指定题目的融合过程（题号或 id）")
    parser.add_argument("--rebuild", action="store_true", help="评测前强制重建向量与 BM25")
    parser.add_argument("--save", action="store_true", help="把结果写入 eval_runs / eval_results")
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        parser.error(f"未知模式：{','.join(unknown)}，可选 {','.join(MODES)}")
    args.modes = modes
    if args.top_k <= 0:
        parser.error("--top-k 必须为正整数")
    return args


def _index_status(session: Session) -> dict[str, str]:
    total = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
    embedded = session.execute(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
    ).scalar_one()
    index = bm25.get_index(session)
    return {
        "chunks": f"{int(total)} 块",
        "vectors": f"{int(embedded)} 条",
        "bm25": f"{index.chunk_count} 块 / {index.token_total} 词元（{index.built_at_text}）",
    }


def _print_header(args: argparse.Namespace, queries: list[EvalQuery], status: dict[str, str]) -> None:
    print("=" * 78)
    print("检索评测")
    print("=" * 78)
    print(f"评测集：{Path(args.queries).as_posix()}（{len(queries)} 条，Recall@{args.top_k}）")
    print(f"索引：{status['chunks']}，向量 {status['vectors']}，BM25 {status['bm25']}")
    if not rerank.is_enabled():
        state = "关（RERANK_ENABLED=false，透传）"
    elif settings.mock_mode:
        state = "开（MOCK：本地 n-gram 打分，不代表真实重排）"
    elif rerank.is_configured():
        state = f"开（{settings.rerank_model}）"
    else:
        state = "开但未配置 RERANK_*，将降级为原序"
    print(f"重排：{state}")
    print(f"向量：{embed.model_name()}（{'MOCK' if settings.mock_mode else '远端接口'}，{embed.dim()} 维）")
    print(f"融合：RRF k={settings.rrf_k}，每路候选 {settings.retrieval_candidates}")
    print()


def _run_mode(session: Session, queries: list[EvalQuery], mode: str, top_k: int) -> dict:
    # warmup：把模型/索引的首次开销挡在计时之外
    warm = queries[0].query
    search_with_ranks(warm, top_k=top_k, mode=mode, session=session)

    doc_flags: list[bool] = []
    page_flags: list[bool] = []
    latencies: list[float] = []
    per_query: list[dict] = []

    for item in queries:
        started = time.perf_counter()
        hits = search_with_ranks(item.query, top_k=top_k, mode=mode, session=session)
        latencies.append((time.perf_counter() - started) * 1000)

        filenames = [hit.chunk.document.filename for hit in hits]
        pairs = [(hit.chunk.document.filename, hit.chunk.page_no) for hit in hits]
        doc_hit = metrics.hit_document(filenames, item.gold_docs)
        page_hit = metrics.hit_page(pairs, item.gold_pairs)
        doc_flags.append(doc_hit)
        page_flags.append(page_hit)
        per_query.append({"id": item.query_id, "doc": doc_hit, "page": page_hit, "hits": hits})

    return {
        "mode": mode,
        "recall_doc": metrics.recall_at_k(doc_flags),
        "recall_page": metrics.recall_at_k(page_flags),
        "avg_ms": sum(latencies) / len(latencies),
        "p95_ms": metrics.percentile(latencies, 0.95),
        "max_ms": max(latencies),
        "per_query": per_query,
    }


def _print_summary(reports: list[dict], top_k: int) -> None:
    print(f"{'模式':<14}{f'Recall@{top_k}(文档)':>16}{f'Recall@{top_k}(页码)':>16}"
          f"{'平均耗时':>10}{'P95':>9}{'最慢':>9}")
    print("-" * 78)
    for report in reports:
        print(
            f"{report['mode']:<14}{report['recall_doc']:>16.3f}{report['recall_page']:>16.3f}"
            f"{report['avg_ms']:>8.0f}ms{report['p95_ms']:>7.0f}ms{report['max_ms']:>7.0f}ms"
        )
    print()


def _print_per_query(queries: list[EvalQuery], reports: list[dict]) -> None:
    print("逐题命中（✓ 命中 / ✗ 未命中，括号内为文档级/页码级）")
    print("-" * 78)
    for index, item in enumerate(queries):
        marks = []
        for report in reports:
            detail = report["per_query"][index]
            mark = "✓" if detail["page"] else ("◐" if detail["doc"] else "✗")
            marks.append(f"{report['mode']}={mark}")
        print(f"{item.query_id}  {'  '.join(marks)}  {item.query[:34]}")
    print("  ✓ 页码命中（真命中）   ◐ 仅文档命中（内容对但没落到金标页）   ✗ 未命中")
    print()


def _print_trace(session: Session, queries: list[EvalQuery], trace: str, top_k: int) -> None:
    target = _pick_query(queries, trace)
    if target is None:
        print(f"[提示] 没有找到题目 {trace}，可用 id 或序号")
        return

    print("=" * 78)
    print(f"融合过程（{target.query_id}）：{target.query}")
    gold_text = "，".join(
        f"{Path(g.doc).stem} p{g.page}" if g.page else Path(g.doc).stem for g in target.gold
    )
    print(f"金标：{gold_text}")
    print(f"　　{target.note}")
    print("-" * 78)

    vector_hits = vector.search(session, target.query, top_k)
    bm25_hits = bm25.search(session, target.query, top_k)
    fused_hits = search_with_ranks(
        target.query, top_k=top_k, mode="hybrid_rerank" if rerank.is_enabled() else "hybrid",
        session=session,
    )

    print(f"【第一路】向量检索 top{top_k}（余弦相似度）")
    _print_route(session, vector_hits)
    print(f"\n【第二路】BM25 检索 top{top_k}（词法分数）")
    _print_route(session, bm25_hits)

    print(f"\n【融合】RRF（k={settings.rrf_k}）")
    print(f"  名次  RRF分数   向量    BM25   来源  文档/页            章节")
    for rank, hit in enumerate(fused_hits, start=1):
        vec_rank = f"#{hit.vector_rank}" if hit.vector_rank else "—"
        bm25_rank = f"#{hit.bm25_rank}" if hit.bm25_rank else "—"
        chunk = hit.chunk
        print(
            f"  {rank:>3}   {hit.score:.5f}  {vec_rank:>5}  {bm25_rank:>5}   "
            f"{hit.sources:<4}  {Path(chunk.document.filename).stem[:8]} p{chunk.page_no:<3} "
            f"{str(chunk.section or '')[:22]}"
        )
    print("  来源标记：V=仅向量命中，B=仅 BM25 命中，VB=两路都命中")
    print()


def _print_route(session: Session, hits: list[tuple[int, float]]) -> None:
    if not hits:
        print("  （无结果）")
        return
    for rank, (chunk_id, score) in enumerate(hits, start=1):
        chunk = session.get(Chunk, chunk_id)
        if chunk is None:
            continue
        print(
            f"  #{rank:<2} {score:>8.4f}  {Path(chunk.document.filename).stem[:8]} "
            f"p{chunk.page_no:<3} {str(chunk.section or '')[:34]}"
        )


def _pick_query(queries: list[EvalQuery], trace: str) -> EvalQuery | None:
    for index, item in enumerate(queries, start=1):
        if trace == item.query_id or trace == str(index):
            return item
    return None


def _save_run(
    session: Session, queries: list[EvalQuery], reports: list[dict], args: argparse.Namespace
) -> int:
    run = EvalRun(
        dataset_name=Path(args.queries).name,
        config={
            "top_k": args.top_k,
            "modes": args.modes,
            "rrf_k": settings.rrf_k,
            "candidates": settings.retrieval_candidates,
            "embedding_model": embed.model_name(),
            "mock_mode": settings.mock_mode,
            "rerank_enabled": rerank.is_enabled(),
        },
        status="finished",
        finished_at=datetime.now(timezone.utc),
    )
    session.add(run)
    session.flush()

    for report in reports:
        for index, item in enumerate(queries):
            detail = report["per_query"][index]
            session.add(
                EvalResult(
                    run_id=run.id,
                    question=item.query,
                    expected_chunk_ids=None,
                    retrieved_chunk_ids=[hit.chunk_id for hit in detail["hits"]],
                    answer=None,
                    metrics={
                        "query_id": item.query_id,
                        "mode": report["mode"],
                        "hit_doc": detail["doc"],
                        "hit_page": detail["page"],
                        "recall_doc": report["recall_doc"],
                        "recall_page": report["recall_page"],
                        "avg_ms": round(report["avg_ms"], 2),
                    },
                )
            )
    session.commit()
    return int(run.id)


def _print_verdict(reports: list[dict]) -> None:
    if not reports:
        return
    best = max(reports, key=lambda report: (report["recall_page"], -report["avg_ms"]))
    print(f"[结论] 页码级最佳：{best['mode']}（Recall={best['recall_page']:.3f}，"
          f"平均 {best['avg_ms']:.0f}ms）")
    if best["recall_page"] < 0.6:
        print("[警告] 页码级 Recall 低于 0.6：先查切块与分词，不要调融合参数。")
    slow = [report for report in reports if report["mode"] != "hybrid_rerank" and report["avg_ms"] > 500]
    if slow:
        print("[警告] 以下模式单条平均耗时超过 500ms：" + "、".join(r["mode"] for r in slow))


if __name__ == "__main__":
    raise SystemExit(main())
