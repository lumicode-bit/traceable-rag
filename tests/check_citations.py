"""引用准确性校验：把评测集的 10 个问题跑一遍，逐条核对回答里的来源是否真实存在。

校验口径（与 `app/llm/citations.py` 一致）：

- 从回答里抽出所有 `[文档名 p.页码]`；
- 逐个和**本次这道题检索到的片段**比对，文档名与页码都对得上才算有效；
- 引用准确率 = 有效引用 / 去重后的引用总数；
- 另外单独检查「该拒答的题是否拒答」：库里确实没有的答案必须回「资料中未找到相关内容」。

用法::

    python tests/check_citations.py                     # 真实模型跑 10 题
    python tests/check_citations.py --mode vector --top-k 8
    MOCK_MODE=1 python tests/check_citations.py         # 无 key：向量与对话都用桩模型
    python tests/check_citations.py --mock              # 只替换对话模型，验证校验逻辑本身
    python tests/check_citations.py --mock --hallucinate  # 故意编一个来源，验证能被抓出来
    python tests/check_citations.py --out data/eval/p3_citations.json

退出码：0 全部通过；1 存在幻觉引用 / 该拒答而未拒答 / 有题目报错；2 环境或配置问题。

注意：`--mock` 与 `MOCK_MODE=1` 只证明管道与校验器正确，**不能**代表真实模型的引用准确率。
桩模型只要有片段就作答，因此「该拒答而未拒答」在这两种模式下几乎必然失败——这是预期，
也是这个脚本没被放水的地方。真实准确率必须用配置好的 `.env` 跑一次不带 `--mock` 的命令。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 支持 `python tests/check_citations.py` 直接运行（此时 sys.path[0] 是 tests/）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.eval.dataset import EvalQuery, load_queries  # noqa: E402
from app.llm import client, embed, mock, pipeline  # noqa: E402
from app.llm import citations  # noqa: E402
from app.llm.citations import REFUSAL_MARKER  # noqa: E402
from app.models import Chunk  # noqa: E402
from app.retrieval import MODES  # noqa: E402

DEFAULT_QUERIES = "data/eval/smoke_queries.jsonl"

# 库里确实没有的问题：手册不写序列号规则，也不写价格与报价。
# 这两题必须拒答，编出任何来源都算失败。
DEFAULT_UNANSWERABLE: tuple[str, ...] = (
    "CP600-4T22G 的出厂序列号编码规则是什么，第几位代表生产年份",
    "这台设备的质量保证期和保修期外的上门维修报价是多少钱",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        queries = load_queries(args.queries)
    except (ValueError, OSError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if getattr(args, "limit", 0):
        queries = queries[: args.limit]
        print(f"[提示] --limit {args.limit}：只跑前 {len(queries)} 题（CI 冒烟用）")

    init_db()
    if _guard_index() != 0:
        return 2

    # 假模型是进程级替换，跑完必须还原：同一进程里再跑一次真实调用不能被污染
    original_stream = client.stream_chat
    if args.mock:
        _install_mock(hallucinate=args.hallucinate)
        print("[提示] --mock 模式：仅替换对话模型，只验证校验逻辑本身")
    elif not client.is_available():
        print(
            "[错误] 对话模型未配置：" + "、".join(client.missing_config()) + "\n"
            "       请在 .env 填写 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，\n"
            "       或设 MOCK_MODE=1 用本地桩模型（结果不代表真实模型效果）。",
            file=sys.stderr,
        )
        return 2

    if not settings.mock_mode and not embed.is_configured():
        print(
            "[错误] 向量接口未配置：" + "、".join(embed.missing_config()) + "\n"
            "       请在 .env 填写 EMBED_* 三项，或设 MOCK_MODE=1 用本地桩向量。",
            file=sys.stderr,
        )
        return 2

    try:
        return _run(args, queries)
    finally:
        client.stream_chat = original_stream


def _run(args: argparse.Namespace, queries: list[EvalQuery]) -> int:
    """跑全部题目并输出报告；返回退出码（0 通过 / 1 未通过）。"""
    print("=" * 78)
    print("引用准确性校验")
    print("=" * 78)
    print(f"评测集：{Path(args.queries).as_posix()}（{len(queries)} 题）")
    if settings.mock_mode:
        model_text = "MOCK_MODE=1（本地桩模型，结果不代表真实模型效果）"
    else:
        model_text = f"对话 {settings.llm_model} / 向量 {embed.model_name()}"
    print(f"检索：mode={args.mode}，top_k={args.top_k}；{model_text}")
    print()

    rows = [_run_query(item, args) for item in queries]
    # 题库里已经带了 `expect=refusal` 的题就不必再跑兜底探针，否则分母里会有重复主题
    has_refusal_items = any(item.expect == "refusal" for item in queries)
    unanswerable = (
        [] if args.skip_unanswerable or has_refusal_items else _run_unanswerable(args)
    )
    if has_refusal_items:
        print(f"[提示] 题库自带 {sum(1 for item in queries if item.expect == 'refusal')} 条拒答题，"
              "跳过脚本内置的兜底探针")

    _print_rows(rows)
    summary = _summarize(rows, unanswerable)
    _print_summary(summary)

    if args.out:
        _write_report(args.out, args, rows, unanswerable, summary)

    return 0 if summary["passed"] else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python tests/check_citations.py", description="回答引用真伪校验"
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="评测集 jsonl 路径")
    parser.add_argument("--mode", default=settings.chat_mode, help=f"检索模式：{','.join(MODES)}")
    parser.add_argument("--top-k", type=int, default=settings.chat_top_k, help="检索片段数")
    parser.add_argument("--mock", action="store_true", help="用本地假模型替代真实调用")
    parser.add_argument(
        "--hallucinate", action="store_true", help="仅配合 --mock：故意编造一个来源"
    )
    parser.add_argument("--skip-unanswerable", action="store_true", help="跳过拒答题检查")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（CI 冒烟用）")
    parser.add_argument("--out", default="", help="把结果写成 JSON 文件")
    args = parser.parse_args(argv)

    if args.mode not in MODES:
        parser.error(f"未知模式：{args.mode}，可选 {','.join(MODES)}")
    if args.top_k <= 0:
        parser.error("--top-k 必须为正整数")
    return args


def _guard_index() -> int:
    """库空或没有任何向量时直接退出：否则检索恒为空，所有题都会被判成「拒答」而假通过。"""
    with SessionLocal() as session:
        total = int(session.execute(select(func.count()).select_from(Chunk)).scalar_one())
        embedded = int(
            session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
            ).scalar_one()
        )
    if total == 0:
        print(
            "[错误] 库内没有任何分块，先跑入库：\n"
            "       python -m app.ingest.pipeline data/raw\n"
            "       python -m app.retrieval.bench --rebuild",
            file=sys.stderr,
        )
        return 2
    if embedded == 0:
        print(
            "[错误] 库内分块没有向量，先补建索引：python -m app.retrieval.bench --rebuild",
            file=sys.stderr,
        )
        return 2
    if embedded < total:
        print(f"[警告] {total - embedded}/{total} 个分块缺少向量，检索结果可能不完整")
    return 0


def _run_query(item: EvalQuery, args: argparse.Namespace) -> dict[str, Any]:
    """跑一题：检索 → 生成 → 校验，返回该题的核对结果。"""
    try:
        result = pipeline.answer_once(item.query, args.mode, args.top_k)
    except Exception as exc:  # 单题失败不中断整轮，但要在汇总里算作失败
        return {
            "id": item.query_id,
            "question": item.query,
            "status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "answer": "",
            "citations": 0,
            "unique": 0,
            "invalid": [],
            "accuracy": None,
        }
    check = result["citation_check"]
    return {
        "id": item.query_id,
        "question": item.query,
        "status": result["answer_status"],
        "answer": result["answer"],
        "citations": result["citations"],
        "occurrences": check.get("occurrences", 0),
        "unique": check.get("unique", 0),
        "invalid": check.get("invalid", []),
        "accuracy": check.get("accuracy"),
        "usage": result["usage"],
        "timings": result["timings"],
        "finish_reason": result.get("finish_reason"),
        "error": (result["error"] or {}).get("message"),
        "expect": item.expect,
        "support": result.get("support_check") or {},
    }


def _run_unanswerable(args: argparse.Namespace) -> list[dict[str, Any]]:
    """拒答题：片段里没有答案时必须拒答，给出任何结论都算编。"""
    results: list[dict[str, Any]] = []
    for question in DEFAULT_UNANSWERABLE:
        try:
            result = pipeline.answer_once(question, args.mode, args.top_k)
            answer = result["answer"]
            # 用与 pipeline 同一套判定：结论段只有「未找到」且全文无引用才算拒答。
            # 用 `REFUSAL_MARKER in answer` 会把「答完再补一句边界」误算成拒答。
            refused = citations.is_refusal(answer)
            reason = "" if refused else "未拒答：回答里没有出现「资料中未找到相关内容」"
        except Exception as exc:
            answer, refused = "", False
            reason = f"{type(exc).__name__}: {exc}"
        results.append(
            {"question": question, "answer": answer, "refused": refused, "reason": reason}
        )
    return results


def _summarize(rows: list[dict[str, Any]], unanswerable: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总结论。**门禁只看「有没有编造」**，误拒单独记为质量指标。

    两类失败性质不同，混在一个布尔值里会让门禁失效：

    - **硬失败**：幻觉引用、该拒答却作答、作答但零引用、调用/生成异常。
      这些违背系统对用户的承诺（可溯源、不编），必须红。
    - **软失败**：该答的题被拒答、回答被 max_tokens 截断。
      这是「宁可不答也不编」策略的代价，属于质量指标——记录下来逐条给人看，
      但**不阻塞 CI**。否则 CI 会因为 1/48 的误拒长期红，红久了就没人看，门禁等于没有。
    """
    unique_total = sum(row["unique"] for row in rows)
    invalid_total = sum(len(row["invalid"]) for row in rows)
    valid_total = unique_total - invalid_total
    accuracy = (valid_total / unique_total) if unique_total else None

    problems: list[str] = []  # 硬：违背契约
    quality: list[str] = []  # 软：质量指标，不阻塞
    refusal_rows = [row for row in rows if row.get("expect") == "refusal"]
    for row in rows:
        if row["status"] in {"exception"}:
            problems.append(f"{row['id']} 调用失败：{row.get('error')}")
            continue
        if row["status"] == "error":
            problems.append(f"{row['id']} 生成失败：{row.get('error')}")
        for issue in row["invalid"]:
            problems.append(f"{row['id']} 幻觉引用 {issue['label']}：{issue['reason']}")
        if row.get("expect") == "refusal":
            # 拒答题：作答就是编，哪怕它引用了真实片段（片段本身不含答案）
            if row["status"] != "refused":
                problems.append(f"{row['id']} 该拒答却作答了：{row.get('question')}")
            continue
        if row["status"] == "answered" and row["unique"] == 0:
            problems.append(f"{row['id']} 有作答但没有任何引用，无法溯源")
        if row["status"] == "refused":
            quality.append(f"{row['id']} 被拒答，但评测集认为该题有答案（误拒）")
        if row.get("finish_reason") == "length":
            quality.append(f"{row['id']} 回答被 max_tokens 截断，引用可能不完整（调大 LLM_MAX_TOKENS）")
    for item in unanswerable:
        if not item["refused"]:
            problems.append(f"该拒答而未拒答：{item['question']}（{item['reason']}）")

    # 拒答率的分母 = 题库里的拒答题 + 脚本自带的兜底探针
    refused_in_file = sum(1 for row in refusal_rows if row["status"] == "refused")
    answerable_total = len(rows) - len(refusal_rows)

    return {
        "questions": len(rows),
        "answerable_questions": answerable_total,
        "refusal_questions": len(refusal_rows),
        "answered": sum(1 for row in rows if row["status"] == "answered"),
        "refused": sum(1 for row in rows if row["status"] == "refused"),
        "failed": sum(1 for row in rows if row["status"] in {"error", "exception"}),
        "unique_citations": unique_total,
        "valid_citations": valid_total,
        "invalid_citations": invalid_total,
        "accuracy": accuracy,
        "unanswerable_total": len(refusal_rows) + len(unanswerable),
        "unanswerable_refused": refused_in_file + sum(1 for item in unanswerable if item["refused"]),
        "support": _support_totals(rows),
        # 硬失败：决定退出码；软失败：只记录
        "problems": problems,
        "quality_issues": quality,
        "hard_fail": bool(problems) or (accuracy is not None and accuracy < 1.0),
        "soft_fail": len(quality),
        "passed": not problems and (accuracy is None or accuracy >= 1.0),
    }


def _support_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总「结论有没有被引用块支撑」。不参与 passed 判定：规则版会漏判定义改写。"""
    totals = {"claims": 0, "checked": 0, "supported": 0, "unsupported": 0, "unknown": 0}
    for row in rows:
        support = row.get("support") or {}
        for key in totals:
            totals[key] += int(support.get(key) or 0)
    totals["ratio"] = (totals["supported"] / totals["checked"]) if totals["checked"] else None
    return totals


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print("逐题")
    print("-" * 78)
    for row in rows:
        accuracy = "—" if row["accuracy"] is None else f"{row['accuracy'] * 100:.0f}%"
        timings = row.get("timings") or {}
        cost = (
            f"{timings.get('retrieval_ms', 0):.0f}+{timings.get('generation_ms', 0):.0f}ms"
            if timings
            else "—"
        )
        print(
            f"{row['id']}  {row['status']:<9} 引用 {row['unique']:>2} 个  无效 "
            f"{len(row['invalid']):>2} 个  准确率 {accuracy:>5}  耗时 {cost}"
        )
        for issue in row["invalid"]:
            print(f"      × {issue['label']}  ← {issue['reason']}")
    print()


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"题目：{summary['questions']} 题（作答 {summary['answered']}，拒答 {summary['refused']}，"
          f"失败 {summary['failed']}）")
    accuracy = summary["accuracy"]
    accuracy_text = "无可校验引用" if accuracy is None else f"{accuracy * 100:.1f}%"
    print(f"引用：去重后 {summary['unique_citations']} 个，有效 {summary['valid_citations']} 个，"
          f"不存在 {summary['invalid_citations']} 个 → 引用准确率 {accuracy_text}")
    print(f"拒答：{summary['unanswerable_refused']}/{summary['unanswerable_total']} 题正确拒答")
    support = summary.get("support") or {}
    if support.get("checked"):
        ratio = support["ratio"] or 0.0
        print(
            f"支撑性：{support['checked']} 条结论可判定，其中 {support['supported']} 条经二次校验确认"
            f"（{ratio * 100:.1f}%），{support['unsupported']} 条的数值/型号在其引用块里找不到；"
            f"另有 {support['unknown']} 条只有定性描述、规则判不了"
        )
    quality = summary.get("quality_issues") or []
    if summary["problems"]:
        print("\n[硬失败] 违背「可溯源、不编」的承诺，必须修：")
        for problem in summary["problems"]:
            print(f"  - {problem}")
    if quality:
        print(f"\n[质量问题] {len(quality)} 条，记录但不阻塞门禁：")
        for item in quality:
            print(f"  - {item}")
    print()
    if summary["passed"]:
        print(f"[结论] 门禁通过（0 条编造；质量项 {len(quality)} 条见上）")
    else:
        print("[结论] 未通过：上面「硬失败」每一条都必须修掉")


def _write_report(
    path: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    unanswerable: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """把明细写成 JSON，便于留档与对比多次运行；只写真实跑出来的数据。"""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "config": {
                    "queries": args.queries,
                    "mode": args.mode,
                    "top_k": args.top_k,
                    "model": settings.llm_model,
                    "mock": args.mock,
                },
                "summary": summary,
                "questions": rows,
                "unanswerable": unanswerable,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[记录] 明细已写入 {target.as_posix()}")


def _install_mock(hallucinate: bool) -> None:
    """把 `client.stream_chat` 换成本地桩模型，用来验证校验逻辑本身。

    桩模型只复述提示词里给出的 `<cite>` 标签：在真实检索结果上必然 100% 有效；
    加 `--hallucinate` 时额外编一个不存在的来源，用来确认脚本能把它抓出来。
    """
    client.stream_chat = lambda messages: mock.stream_stub(  # type: ignore[assignment]
        list(messages), hallucinate=hallucinate
    )


if __name__ == "__main__":
    raise SystemExit(main())
