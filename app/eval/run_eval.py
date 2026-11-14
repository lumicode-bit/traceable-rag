"""评测运行器：三种检索配置 × 一份题库，出对比表 + 失败案例，并生成 `results/report.md`。

与 `app/retrieval/bench.py` 的区别：

- `bench` 用页级金标（smoke_queries），适合「大方向对不对」；
- 本模块用**块级金标**（`questions.jsonl` 里每条题都指向出题时依据的那个 chunk），
  才能回答「答案有没有真的进上下文」——页级指标会把「同页的接线图」也算命中。

输出是**唯一权威报告**：`results/report.md`。数字只在这里生成，README/DESIGN 引用它，
不再手工抄第二份。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.db import SessionLocal, init_db
from app.retrieval import MODES, search_with_ranks

DEFAULT_QUERIES = "data/eval/questions.jsonl"
DEFAULT_OUT = "results/report.md"
CITATIONS_JSON = "data/eval/p3_citations.json"

TYPE_LABELS = {
    "lookup": "查表型",
    "reasoning": "推理型",
    "multi_doc": "跨文档综合",
    "unanswerable": "无答案（不参与召回）",
}


@dataclass
class Question:
    """一道评测题：`gold` 为空表示「库里没有答案」，不参与召回统计。"""

    id: str
    type: str
    query: str
    gold: list[dict] = field(default_factory=list)

    @property
    def gold_chunks(self) -> set[int]:
        return {int(item["chunk_id"]) for item in self.gold if item.get("chunk_id") is not None}

    @property
    def gold_pages(self) -> set[tuple[str, int]]:
        return {
            (item["doc"], int(item["page"]))
            for item in self.gold
            if item.get("doc") and item.get("page") is not None
        }

    @property
    def gold_docs(self) -> set[str]:
        return {item["doc"] for item in self.gold if item.get("doc")}


@dataclass
class ModeReport:
    """一种检索模式的汇总。"""

    mode: str
    total: int = 0
    hit_chunk: int = 0
    hit_page: int = 0
    hit_doc: int = 0
    latencies: list[float] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def recall_chunk(self) -> float:
        return self.hit_chunk / self.total if self.total else 0.0

    @property
    def recall_page(self) -> float:
        return self.hit_page / self.total if self.total else 0.0

    @property
    def recall_doc(self) -> float:
        return self.hit_doc / self.total if self.total else 0.0

    @property
    def median_ms(self) -> float:
        """用中位数而不是平均值：远端接口偶发重试会把平均值拉到 P95 之上，
        出现「平均 882ms、P95 483ms」这种自相矛盾的表。"""
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(int(len(ordered) * 0.95), len(ordered) - 1)
        return ordered[index]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    questions = _load(Path(args.queries))
    if not questions:
        print(f"[错误] 题库为空或不存在：{args.queries}", file=sys.stderr)
        return 2

    answerable = [item for item in questions if item.gold]
    skipped = len(questions) - len(answerable)
    print(f"题库 {len(questions)} 条（可评分 {len(answerable)} 条，无答案 {skipped} 条不参与召回）")

    init_db()
    reports = [_run_mode(mode, answerable, args.top_k) for mode in args.modes]
    for report in reports:
        print(
            f"{report.mode:<15} chunk {report.recall_chunk:.3f}  页码 {report.recall_page:.3f}  "
            f"中位 {report.median_ms:.0f}ms"
        )

    markdown = _render(reports, questions, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(f"\n报告已写入 {out.as_posix()}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run_eval",
        description="块级金标的三配置对比评测，并生成 results/report.md",
    )
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--failures", type=int, default=10, help="失败案例条数上限")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    for mode in args.modes:
        if mode not in MODES:
            parser.error(f"未知模式：{mode}（可选 {'、'.join(MODES)}）")
    if args.top_k <= 0:
        parser.error("--top-k 必须为正整数")
    return args


def _load(path: Path) -> list[Question]:
    if not path.is_file():
        return []
    items: list[Question] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(
            Question(
                id=raw.get("id") or f"q{len(items) + 1:03d}",
                type=raw.get("type") or "unknown",
                query=raw["query"],
                gold=raw.get("gold") or [],
            )
        )
    return items


def _run_mode(mode: str, questions: list[Question], top_k: int) -> ModeReport:
    report = ModeReport(mode=mode)
    with SessionLocal() as session:
        for index, question in enumerate(questions):
            started = time.perf_counter()
            try:
                hits = search_with_ranks(question.query, top_k=top_k, mode=mode, session=session)
            except Exception as exc:  # 单题失败不该中断整轮
                print(f"  ! {question.id} 检索失败：{type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            elapsed = (time.perf_counter() - started) * 1000
            # 第一条要付索引构建/分词的冷启动，计进去会让「平均」比「P95」还大
            if index > 0:
                report.latencies.append(elapsed)
            report.total += 1

            ids = [hit.chunk_id for hit in hits]
            got_chunk = bool(question.gold_chunks & set(ids))
            got_page = any(
                (hit.chunk.document.filename, hit.chunk.page_no) in question.gold_pages
                for hit in hits
                if hit.chunk.document is not None
            )
            got_doc = any(
                hit.chunk.document is not None
                and hit.chunk.document.filename in question.gold_docs
                for hit in hits
            )
            report.hit_chunk += int(got_chunk)
            report.hit_page += int(got_page)
            report.hit_doc += int(got_doc)

            bucket = report.by_type.setdefault(question.type, {"total": 0, "chunk": 0})
            bucket["total"] += 1
            bucket["chunk"] += int(got_chunk)

            if not got_chunk:
                report.failures.append(
                    {
                        "id": question.id,
                        "type": question.type,
                        "query": question.query,
                        "gold": _gold_text(question),
                        "top1": _hit_text(hits[0]) if hits else "（无结果）",
                        "verdict": _verdict(got_page, got_doc),
                    }
                )
    return report


def _gold_text(question: Question) -> str:
    return "、".join(
        f"{item['doc']} p.{item['page']}" for item in question.gold if item.get("doc")
    ) or "—"


def _hit_text(hit) -> str:  # noqa: ANN001 - SearchHit
    document = hit.chunk.document
    name = document.filename if document else "未知文档"
    return f"{name} p.{hit.chunk.page_no}"


def _verdict(got_page: bool, got_doc: bool) -> str:
    if got_page:
        return "**页码命中、块未命中**（同页其它块挤掉了答案块）"
    if got_doc:
        return "文档命中、页码不同"
    return "未命中"


def _render(reports: list[ModeReport], questions: list[Question], args: argparse.Namespace) -> str:
    answerable = sum(1 for item in questions if item.gold)
    types = sorted({item.type for item in questions if item.gold})
    best = max(reports, key=lambda item: item.recall_chunk)

    lines: list[str] = []
    lines.append("# 评测报告")
    lines.append("")
    lines.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M')}（由 `python -m app.eval.run_eval` 自动生成）")
    lines.append(f"> 向量：`{settings.embed_model or '未配置'}`（{settings.embed_dim} 维）")
    lines.append(f"> 题库：`{args.queries}`，{len(questions)} 条（可评分 {answerable} 条）")
    lines.append(f"> 指标：块级 Recall@{args.top_k}——金标块是否真的进了上下文；页码级/文档级仅作对照")
    cache_state = "开启（重复问题命中缓存，延迟偏乐观）" if settings.embed_cache_enabled else "关闭（每次查询都走一次真实 embedding）"
    lines.append(f"> 向量缓存：{cache_state}")
    lines.append("")

    lines.append(f"## 一、三种配置对比（{answerable} 题）")
    lines.append("")
    lines.append(f"| 模式 | 块级 Recall@{args.top_k} | 页码级 | 文档级 | 中位耗时 | P95 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for report in reports:
        lines.append(
            f"| `{report.mode}` | **{report.recall_chunk:.3f}** | {report.recall_page:.3f} | "
            f"{report.recall_doc:.3f} | {report.median_ms:.0f}ms | {report.p95_ms:.0f}ms |"
        )
    lines.append("")

    if types:
        lines.append(f"### 按题型拆开（`{best.mode}`）")
        lines.append("")
        lines.append(f"| 题型 | 题数 | 块级命中 | 块级 Recall@{args.top_k} |")
        lines.append("| --- | --- | --- | --- |")
        for kind in types:
            bucket = best.by_type.get(kind, {"total": 0, "chunk": 0})
            recall = bucket["chunk"] / bucket["total"] if bucket["total"] else 0.0
            lines.append(
                f"| {TYPE_LABELS.get(kind, kind)} | {bucket['total']} | {bucket['chunk']} | {recall:.3f} |"
            )
        lines.append("")

    lines.append(f"## 二、失败案例（`{best.mode}`，最多 {args.failures} 条）")
    lines.append("")
    if not best.failures:
        lines.append("无失败案例。")
    else:
        lines.append("| 题号 | 问题 | 金标位置 | 首条结果 | 判定 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in best.failures[: args.failures]:
            lines.append(
                f"| {item['id']} | {_cell(item['query'])} | {_cell(item['gold'])} | "
                f"{_cell(item['top1'])} | {item['verdict']} |"
            )
        lines.append("")
        lines.append(
            "「页码命中、块未命中」是最值得看的一类：检索到了对的页，但答案所在的那一段没进上下文。"
            "页级指标会把这类算成成功，块级指标才能把它暴露出来。"
        )
    lines.append("")

    lines.extend(_citations_section())
    lines.extend(_boundary_section(len(questions), answerable, args.top_k, len(types)))
    return "\n".join(lines)


def _citations_section() -> list[str]:
    """生成侧数字直接读验收产物，避免第二次手工抄写。"""
    path = Path(CITATIONS_JSON)
    if not path.is_file():
        return ["## 三、生成与引用", "", f"未找到 `{CITATIONS_JSON}`，跳过。", ""]
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    config = payload.get("config", {})
    accuracy = summary.get("accuracy")
    accuracy_text = "—" if accuracy is None else f"{accuracy * 100:.0f}%"
    return [
        "## 三、生成与引用",
        "",
        f"模型 `{config.get('model', '未知')}`，另跑 {summary.get('questions', 0)} 题验收"
        f"（含 {summary.get('unanswerable_total', 0)} 条无答案题）。",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| 作答 | {summary.get('answered', 0)} 题 |",
        f"| 引用（去重后） | {summary.get('unique_citations', 0)} 个 |",
        f"| 不存在的引用 | {summary.get('invalid_citations', 0)} 个 |",
        f"| 引用准确率 | {accuracy_text} |",
        f"| 无答案题拒答 | {summary.get('unanswerable_refused', 0)}/{summary.get('unanswerable_total', 0)} |",
        _support_row(summary),
        "",
        f"明细：`{CITATIONS_JSON}`。",
        "",
    ]


def _support_row(summary: dict) -> str:
    """支撑性一行；没有数据时也要说清楚「没测」，不能留空让人以为满分。"""
    support = summary.get("support") or {}
    checked = int(support.get("checked") or 0)
    if not checked:
        return "| 结论支撑性（规则版） | 未测 |"
    ratio = support.get("ratio") or 0.0
    return (
        f"| 结论支撑性（规则版） | {support.get('supported', 0)}/{checked} 条可判定结论的数值/型号"
        f"在其引用块中找到（{ratio * 100:.0f}%），{support.get('unknown', 0)} 条为定性描述判不了 |"
    )


def _boundary_section(total: int, answerable: int, top_k: int, type_count: int) -> list[str]:
    return [
        "## 四、这套数字的边界",
        "",
        f"- **{answerable} 条可评分题，{type_count} 类**。样本量比 10 条时可信得多，但仍是"
        "「同一批文档、同一批出题人（模型）+ 人工复核」，没有跨设备、跨版本的泛化验证。",
        "- **金标来自出题时的块**，不是从检索结果反推的；因此块级指标没有「照着答案找证据」的偏差。",
        "- **块级 Recall 仍然只说明上下文里有没有答案**，不说明模型有没有用对、引用得准不准。"
        "后者由 `tests/check_citations.py` 单独管。",
        _latency_note(),
        "- **重排没有独立评测**：`hybrid_rerank` 在 `RERANK_ENABLED=false` 时是透传，"
        "与 `hybrid` 的差异纯属噪声。要评估重排必须先配 `RERANK_*` 再重跑本命令。",
        "",
    ]


def _latency_note() -> str:
    """耗时口径必须跟着缓存开关走，否则报告里的数字会被自己打脸。"""
    if settings.embed_cache_enabled:
        return (
            "- **耗时口径**：缓存开启，`vector` 先跑并付掉 embedding 往返，后两个模式命中缓存，"
            "显示的是本地检索时间（几毫秒）。同一行数字不能横向比较，"
            "「用户问一句新问题」的体验请看 `vector` 那一行。"
        )
    return (
        "- **耗时口径**：缓存关闭，**三种模式都各自走了一次真实 embedding**，所以表中的耗时都是冷查询。"
        "行间差异来自网络抖动（同一套代码两次运行能差 30%~70%），不代表算法本身慢；"
        "`hybrid` 那一行的 P95 偏高是重试造成的。首条查询的冷启动不计入统计。"
    )


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
