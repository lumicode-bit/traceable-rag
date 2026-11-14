"""跨厂商手册压测：拿没见过的 PDF 直接入库，只记录「在哪一层崩」，不做任何调参。

为什么要有这个脚本：现有参数是在三份同源手册上调出来的，别人问一句「换个厂的行不行」
就答不上来。压测的价值不在分数，而在**失败清单**——它能告诉你哪一层最脆，
而且是答辩时唯一能说「我在 N 份陌生手册上验证过，已知失效模式有三种」的证据。

用法（不污染主库：默认写入独立的 SQLite 文件）：

    # 1. 把下载来的手册丢进 data/foreign/（不要放进 data/raw）
    # 2. 跑压测，不需要 API key
    python -m app.eval.stress_ingest
    # 3. 看 results/stress_report.md

刻意不建向量索引：这一步只测解析与切块（最脆的部分），不花 embedding 的钱。
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SRC = "data/foreign"
DEFAULT_OUT = "results/stress_report.md"
DEFAULT_DB = "data/foreign/stress.db"


@dataclass
class FileStats:
    """单个文件的解析体检结果。"""

    name: str
    status: str = "unknown"
    pages: int = 0
    ocr_pages: int = 0
    chunks: int = 0
    reason: str = ""
    chunk_lengths: list[int] = field(default_factory=list)
    table_chunks: int = 0
    short_line_blocks: int = 0
    broken_tables: int = 0      # 数据行列数对不上：真会读错
    merged_tables: int = 0      # 表头合并被拍平成空单元格：数据仍可读，属轻度

    @property
    def chars(self) -> int:
        return sum(self.chunk_lengths)

    @property
    def max_chunk(self) -> int:
        return max(self.chunk_lengths) if self.chunk_lengths else 0

    @property
    def median_chunk(self) -> int:
        return int(statistics.median(self.chunk_lengths)) if self.chunk_lengths else 0

    @property
    def ocr_ratio(self) -> float:
        return self.ocr_pages / self.pages if self.pages else 0.0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.src)
    if not src.is_dir():
        print(f"[错误] 目录不存在：{src}", file=sys.stderr)
        print("把下载来的手册放进这个目录再跑（不要放进 data/raw，避免污染主库）", file=sys.stderr)
        return 2

    # 关键顺序：先改环境变量再 import app.*，否则 `app.db` 的引擎已经按主库建好了，
    # 压测就会往生产库里写（第一次跑就踩过：文件被判成「重复」）。
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

    from app.db import SessionLocal, init_db  # noqa: PLC0415
    from app.ingest import loader  # noqa: PLC0415
    from app.ingest.pipeline import iter_files  # noqa: PLC0415

    files = iter_files(src)
    if not files:
        print(
            f"[错误] {src} 里没有支持的文件（{', '.join(sorted(loader.SUPPORTED_SUFFIXES))}）",
            file=sys.stderr,
        )
        return 2

    init_db()
    print(f"压测 {len(files)} 份文件 → {db_path.as_posix()}")
    print("-" * 70)

    stats: list[FileStats] = []
    with SessionLocal() as session:
        for path in files:
            item = _ingest_one(session, path, src)
            stats.append(item)
            print(
                f"{item.status:<8} {item.name[:34]:<34} "
                f"{item.pages:>3}页 OCR{item.ocr_pages:>2} {item.chunks:>3}块 {item.chars:>6}字"
                + (f"  ← {item.reason[:40]}" if item.reason else "")
            )

    markdown = _render(stats, src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print("-" * 70)
    print(f"报告已写入 {out.as_posix()}")
    _print_modes(stats)
    return 0 if all(item.status == "indexed" for item in stats) else 1


def _ingest_one(session, path: Path, root: Path) -> FileStats:  # noqa: ANN001 - Session
    from app.ingest.pipeline import ingest_file  # noqa: PLC0415

    item = FileStats(name=path.name)
    try:
        result = ingest_file(session, path, root)
    except Exception as exc:  # 解析崩了也要留在表里，这正是要记录的东西
        item.status = "crashed"
        item.reason = f"{type(exc).__name__}: {exc}"
        return item

    item.status = result.status if result.status != "skipped" else "indexed"
    item.pages = result.pages
    item.ocr_pages = result.ocr_pages
    item.chunks = result.chunks
    item.reason = result.reason or ""
    # 即使被跳过（重复文件）也要回来剖面：页数与块数以库里的记录为准，
    # 否则表里会出现「0 页 0 块但 18485 字」这种自相矛盾的行
    _profile(session, path, root, item)
    return item


def _profile(session, path: Path, root: Path, item: FileStats) -> None:  # noqa: ANN001 - Session
    """从库里的记录反推页数与三个最容易出问题的指标。"""
    from sqlalchemy import select  # noqa: PLC0415

    from app.models import Chunk, Document  # noqa: PLC0415

    relative = path.relative_to(root).as_posix()
    document = session.execute(
        select(Document).where(Document.source_path == relative)
    ).scalar_one_or_none()
    if document is None:
        return
    item.pages = item.pages or int(document.page_count or 0)
    item.ocr_pages = item.ocr_pages or int(document.ocr_pages or 0)
    contents = list(
        session.execute(select(Chunk.content).where(Chunk.document_id == document.id)).scalars()
    )
    item.chunks = item.chunks or len(contents)
    item.chunk_lengths = [len(text) for text in contents]
    for text in contents:
        if "|" in text:
            item.table_chunks += 1
            broken, merged = _table_profile(text)
            item.broken_tables += int(broken)
            item.merged_tables += int(merged)
        if _looks_like_column_smash(text):
            item.short_line_blocks += 1


def _table_profile(text: str) -> tuple[bool, bool]:
    """表格体检，返回 (数据行列错位, 表头合并被拍平)。

    两件事必须分开，否则清单会不可信：

    - **列错位（真会读错）**：多数数据行是 N 列，少数行列数差 ≥2。这才值得修。
    - **表头合并（轻度）**：表头里出现连续空单元格（`| 外形尺寸 |  |  |  |`），
      说明 PDF 里的跨列表头被拍平了。数据行仍然对齐，读表不受影响，
      但它会让模型误判「这一列是什么」。

    第一版只看「有几种列数」，连自己那份干净手册都报了一条——那种报警跑十份外来手册
    会攒出几十条噪音，反而毁了清单的可信度。
    """
    rows = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ")
    ]
    if len(rows) < 3:
        return (False, False)

    header = rows[0]
    data = rows[1:]
    widths = [row.count("|") for row in data]
    if not widths:
        return (False, False)

    # 多数数据行的列宽当基准（用众数，不用表头：表头本身可能就是被拍平的）
    modal = max(set(widths), key=widths.count)
    misaligned = sum(1 for width in widths if abs(width - modal) >= 2)
    broken = misaligned >= 2 and misaligned / len(widths) >= 0.2

    cells = [cell.strip() for cell in header.strip("|").split("|")]
    blanks = sum(1 for cell in cells if not cell)
    merged = blanks >= 2 and not broken
    return (broken, merged)


def _looks_like_column_smash(text: str) -> bool:
    """双栏排版被串成一行：出现大量 1~2 字的孤立片段，且整行很长。"""
    if len(text) < 200:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    tiny = sum(1 for line in lines if 0 < len(line.strip()) <= 2)
    return tiny / len(lines) > 0.3


def _render(stats: list[FileStats], src: Path) -> str:
    ok = [item for item in stats if item.status == "indexed"]
    lines = [
        "# 跨厂商手册压测报告",
        "",
        f"> 语料目录：`{src.as_posix()}`　文件数：{len(stats)}　成功：{len(ok)}",
        "> 本报告只测**解析与切块**，不建向量、不调任何参数——目的是暴露失效模式，不是刷分。",
        "",
        "## 一、逐份结果",
        "",
        "| 文件 | 状态 | 页数 | OCR 页 | 块数 | 总字数 | 中位块长 | 最长块 | 表格块 | 列错位（会读错） | 表头合并（轻度） | 疑似双栏串行 | 人工复核（待填） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in stats:
        lines.append(
            f"| {item.name} | {item.status} | {item.pages} | {item.ocr_pages} | {item.chunks} | "
            f"{item.chars} | {item.median_chunk} | {item.max_chunk} | {item.table_chunks} | "
            f"{item.broken_tables} | {item.merged_tables} | {item.short_line_blocks} |  |"
        )
    lines.append("")

    failed = [item for item in stats if item.status != "indexed"]
    if failed:
        lines += ["## 二、直接失败的文件", ""]
        for item in failed:
            lines.append(f"- **{item.name}**：{item.status} —— {item.reason or '无原因记录'}")
        lines.append("")

    lines += ["## 三、失效模式汇总", ""]
    lines += _mode_lines(stats)
    lines += [
        "",
        "## 四、怎么把这张表用起来",
        "",
        "1. 逐条确认是真失效还是误报（`疑似` 两类是启发式，必须人眼看一段原文），"
        "结论填进最后一列；**工具输出与人工判断必须分开记**，混在一起那份清单就不可信了；",
        "2. 把确认后的失效模式写进 `docs/DESIGN.md` 的第四节，并注明样本量；",
        "3. 只针对出现次数最多的那一类改造，不要同时动切块和提示词——分不清是谁的功劳。",
        "",
        "**不要**把本报告的分数当成检索指标：这里根本没有建索引。",
        "",
    ]
    return "\n".join(lines)


def _mode_lines(stats: list[FileStats]) -> list[str]:
    if not stats:
        return ["（无数据）"]
    ocr_heavy = [item for item in stats if item.ocr_ratio > 0.3]
    smashed = [item for item in stats if item.short_line_blocks > 0]
    broken = [item for item in stats if item.broken_tables > 0]
    merged = [item for item in stats if item.merged_tables > 0]
    failed = [item for item in stats if item.status != "indexed"]
    huge = [item for item in stats if item.max_chunk > 3000]

    return [
        f"- **解析失败**：{len(failed)}/{len(stats)} 份 —— 加密、扫描图无文字层、非标准 PDF 居多。",
        f"- **OCR 占比过高（>30% 页）**：{len(ocr_heavy)} 份 —— 这类文档的答案块召回质量会明显下降，"
        "因为 OCR 文本里错字多，BM25 与向量都受影响。",
        f"- **疑似双栏串行**：{len(smashed)} 份 —— 双栏/竖排被按水平线拼成一行，句子顺序会错乱。",
        f"- **表格列错位（会读错）**：{len(broken)} 份 —— 数据行列数对不上，模型照着错的表读会给出错误数值。",
        f"- **表头合并被拍平（轻度）**：{len(merged)} 份 —— 跨列表头变成空单元格，数据行仍对齐，"
        "但模型可能误解列含义；这类不必急着修。",
        f"- **超长块（>3000 字）**：{len(huge)} 份 —— 超过 `CONTEXT_MAX_CHARS` 会被截断，等于腰斩表格。",
    ]


def _print_modes(stats: list[FileStats]) -> None:
    for line in _mode_lines(stats):
        print(line.replace("- ", "", 1))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.stress_ingest",
        description="跨厂商手册压测：解析 + 切块体检，输出失败模式清单",
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"待测手册目录（默认 {DEFAULT_SRC}）")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"报告输出路径（默认 {DEFAULT_OUT}）")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"独立库文件（默认 {DEFAULT_DB}）")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
