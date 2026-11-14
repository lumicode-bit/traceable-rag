"""入库流水线：解析 → 切块 → 入库。

用法：
    python -m app.ingest.pipeline data/raw
    python -m app.ingest.pipeline data/raw --sample 5

同一个文件按 sha256 去重：重复文件直接跳过，保证重复执行不会新增块。
单个文件失败只记录 failed_reason，不中断其它文件。
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.ingest import chunker, loader
from app.models import Chunk, Document

STATUS_PENDING = "pending"
STATUS_PARSING = "parsing"
STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"

TITLE_MAX_CHARS = 255
PATH_MAX_CHARS = 512
REASON_MAX_CHARS = 1000


@dataclass
class FileResult:
    """单个文件的处理结果，用于汇总统计。"""

    path: Path
    status: str  # indexed | skipped | failed
    pages: int = 0
    ocr_pages: int = 0
    chunks: int = 0
    reason: str | None = None


def sha256_of(path: Path) -> str:
    """按流式计算 sha256，避免大文件占满内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    """列出支持的文件，跳过隐藏文件与 Office 临时文件。"""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in loader.SUPPORTED_SUFFIXES
        and not path.name.startswith(("~$", "."))
    )


def ingest_file(session: Session, path: Path, root: Path) -> FileResult:
    """处理单个文件；异常一律落到 failed_reason 后返回。"""
    relative = path.relative_to(root).as_posix()
    digest = sha256_of(path)

    existing = session.scalar(select(Document).where(Document.sha256 == digest))
    if existing is not None and existing.status == STATUS_INDEXED:
        print(f"[跳过] 重复文件：{relative}（已入库 id={existing.id}）")
        return FileResult(path=path, status="skipped")
    if existing is not None:
        # 上次 failed / parsing / pending 的残留记录直接重建，保证重跑能修复问题
        print(f"[重试] 已有未完成记录：{relative}（状态={existing.status}，原因={existing.failed_reason}）")
        session.delete(existing)
        session.commit()

    document = Document(
        title=path.stem[:TITLE_MAX_CHARS],
        filename=path.name[:TITLE_MAX_CHARS],
        source_path=relative[:PATH_MAX_CHARS],
        doc_type=path.suffix.lower().lstrip("."),
        sha256=digest,
        status=STATUS_PARSING,
    )
    session.add(document)
    session.commit()

    print(f"[解析] {relative} …")
    try:
        parsed = loader.load(path)
        drafts = chunker.split(parsed)
        if not drafts:
            raise ValueError("解析成功但未切出任何文本块")

        session.add_all(
            Chunk(
                document_id=document.id,
                page_no=draft.page_no,
                section=draft.section,
                content=draft.content,
            )
            for draft in drafts
        )
        document.page_count = parsed.page_count
        document.ocr_pages = parsed.ocr_page_count
        document.status = STATUS_INDEXED
        document.failed_reason = None
        session.commit()
    except Exception as exc:  # 任何异常都要落到 failed_reason，不静默吞掉
        session.rollback()
        reason = f"{type(exc).__name__}: {exc}".strip()[:REASON_MAX_CHARS]
        failed = session.scalar(select(Document).where(Document.sha256 == digest))
        if failed is not None:
            failed.status = STATUS_FAILED
            failed.failed_reason = reason
            session.commit()
        print(f"[失败] {relative}：{reason}")
        return FileResult(path=path, status="failed", reason=reason)

    print(
        f"[入库] {relative}：{parsed.page_count} 页"
        f"（OCR {parsed.ocr_page_count} 页），{len(drafts)} 块"
    )
    return FileResult(
        path=path,
        status="indexed",
        pages=parsed.page_count,
        ocr_pages=parsed.ocr_page_count,
        chunks=len(drafts),
    )


def _print_report(session: Session, results: list[FileResult], root: Path) -> None:
    """输出统计：文件数、页数、OCR 页数、块数、失败清单。"""
    indexed = [item for item in results if item.status == "indexed"]
    skipped = [item for item in results if item.status == "skipped"]
    failed = [item for item in results if item.status == "failed"]

    total_chunks = session.scalar(select(func.count()).select_from(Chunk)) or 0
    total_docs = session.scalar(select(func.count()).select_from(Document)) or 0

    print()
    print("=" * 60)
    print("入库统计")
    print("=" * 60)
    print(f"扫描目录：{root}")
    print(f"文件数：{len(results)}（入库 {len(indexed)}，跳过重复 {len(skipped)}，失败 {len(failed)}）")
    print(f"总页数：{sum(item.pages for item in indexed)}")
    print(f"OCR 页数：{sum(item.ocr_pages for item in indexed)}")
    print(f"本次新增块数：{sum(item.chunks for item in indexed)}")
    print(f"库内累计：{total_docs} 份文档 / {total_chunks} 块")

    if failed:
        print()
        print(f"失败文件（{len(failed)} 个，未计入上方页数与块数）：")
        for item in failed:
            print(f"  - {item.path.name}：{item.reason}")
    print("=" * 60)


def _print_samples(session: Session, count: int) -> None:
    """随机抽样打印块内容，供人工核对切块质量。"""
    rows = session.execute(
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(func.random())
        .limit(count)
    ).all()

    print()
    print("=" * 60)
    print(f"随机抽样 {len(rows)} 个块")
    print("=" * 60)
    for index, (chunk, filename) in enumerate(rows, start=1):
        page = f"第 {chunk.page_no} 页" if chunk.page_no else "页号未知"
        print(f"\n【抽样 {index}】{filename} | {page} | 章节：{chunk.section or '（无）'}")
        print(f"字数：{len(chunk.content)}")
        print("-" * 60)
        print(chunk.content)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="把原始文档解析切块后写入数据库")
    parser.add_argument("source", nargs="?", default="data/raw", help="文档目录，默认 data/raw")
    parser.add_argument("--sample", type=int, default=0, help="结束后随机打印 N 个块")
    args = parser.parse_args(argv)

    root = Path(args.source)
    if not root.exists():
        print(f"[错误] 目录不存在：{root}")
        return 2

    files = iter_files(root)
    print(f"[扫描] {root}：找到 {len(files)} 个支持的文件")
    if not files:
        return 0

    try:
        init_db()
    except Exception as exc:
        print(f"[错误] 数据库初始化失败：{type(exc).__name__}: {exc}")
        return 1

    results: list[FileResult] = []
    with SessionLocal() as session:
        for path in files:
            results.append(ingest_file(session, path, root))
        _print_report(session, results, root)
        if args.sample > 0:
            _print_samples(session, args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
