"""诊断工具：按文档与页码范围打印切块结果，用来定位「答案块为什么没被切出来」。

只读，不写库、不改索引。典型用法（在项目根目录）：

    python tools/diag_chunks.py --doc CP600 --from 30 --to 33
    python tools/diag_chunks.py --doc CP320 --from 17 --to 19 --chars 300
    python tools/diag_chunks.py --stats

`--stats` 打印全局分布：块长分位、跨页块数（开始页与结束页不一致无法直接判断，
这里用相邻块页码跳变做近似）、疑似表格块数、超长块数。
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
from pathlib import Path

DEFAULT_DB = "data/kb.db"


def _connect(path: str) -> sqlite3.Connection:
    """复制一份再读。

    直接以只读模式打开 WAL 模式的库会报 `disk I/O error`（缺 -shm 时无法建立读快照），
    而以读写模式打开又可能和服务进程抢锁、顺带触发 checkpoint。复制整个库副本最省事：
    2MB 的文件，拷一次几毫秒，且永远不会影响正在跑的服务。
    """
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"库文件不存在：{path}")
    workdir = Path(tempfile.mkdtemp(prefix="diag_chunks_"))
    target = workdir / source.name
    shutil.copy2(source, target)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, workdir / sidecar.name)
    return sqlite3.connect(target)


def cmd_list(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    sql = """
        select c.id, d.filename, c.page_no, c.section, length(c.content), c.content
        from chunks c join documents d on c.document_id = d.id
        where d.filename like ? and c.page_no between ? and ?
        order by c.page_no, c.id
    """
    rows = conn.execute(sql, (f"%{args.doc}%", args.from_page, args.to_page)).fetchall()
    if not rows:
        print("没有匹配的块，检查 --doc / 页码范围")
        return 1
    print(f"共 {len(rows)} 块（{args.doc}，p.{args.from_page}–{args.to_page}）\n")
    for cid, filename, page, section, size, content in rows:
        head = (content or "").replace("\n", " ")[: args.chars]
        print(f"[#{cid}] p.{page} | {size} 字 | {section or '（无小节）'}")
        print(f"    {head}")
        print()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = _connect(args.db)
    rows = conn.execute("select id, page_no, section, content from chunks order by id").fetchall()
    if not rows:
        print("库里没有块")
        return 1
    sizes = sorted(len(r[3] or "") for r in rows)
    n = len(sizes)

    def pct(p: float) -> int:
        return sizes[min(n - 1, int(n * p))]

    tables = [r for r in rows if "| ---" in (r[3] or "")]
    huge = [r for r in rows if len(r[3] or "") > 3000]
    no_section = [r for r in rows if not r[2]]
    print(f"块数 {n}")
    print(f"块长 中位 {pct(0.5)} / P90 {pct(0.9)} / 最长 {sizes[-1]}")
    print(f"含表格的块（出现 | --- 分隔行）{len(tables)}")
    print(f"超长块（>3000 字）{len(huge)}")
    print(f"无小节路径的块 {len(no_section)}")
    print("\n最长的 5 块：")
    for cid, page, section, content in sorted(rows, key=lambda r: -len(r[3] or ""))[:5]:
        print(f"  #{cid} p.{page} {len(content)} 字 | {section or '（无）'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="按文档/页码打印切块结果（只读）")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--doc", default="", help="文件名片段，如 CP600")
    parser.add_argument("--from", dest="from_page", type=int, default=1)
    parser.add_argument("--to", dest="to_page", type=int, default=9999)
    parser.add_argument("--chars", type=int, default=160, help="每块打印前 N 字")
    parser.add_argument("--stats", action="store_true", help="打印全局切块统计")
    args = parser.parse_args()
    return cmd_stats(args) if args.stats else cmd_list(args)


if __name__ == "__main__":
    raise SystemExit(main())
