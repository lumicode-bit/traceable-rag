"""逐条核验题库金标：题目里的确切标识符，能不能在金标块里找到。

为什么要做这件事：q014 暴露了一个更根本的问题——它的答案行在块 112，金标却写着 113。
金标错一条，压低的不是某一方，而是整份评测的绝对数值。金标不可信，前面所有指标都不成立。

做法复用 `app/llm/support.py` 的事实抽取（功能码、型号、故障码、端子号），
把题目里的标识符抽出来，到金标块正文里查。查不到 = 可疑，人工看一眼就知道是
「金标指错块」还是「题目问法里带了个不该带的词」。

用法：

    python tools/check_gold.py                     # 用 data/eval/questions.jsonl
    python tools/check_gold.py --queries x.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.llm.support import detect_tokens  # noqa: E402
from app.models import Chunk, Document  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    suspicious: list[tuple[str, list[str], str, list[int]]] = []
    checked = skipped = 0

    with SessionLocal() as session:
        for row in rows:
            gold = row.get("gold") or []
            if not gold:
                skipped += 1
                continue
            # 只核对「带确切标识符」的题；纯口语问题（断电后等多久）抽不出 token，核不了
            tokens = [t for t in detect_tokens(row["query"]) if len(t) >= 3]
            if not tokens:
                skipped += 1
                continue

            bodies: list[str] = []
            ids: list[int] = []
            for item in gold:
                chunk = session.get(Chunk, int(item["chunk_id"]))
                if chunk is None:
                    continue
                ids.append(chunk.id)
                # 文档名也算进可匹配文本：题目里的型号可能只出现在文档名上
                name = chunk.document.filename if chunk.document else ""
                bodies.append(f"{name} {chunk.content}")
            if not bodies:
                suspicious.append((row["id"], tokens, "金标块在库里不存在", ids))
                continue

            checked += 1
            haystack = "".join("".join(body.split()) for body in bodies).lower()
            missing = [t for t in tokens if t.lower() not in haystack]
            if missing:
                suspicious.append((row["id"], missing, row["query"][:38], ids))

    print(f"核验 {checked} 题（跳过 {skipped} 题：无金标 / 题目里没有确切标识符）")
    print(f"可疑 {len(suspicious)} 条")
    for qid, missing, question, ids in suspicious:
        print(f"  {qid} 金标块 {ids} 缺 {missing}　｜{question}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用事实抽取核验金标块是否包含题目里的标识符")
    parser.add_argument("--queries", default="data/eval/questions.jsonl", help="题库 jsonl")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
