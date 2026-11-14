"""从已入库的分块批量生成评测题候选：**agent 出候选，人删没依据的**。

为什么这么做：手写 60 道题要三天，而且写的人会不自觉地照着检索结果出题。
让模型逐块出「只能靠这一段回答」的题，金标天然就是该块的 `(文档名, 页码)`——
出题时还不知道检索结果，评起来才没有偏。

四类题的比例固定（`--quota` 可改）：

| 类型 | 来源 | 作用 |
| --- | --- | --- |
| `lookup` | 含表格/参数的块 | 查表型，考精确读数 |
| `reasoning` | 叙述型段落 | 需要把一两句话读懂再答 |
| `multi_doc` | 两份文档的同类章节 | 跨文档综合，考「别把 A 的页码安到 B 上」 |
| `unanswerable` | 人工给定主题 + 关键词核验 | 拒答率的分母，**至少 15%** |

生成完要做的事：`python -m app.eval.run_eval` 会把不能命中的题列出来，
人工再看一遍哪些是「题目问得不好」，哪些是「检索真的不行」——两者要分开处理。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.llm import client
from app.models import Chunk, Document

DEFAULT_OUT = "data/eval/questions.jsonl"

# 四类配额；无答案型按「至少 15%」倒推
DEFAULT_QUOTA = {"lookup": 27, "reasoning": 17, "multi_doc": 12, "unanswerable": 11}

QUESTION_PROMPT = """你在为工业设备手册编写评测题。下面给出手册原文的一段内容，
请写一道**只能靠这段原文回答**的问题。

硬性要求：
1. 用现场工人提问的口吻，不要出现「根据上文」「这段文字」「本段」这类字眼；
2. 问题里不许包含答案本身（如果答案是数值，问题里不能出现这个数值）；
3. **只问一个事实**，不许把两三个问题串起来（不要出现「分别」「各是多少」「对照说一下」）；
4. **答案必须完整地落在下面这段原文里**。如果回答它还需要另一段原文（例如问版本变更，
   而这里只有封面上的版本号），一律输出 answerable:false，不要猜；
5. 不要问「这一段讲了什么」这种概括性问题，要问具体事实；
6. 只输出一行 JSON，不要解释、不要代码块：
   {{"question": "问题", "answer": "原文里的答案", "answerable": true}}
7. 如果这段原文太碎，无法单独支撑一个具体问题，输出 {{"answerable": false}}。

文档：{doc}　页码：{page}　章节：{section}
原文：
{content}
"""

MULTI_DOC_PROMPT = """你在为工业设备手册编写「跨文档」评测题。下面给出两份手册中同一主题的两段原文。

请写一道需要**同时看两段**才能完整回答的问题，例如「两份手册对同一故障的处理办法有什么不同」。

硬性要求：
1. 用现场工人提问的口吻，不要出现「根据材料」「文中」这类字眼；
2. 只输出一行 JSON，不要解释：
   {{"question": "问题", "answer": "两段的要点对比", "answerable": true}}
3. 两段内容实际高度重复、问不出差异时，输出 {{"answerable": false}}。

【A】{doc_a} 第 {page_a} 页　{section_a}
{content_a}

【B】{doc_b} 第 {page_b} 页　{section_b}
{content_b}
"""

# 无答案型：主题必须是语料里确实没有的。生成后会再过一遍关键词核验。
# 选题原则：只挑「设备说明书天然不写」的东西（价格、库存、证书、工期、报价……），
# 不挑「可能写也可能不写」的（联网能力、海拔降额），否则会把可答题误判成应拒答。
UNANSWERABLE_TOPICS = [
    ("采购与价格", "这台设备的采购单价、批量折扣和付款方式是怎样的"),
    ("备件库存", "常用备件的安全库存量和补货周期是多少"),
    ("人员与培训", "操作人员需要持什么证，培训学时和培训计划是怎样的"),
    ("序列号规则", "设备出厂序列号每一位代表什么，怎么从序列号看生产日期"),
    ("包装与运输", "整机出厂包装尺寸、毛重多少，一个 40 尺集装箱能装几台"),
    ("认证与合规", "CE 证书编号是多少，发证机构是谁，有效期到哪年"),
    ("保修条款", "整机质保期多久，保修范围包含哪些部件"),
    ("安装工期", "从到货到验收，标准安装工期是几个工作日"),
    ("能耗测算", "这台设备在满载工况下的年耗电量与电费估算"),
    ("竞品对比", "与同功率段的竞品相比，这台设备的效率优势是多少"),
    ("使用反馈", "已交付客户的故障率和返修率是多少，有没有客户案例"),
]

# 关键词核验：**只放决定性词**。宽泛词（如「柜」「培训」）会把本来无答案的题误杀，
# 这也是第一轮只凑到 6 条的根因。
ABSENT_PROBES = {
    "采购与价格": ["采购单价", "报价", "折扣", "付款方式"],
    "备件库存": ["安全库存", "补货周期", "备件库存"],
    "人员与培训": ["培训学时", "持证上岗", "培训计划"],
    "序列号规则": ["序列号", "生产日期"],
    "包装与运输": ["包装尺寸", "毛重", "净重", "集装箱"],
    "认证与合规": ["证书编号", "发证机构", "认证有效期"],
    "保修条款": ["质保期", "保修期", "保修期限"],
    "安装工期": ["安装工期", "施工周期", "调试工期"],
    "能耗测算": ["年耗电量", "电费"],
    "竞品对比": ["竞品", "效率优势"],
    "使用反馈": ["客户案例", "故障率", "返修率"],
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.quota = _quota(args.quota)  # 允许用 --quota lookup=20,... 覆盖默认配额
    init_db()

    with SessionLocal() as session:
        chunks = _load_chunks(session)
        if not chunks:
            print("[错误] 库里没有分块，先跑入库：python -m app.ingest.pipeline data/raw", file=sys.stderr)
            return 2

        questions: list[dict] = []
        questions += _gen_single(session, chunks, "lookup", args.quota["lookup"], args)
        questions += _gen_single(session, chunks, "reasoning", args.quota["reasoning"], args)
        questions += _gen_multi_doc(session, chunks, args.quota["multi_doc"], args)
        questions += _gen_unanswerable(session, chunks, args.quota["unanswerable"], args)

    questions = _finalize(questions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in questions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    _print_summary(questions, out)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.generate_questions",
        description="从已入库分块生成评测题候选（需在 .env 配好 LLM_*）",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"输出 jsonl（默认 {DEFAULT_OUT}）")
    parser.add_argument(
        "--quota",
        default="",
        help="覆盖配额，如 lookup=20,reasoning=10,multi_doc=8,unanswerable=8",
    )
    parser.add_argument("--limit", type=int, default=0, help="每类最多出几题（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印抽样计划，不调模型")
    return parser.parse_args(argv)


def _quota(text: str) -> dict[str, int]:
    if not text.strip():
        return dict(DEFAULT_QUOTA)
    quota = dict(DEFAULT_QUOTA)
    for pair in text.split(","):
        key, _, value = pair.partition("=")
        key = key.strip()
        if key not in quota:
            raise SystemExit(f"未知类型：{key}（可选 {'、'.join(quota)}）")
        quota[key] = int(value)
    return quota


def _load_chunks(session: Session) -> list[dict]:
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.page_no,
            Chunk.section,
            Chunk.content,
            Document.filename,
        )
        .join(Document, Document.id == Chunk.document_id)
        .order_by(Chunk.id)
    ).all()
    return [
        {
            "chunk_id": int(row[0]),
            "page": row[1],
            "section": row[2] or "",
            "content": row[3],
            "doc": row[4],
        }
        for row in rows
    ]


def _gen_single(
    session: Session, chunks: list[dict], kind: str, want: int, args: argparse.Namespace
) -> list[dict]:
    """单块出题：lookup 优先表格块，reasoning 优先叙述块。"""
    pool = [c for c in chunks if _pool_ok(c, kind)]
    targets = _spread(pool, want)
    print(f"[{kind}] 候选块 {len(pool)} 个，抽 {len(targets)} 个")
    if args.dry_run:
        for item in targets:
            print(f"   - {item['doc']} p.{item['page']} {item['section'][:40]}")
        return []

    picked: list[dict] = []
    for index, chunk in enumerate(targets, start=1):
        if args.limit and len(picked) >= args.limit:
            break
        prompt = QUESTION_PROMPT.format(
            doc=chunk["doc"],
            page=chunk["page"],
            section=chunk["section"] or "（无）",
            content=chunk["content"][:1800],
        )
        payload = _ask(prompt)
        question = (payload or {}).get("question", "").strip()
        if not question or not payload.get("answerable", True):
            print(f"   [{index}/{len(targets)}] 跳过（模型认为不可单独成题）")
            continue
        picked.append(_item(kind, question, chunk, payload.get("answer", "")))
        print(f"   [{index}/{len(targets)}] {question[:40]}")
    return picked


def _gen_multi_doc(
    session: Session, chunks: list[dict], want: int, args: argparse.Namespace
) -> list[dict]:
    """跨文档：按「同一章节标题」在两份不同文档里配对。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = _section_key(chunk["section"])
        if key:
            groups[key].append(chunk)

    pairs: list[tuple[dict, dict]] = []
    for members in groups.values():
        by_doc: dict[str, dict] = {}
        for member in members:
            by_doc.setdefault(member["doc"], member)
        if len(by_doc) >= 2:
            docs = list(by_doc.values())[:2]
            pairs.append((docs[0], docs[1]))

    print(f"[multi_doc] 找到 {len(pairs)} 组跨文档同章节配对，要用 {min(want, len(pairs))} 组")
    if args.dry_run:
        for left, right in pairs[:want]:
            print(f"   - {left['section'][:30]} : {left['doc']} ↔ {right['doc']}")
        return []

    picked: list[dict] = []
    for left, right in pairs[:want]:
        if args.limit and len(picked) >= args.limit:
            break
        prompt = MULTI_DOC_PROMPT.format(
            doc_a=left["doc"],
            page_a=left["page"],
            section_a=left["section"] or "（无）",
            content_a=left["content"][:900],
            doc_b=right["doc"],
            page_b=right["page"],
            section_b=right["section"] or "（无）",
            content_b=right["content"][:900],
        )
        payload = _ask(prompt)
        question = (payload or {}).get("question", "").strip()
        if not question or not payload.get("answerable", True):
            print(f"   跳过（两段太像）：{left['section'][:30]}")
            continue
        item = _item("multi_doc", question, left, payload.get("answer", ""))
        item["gold"] = [
            {"chunk_id": left["chunk_id"], "doc": left["doc"], "page": left["page"]},
            {"chunk_id": right["chunk_id"], "doc": right["doc"], "page": right["page"]},
        ]
        picked.append(item)
        print(f"   {question[:40]}")
    return picked


def _gen_unanswerable(
    session: Session, chunks: list[dict], want: int, args: argparse.Namespace
) -> list[dict]:
    """无答案型：人工给主题，再用关键词核验「语料里确实没有」。"""
    corpus = "\n".join(chunk["content"] for chunk in chunks)
    picked: list[dict] = []
    for topic, question in UNANSWERABLE_TOPICS:
        if args.limit and len(picked) >= args.limit:
            break
        hits = [word for word in ABSENT_PROBES.get(topic, []) if word in corpus]
        if hits:
            print(f"[unanswerable] 跳过「{topic}」：语料里出现了 {hits}，不算无答案")
            continue
        picked.append(
            {
                "id": "",
                "type": "unanswerable",
                "query": question,
                "gold": [],
                # 让 check_citations 知道这题「应该拒答」，而不是「应该有引用」
                "expect": "refusal",
                "note": f"自动生成：{topic}（关键词核验：语料无 {'/'.join(ABSENT_PROBES.get(topic, []))}）",
                "source": "topic-list",
            }
        )
        print(f"[unanswerable] 收录：{question[:36]}")
        if len(picked) >= want:
            break
    if len(picked) < want:
        print(f"[警告] 无答案型只凑到 {len(picked)} 条（目标 {want}）")
    return picked


def _ask(prompt: str) -> dict | None:
    """调一次对话模型并解析 JSON；失败返回 None，不中断整批。"""
    result = client.complete([{"role": "user", "content": prompt}])
    if not result.ok:
        print(f"   ! 调用失败：{result.error.as_text() if result.error else '未知'}")
        return None
    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _item(kind: str, question: str, chunk: dict, answer: str) -> dict:
    return {
        "id": "",
        "type": kind,
        "query": question,
        "gold": [{"chunk_id": chunk["chunk_id"], "doc": chunk["doc"], "page": chunk["page"]}],
        "note": f"自动生成，答案依据：{answer[:80]}" if answer else "自动生成",
        "source": "llm",
    }


def _pool_ok(chunk: dict, kind: str) -> bool:
    text = chunk["content"]
    if len(text) < 200 or len(text) > 3000:
        return False
    # 安全注意事项/安全声明 通篇是「必须断电」这类套话，问不出具体事实
    if _section_key(chunk["section"]).startswith("安全"):
        return False
    has_table = "|" in text
    return has_table if kind == "lookup" else not has_table


def _spread(pool: list[dict], want: int) -> list[dict]:
    """均匀取样：先按章节去重（同一章节最多 3 题），再等距抽取。

    不去重的后果很直观：参数表有 100+ 块，抽出来的 27 题会全是「某参数默认值是多少」。
    """
    if want <= 0:
        return []
    per_section: dict[str, int] = defaultdict(int)
    filtered = []
    for chunk in pool:
        key = _section_key(chunk["section"]) or chunk["doc"]
        if per_section[key] >= 3:
            continue
        per_section[key] += 1
        filtered.append(chunk)

    source = filtered if len(filtered) >= want else pool
    if len(source) <= want:
        return source
    step = len(source) / want
    return [source[int(index * step)] for index in range(want)]


def _section_key(section: str) -> str:
    """把 `6 故障处理 > 6.2 常见故障及处理方法` 归一成 `常见故障及处理方法` 用于跨文档配对。"""
    if not section:
        return ""
    last = re.split(r"[>＞/]", section)[-1]
    last = re.sub(r"^[\d.\s、]+", "", last).strip()
    return last if len(last) >= 4 else ""


def _finalize(items: list[dict]) -> list[dict]:
    """去重 + 编号；同一问题只留第一条。"""
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        key = re.sub(r"[\s，。？?、]+", "", item["query"])
        if key in seen:
            continue
        seen.add(key)
        item["id"] = f"q{len(result) + 1:03d}"
        result.append(item)
    return result


def _print_summary(items: list[dict], out: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item["type"]] += 1
    total = len(items)
    print("\n" + "=" * 60)
    print(f"共 {total} 条，写入 {out.as_posix()}")
    for kind, count in counts.items():
        share = count / total * 100 if total else 0
        print(f"  {kind:<13} {count:>3} 条（{share:.0f}%）")
    if total and counts.get("unanswerable", 0) / total < 0.15:
        print("  [警告] 无答案型不足 15%，拒答率的分母会偏小")
    print("=" * 60)
    print("下一步：python -m app.eval.run_eval --queries " + out.as_posix())


if __name__ == "__main__":
    raise SystemExit(main())
