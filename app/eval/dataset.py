"""评测数据集加载：读取 data/eval 下的 jsonl 评测集。

格式（每行一个 JSON 对象，UTF-8）::

    {"id": "q01", "query": "上电无显示怎么排查",
     "gold": [{"doc": "CP650空压机一体机手册.pdf", "page": 35}],
     "note": "4.2 常见故障及处理方法"}

字段校验从严：缺 `query` 或 `gold` 直接抛错，不静默跳过——评测集缺行会让分数虚高。
`page` 可以省略，表示只要命中该文档即可。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class DatasetError(ValueError):
    """评测集格式不合法。"""


@dataclass(frozen=True)
class GoldTarget:
    """一个金标命中点：文档名 + 可选页码。"""

    doc: str
    page: int | None = None


@dataclass
class EvalQuery:
    """一条评测样本。"""

    query_id: str
    query: str
    gold: list[GoldTarget]
    note: str = ""
    # answer：语料里有答案，应当作答并带引用；refusal：语料里没有，应当明确拒答
    expect: str = "answer"
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def gold_docs(self) -> set[str]:
        return {target.doc for target in self.gold}

    @property
    def gold_pairs(self) -> set[tuple[str, int]]:
        return {(target.doc, target.page) for target in self.gold if target.page is not None}


def load_queries(path: str | Path) -> list[EvalQuery]:
    """逐行读取评测集；空行跳过，其余任何不合法都直接报错。"""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"评测集不存在：{file_path}")

    queries: list[EvalQuery] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{file_path.name} 第 {line_no} 行不是合法 JSON：{exc}") from exc

            query_text = str(payload.get("query", "")).strip()
            if not query_text:
                raise DatasetError(f"{file_path.name} 第 {line_no} 行缺少 query 字段")

            # `expect=refusal` 表示这道题在语料里**确实没有答案**，期望模型拒答。
            # 这类题允许 gold 为空；其它题缺 gold 仍然直接报错——否则少标一行，
            # 该题就悄悄变成「怎么答都算对」，分数会虚高。
            expect = str(payload.get("expect", "answer")).strip().lower()
            if expect not in {"answer", "refusal"}:
                raise DatasetError(f"{file_path.name} 第 {line_no} 行的 expect 只能是 answer 或 refusal")

            raw_gold = payload.get("gold") or []
            if not raw_gold and expect != "refusal":
                raise DatasetError(f"{file_path.name} 第 {line_no} 行缺少 gold 字段")

            gold: list[GoldTarget] = []
            for item in raw_gold:
                doc = str(item.get("doc", "")).strip()
                if not doc:
                    raise DatasetError(f"{file_path.name} 第 {line_no} 行的 gold 项缺少 doc")
                page = item.get("page")
                gold.append(
                    GoldTarget(doc=doc, page=int(page) if page is not None else None)
                )

            queries.append(
                EvalQuery(
                    query_id=str(payload.get("id") or f"q{len(queries) + 1:02d}"),
                    query=query_text,
                    gold=gold,
                    note=str(payload.get("note", "")),
                    expect=expect,
                )
            )

    if not queries:
        raise DatasetError(f"评测集为空：{file_path}")
    return queries
