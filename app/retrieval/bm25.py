"""BM25 词法检索：jieba 分词 + rank_bm25，索引常驻内存。

设计要点：
1. **索引在首次检索时从数据库构建**，之后常驻进程内存；`rebuild()` 供入库后强制重建。
   每次检索只做一次 `count(*)` 比对，块数变了就自动重建，避免入库后索引变旧。
2. **自定义词典**：`data/dict.txt` 收入设备型号（CP600、4T22G）与专业术语
   （母线电压、载波频率、休眠唤醒压力…）。不加词典时 jieba 会把「母线电压」切成
   「母线 / 电压」，把「CP600-4T22G」切碎，型号与术语的召回直接失效。
3. 分词统一小写、丢弃纯标点，保证 query 与文档用同一套切法。
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk

# 只保留中文、字母、数字；标点与空白一律丢弃，避免噪声词进倒排
_TOKEN_RE = re.compile(r"[0-9a-z\u4e00-\u9fff]+")

# 带连字符的标识符（功能码 F4-01、参数组 A8-58）。
# jieba 会把它们切成 `f4` + `01` 两截，两截都是高频词，等于把这个标识符丢掉了。
# 诊断脚本 `tools/diag_chunks.py` + `tokenize('...F4-01设成几')` 实测：
#   ['f4', '01'] —— 查询里最强的那条线索被稀释成两个噪声词，金标块直接掉出前 50。
_ID_WITH_DASH = re.compile(r"[a-z]{1,3}\d{1,3}(?:-\d{1,3})+")

_dict_loaded = False
_dict_lock = threading.Lock()


def load_user_dict(path: str | Path | None = None) -> int:
    """加载自定义词典，返回加入的词条数；词典缺失只告警不阻断。"""
    global _dict_loaded
    with _dict_lock:
        if _dict_loaded:
            return 0
        dict_path = Path(path or settings.bm25_dict_path)
        added = 0
        if dict_path.exists():
            with dict_path.open(encoding="utf-8") as handle:
                for line in handle:
                    word = line.split()[0] if line.strip() else ""
                    if not word or line.lstrip().startswith("#"):
                        continue
                    # 词频留空，词性标 nz（专名），让 jieba 优先把整词当作一个 token
                    jieba.add_word(word, freq=None, tag="nz")
                    added += 1
        _dict_loaded = True
        return added


def tokenize(text: str) -> list[str]:
    """分词：加载词典后按 jieba 精确模式切分，只保留中文/字母/数字。

    额外补一步：把被连字符切碎的标识符（`F4-01` → `f4` + `01`）**再拼成一个整 token 追加**。
    用「追加」而不是「替换」：拆开的半截在别的语境里仍有意义（`f4` 是参数组），
    同时保留整 token 才能让 BM25 靠 IDF 把它顶上来。查询与文档走同一个函数，
    所以两侧的整 token 一定能对上。
    """
    if not text:
        return []
    load_user_dict()
    lowered = text.lower()
    tokens: list[str] = []
    for piece in jieba.lcut(lowered, cut_all=False):
        for token in _TOKEN_RE.findall(piece):
            if token:
                tokens.append(token)
    for whole in _ID_WITH_DASH.findall(lowered):
        if whole not in tokens:
            tokens.append(whole)
    return tokens


@dataclass
class Bm25Index:
    """内存中的 BM25 索引快照。"""

    chunk_ids: list[int]
    bm25: BM25Okapi
    chunk_count: int
    built_at: float
    token_total: int

    @property
    def built_at_text(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.built_at))


_index: Bm25Index | None = None
_index_lock = threading.Lock()


def rebuild(session: Session) -> Bm25Index:
    """从数据库全量重建索引，返回新索引；入库完成后应显式调用一次。"""
    global _index
    rows = session.execute(
        select(Chunk.id, Chunk.section, Chunk.content).order_by(Chunk.id)
    ).all()
    chunk_ids = [int(row[0]) for row in rows]
    corpus = [tokenize(_index_text(row[1], row[2])) for row in rows]

    # rank_bm25 不接受空语料；空库时给一个占位文档，检索自然返回空
    token_total = sum(len(tokens) for tokens in corpus)
    if not corpus:
        corpus = [["<empty>"]]
    index = Bm25Index(
        chunk_ids=chunk_ids,
        bm25=BM25Okapi(corpus),
        chunk_count=len(chunk_ids),
        built_at=time.time(),
        token_total=token_total,
    )
    with _index_lock:
        _index = index
    return index


def get_index(session: Session) -> Bm25Index:
    """取当前索引；未构建或库内块数已变化时自动重建。"""
    global _index
    current = _index
    if current is not None:
        total = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
        if int(total) == current.chunk_count:
            return current
    with _index_lock:
        if _index is not None:
            total = session.execute(select(func.count()).select_from(Chunk)).scalar_one()
            if int(total) == _index.chunk_count:
                return _index
    return rebuild(session)


def invalidate() -> None:
    """丢弃内存索引，下一次检索时重建。"""
    global _index
    with _index_lock:
        _index = None


def _index_text(section: str | None, content: str) -> str:
    """参与分词的文本：章节路径 + 正文，标题里的型号与术语同样能被召回。"""
    if not section:
        return content
    if content.startswith(section):
        return content
    return f"{section} {content}"


def search(session: Session, query: str, top_k: int) -> list[tuple[int, float]]:
    """词法召回，返回 [(chunk_id, BM25 分数)]，按分数降序，只保留正分。"""
    index = get_index(session)
    if not index.chunk_ids:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = np.asarray(index.bm25.get_scores(tokens), dtype=np.float64)
    order = np.argsort(-scores)[:top_k]
    return [
        (index.chunk_ids[i], float(scores[i]))
        for i in order
        if scores[i] > 0.0
    ]
