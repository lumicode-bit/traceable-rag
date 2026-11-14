"""提示词模板：把「只能依据片段、必须带来源、不足就拒答」写成硬约束。

三条设计决定，都是为了把幻觉压到可校验的范围：

1. **不写人设**。「你是一个资深工业运维专家」这类话只会让模型更敢编；
   这里只写禁止项与输出格式，每条都能被 `tests/check_citations.py` 检查。
2. **来源由检索给定，不由模型选择**。每个片段自带 `<cite>` 标签，标签内容就是唯一
   合法的引用写法，模型只需原样复制。这样「引用哪些片段」由检索决定，
   「引用是否真实」可以机械核对。
3. **系统提示词是静态文本**，片段与问题放在 user 消息里。静态前缀有利于前缀缓存，
   也避免把语料拼进系统角色。

片段正文按 `CONTEXT_MAX_CHARS` 截断（超长表格块在 P2 已暴露出 1700+ 字的极端情况）；
截断只影响送给模型的上下文，返回给前端的引用摘要仍是完整块的前若干字。
"""
from __future__ import annotations

from app.config import settings
from app.llm.citations import REFUSAL_MARKER, Citation

Message = dict[str, str]

# 系统提示词：静态、无占位符；改这里等于改行为，必须同步 tests/test_generation.py
SYSTEM_PROMPT = f"""你的全部依据只有用户消息里给出的 <片段>。片段以外的知识一律不许使用。

必须遵守的规则：
1. 只依据 <片段> 的内容作答。不得补充片段里没有的型号、参数、数值、步骤或经验推理。
2. 每条结论、每个排查步骤的末尾都必须紧跟来源。来源就是该片段 <cite> 标签里的原文，
   格式为 [文档名 p.页码]，例如 [CP600空压机一体机手册.pdf p.24]。
   原样复制标签内容：不许改文档名，不许改页码，不许写出标签里没有的来源。
   **禁止把 A 文档的页码安到 B 文档上，也禁止凭印象写出记忆里的页码**——
   来源只能从用户消息里的 <合法来源清单> 中原样复制，清单里没有的一条都不许写。
3. 如果 <片段> 不足以回答问题，只回复这一句：「{REFUSAL_MARKER}」，
   再另起一行写「缺少的信息：……」说明缺的是哪一类信息
   （例如「该机型的液压系统额定压力」）。这种情况不要给结论、不要给排查步骤、不要引用任何片段。
4. 如果 <片段> 之间互相矛盾（同一事项给出不同结论或不同数值），把矛盾指出来：
   分别写明两种说法及其来源，不要自己挑一个，也不要替它们调和。
5. 如果 <片段> 只覆盖问题的一部分，只回答被覆盖的部分；未覆盖的部分按第 3 条说明缺什么。
6. 用中文回答，结构固定为下面三节，顺序不变，标题原样保留：
   结论
   排查步骤
   来源
   排查步骤分条编号；「来源」一节只列本次回答实际引用过的片段，一行一个。
7. 不写寒暄、不重复问题，不写「作为专家」「建议联系厂家」这类没有片段支撑的话。

<片段> 由用户消息给出，每段带一个 <cite> 标签，标签内容就是唯一的合法来源写法。"""

def build_passages(citations: list[Citation], clip: int | None = None) -> str:
    """把片段渲染成带 <cite> 标签的文本；没有片段时明确写「无」。

    送模型的是 `Citation.content`（完整块正文），不是给前端看的那 120 字摘要。
    早期版本错把摘要当正文，故障码表只剩表头，模型只能拒答。
    """
    if not citations:
        return "（本次检索没有返回任何片段）"
    limit = clip if clip is not None else settings.context_max_chars
    blocks: list[str] = []
    for index, citation in enumerate(citations, start=1):
        body = _clip(citation.content, limit)
        header = f"[片段 {index}] <cite>{citation.label}</cite>"
        meta = f"文档：{citation.doc}　页码：{citation.page if citation.page else '无'}"
        if citation.section:
            meta += f"　章节：{citation.section}"
        blocks.append(f"{header}\n{meta}\n正文：\n{body}")
    return "\n\n".join(blocks)


def build_user_message(question: str, citations: list[Citation], clip: int | None = None) -> str:
    """用户消息：先片段、后清单、最后问题（问题放最后，避免被长上下文淹没）。

    单独给一份 `<合法来源清单>` 不是冗余：真实评测里模型会把 CP320 的 p.30 安到
    CP650 上（两份手册表格几乎一样），而把所有合法标签集中列一遍能显著减少这种拼装错误。
    """
    listing = "\n".join(citation.label for citation in citations) or "（无）"
    return (
        "<片段>\n"
        f"{build_passages(citations, clip)}\n"
        "</片段>\n\n"
        "<合法来源清单>\n"
        f"{listing}\n"
        "</合法来源清单>\n\n"
        f"【问题】{question}\n\n"
        "请按规则作答。来源只能从上面的 <合法来源清单> 里原样复制，一行一个。"
    )


def build_messages(question: str, citations: list[Citation], clip: int | None = None) -> list[Message]:
    """组装单轮对话的消息列表；本项目不做多轮记忆，永远只有 system + 一条 user。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, citations, clip)},
    ]


def _clip(text: str, limit: int) -> str:
    """按字符截断，保留开头（首行是小节标题，含型号与部件名，信息密度最高）。"""
    body = text.strip()
    if limit <= 0 or len(body) <= limit:
        return body
    return body[:limit].rstrip() + "…（本段已截断）"
