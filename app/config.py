"""应用配置：唯一负责从环境变量 / .env 读取配置，禁止在其它模块散落 os.getenv。

本项目的核心取舍：**本机不装任何深度学习依赖**。向量、重排、对话全部走 HTTP，
存储用 SQLite 单文件。因此这里没有任何 torch / CUDA / 模型路径相关的配置项。
"""
from __future__ import annotations

from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中声明全部配置项；容器内优先读取 docker-compose 注入的环境变量。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, data: Any) -> Any:
        """把空字符串当成「没配置」，回落到默认值。

        两个真实场景逼出来的：

        1. **GitHub Actions 里未设置的 secret 会变成空字符串**（不是不存在的变量）。
           于是 `EMBED_DIM=` 会让 pydantic 报 `int_parsing`，整个 Settings 构造失败，
           pytest 直接收集不了用例（退出码 4），CI 红得莫名其妙。
        2. 用户在 `.env` 里把某个数值项留空（`EMBED_DIM=`）也是同样的崩法。

        空字符串对字符串字段本来就等于「用默认值」，对数值字段则必须丢弃——
        否则 `""` 会被当成待解析的值。
        """
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value not in ("", None)}
        return data

    # ---------------- 存储 ----------------
    # 默认 SQLite 单文件：不需要 Postgres，clone 下来就能跑。换 pgvector 时只改这一行
    database_url: str = "sqlite:///./data/kb.db"
    # 超过这个块数就打印警告，提示该换 pgvector 了（numpy 全量点积在这里开始变慢）
    vector_warn_chunks: int = 50000

    # ---------------- 假模型（无 key / CI 用） ----------------
    # MOCK_MODE=1：向量、重排、对话全部换成本地桩实现，不联网、不需要任何 key。
    # 管道与引用校验逻辑完全一致，只有「模型输出质量」不同。
    mock_mode: bool = False

    # ---------------- 远端向量模型（OpenAI 兼容 /embeddings） ----------------
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    # 返回维度；与接口实际返回不一致时直接报错，避免把不同模型的向量混进一个矩阵
    embed_dim: int = 512
    embed_batch_size: int = 32
    embed_timeout_seconds: float = 30.0
    embed_max_retries: int = 2
    # bge 系列本地模型建议加检索指令前缀；bge-m3 等接口模型通常留空
    embed_query_instruction: str = ""
    # 文本 → 向量的本地缓存（表 embedding_cache），同一文本只发一次网络请求
    embed_cache_enabled: bool = True

    # ---------------- 远端重排（/rerank） ----------------
    # 默认关闭：融合结果已经够用，开会多一次网络往返
    rerank_enabled: bool = False
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    # 只重排融合后的前 N 个候选
    rerank_candidates: int = 10
    rerank_timeout_seconds: float = 30.0

    # ---------------- 检索层 ----------------
    # jieba 自定义词典：设备型号、专业术语
    bm25_dict_path: str = "data/dict.txt"
    # 融合前每一路的候选数量
    retrieval_candidates: int = 50
    # RRF 平滑常数，经验值 60
    rrf_k: int = 60
    # 表格行级索引：把每行当检索单元，命中后给**它所属的整块**加分（不新增候选）。
    # 解决「整块是几十行平均方向、问某一行时排不上来」（q014 就是这种）。
    # 默认关闭：开到 0.5 权重的实测结果见 docs/DESIGN.md，收益不足以抵消复杂度。
    table_row_index: bool = False
    # 行级命中给整块的加分量（相对 RRF 分数的比例）。0.5 表示「最多相当于半路命中」。
    table_row_weight: float = 0.5

    # ---------------- 生成层（P3） ----------------
    # OpenAI 兼容接口；三项任一为空即视为未配置，接口返回结构化错误而不是抛异常
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # 单次调用的超时（秒）：连接、读取、整体截止时间共用这一个值
    llm_timeout_seconds: float = 30.0
    # 首发失败后的重试次数与指数退避基数（1s、2s）
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0
    # 生成参数：工业问答不需要发散，温度压低；长度上限防止模型复述整段手册。
    # 4000 而不是 1200：思考型模型会把推理写进输出长度里，1200 会被吃掉导致正文为空
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    # 厂商特有字段，JSON 字符串，原样并入请求体。
    # 例：DeepSeek 关思考模式 → {"thinking": {"type": "disabled"}}
    llm_extra_body: str = ""
    # 请求流式用量统计（stream_options.include_usage）；服务端不认则自动退回估算
    llm_include_usage: bool = True

    # ---------------- Web（P5） ----------------
    # 文档目录：上传落这里，入库也扫这里
    raw_dir: str = "data/raw"
    # 评测报告：优先读这个文件，缺失时回退到 data/eval 下最新的 .md
    eval_report_path: str = "results/report.md"
    eval_report_dir: str = "data/eval"
    # 上传大小上限（MB）：nginx 的 client_max_body_size 要同步调大，否则 413 先于这里生效
    max_upload_mb: int = 50

    # 问答默认检索参数：P2 实测 hybrid 页码级 Recall@5 已到 1.000，重排默认不开
    chat_mode: str = "hybrid"
    chat_top_k: int = 5
    # 单块正文送进提示词的字符上限。库内最长块 2922 字（故障码表），3000 可覆盖全部；
    # 早期设 1200 时 203 块里有 58 块被截断，表格只剩表头，模型只能拒答
    context_max_chars: int = 3000
    # 表格超过这个字数就按行拆块（每块复制表头）；**0 表示不拆**（默认）。
    # 默认关掉不是懒得做，是实测证明拆完更差：块数 203→274、块级 Recall@5 0.979→0.896。
    # 原因见 docs/DESIGN.md「表格拆块：试过，数据说不要」。想自己验证就设成 1200 重跑。
    table_split_chars: int = 0


settings = Settings()
