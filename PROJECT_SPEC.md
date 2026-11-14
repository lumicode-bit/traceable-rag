# 工业设备运维知识库 —— 项目规格说明书

> 版本：v1.3.0（拒答判定改为「结论段 + 无引用」，作答率假波动归零：47/48 三轮一致）
> 本文描述已落地的代码与规则，不包含、也不编造任何示例文档或示例数据。

## 1. 项目目标

把设备手册、维保规范、故障案例等文档入库，工人输入故障现象后，系统检索出最相关的原文段落，并由大模型生成**带引用**的回答，使结论可回溯到具体文档、页码与章节。

```
原始文档 → 解析 / OCR → 切块 → 入库（P1 已完成）
故障现象 → BM25 + 向量混合召回 → 重排 → 大模型生成（带引用）→ 校验引用（P3）
```

P1 的目标：把 `data/raw/` 下的真实文档解析成可检索的块，写入 `documents` / `chunks`，并且重复执行不产生重复块。

P2 的目标：在已入库的 203 个块上实现 vector / hybrid / hybrid_rerank 三种检索模式，用统一的 10 条评测集量化对比 Recall@5 与耗时，并把融合过程打印出来可核查。

P3 的目标：接 OpenAI 兼容大模型，用 SSE 流式输出**带来源**的回答；片段不足以作答时明确拒答；每次调用的用量与耗时落库；用脚本逐条核对回答里的来源是否真实存在于本次检索的片段中，把引用准确率作为硬指标。

P6 的目标：把项目变成能拿去参赛和投简历的材料——README 让人 10 分钟跑起来，
DESIGN 把四个设计问题答清楚（含代价与失败模式），DEMO 给出按秒的演示脚本，
简历 bullet 只写实测数字。**不编造用户量、准确率、合作企业**。

P5 的目标：给评委一个能当场操作的界面（问答、文档管理、评测报告三页），并部署到公网。
做法约束：Jinja2 + 原生 JS，无框架无构建；白底深色、无阴影渐变；手机可用；
nginx 反代 + HTTPS；`DEPLOY.md` 从空 VPS 一路写到能访问。

**环境约束（v0.5.0 起生效，优先于上面所有阶段）**：本机不安装任何深度学习框架。
不引入 torch / sentence-transformers / paddlepaddle / transformers，连 `try: import` 都不写。
向量、重排、对话全部改成 HTTP 调用；存储改为 SQLite 单文件；OCR 改为 onnxruntime 后端。
理由是 5~8GB 的依赖与 CUDA/DLL/中文路径问题会吃掉整个项目一周，而它们对答辩毫无帮助。
代价是：**本机跑出来的是「管道正确」，不是「模型效果优秀」**——后者必须在配好 `.env` 的环境里实测。

## 2. 技术栈（固定，不更换）

| 层 | 选型 |
| --- | --- |
| 语言 / Web | Python 3.11（Docker）/ 3.12（本机）、FastAPI |
| ORM | SQLAlchemy 2.0 |
| 数据库 | SQLite 单文件 `data/kb.db`（WAL 模式）；第六周部署时可换 Postgres |
| 文档解析 | PyMuPDF；DOCX 补充 python-docx |
| 扫描件 OCR | RapidOCR（`rapidocr-onnxruntime`，纯 CPU，约 80MB） |
| 分词 / 词法检索 | jieba；rank_bm25（P2） |
| 向量 | **远端** OpenAI 兼容 `/embeddings`（httpx），本机不加载模型；本地缓存表 `embedding_cache` |
| 向量检索 | numpy 全量点积（float32 BLOB，2 万块 × 512 维 ≈ 40MB，单次 10~30ms） |
| 重排 | **远端** `/rerank`（httpx），未配置或失败时降级为原序并标记 `degraded` |
| 大模型 | 任意 OpenAI 兼容接口，`base_url` + `api_key` + `model` 从 `.env` 读取（P3） |
| HTTP 客户端 | `httpx` 直连各接口，不引入 openai / sentence-transformers SDK |
| 假模型 | `app/llm/mock.py`：`MOCK_MODE=1` 时替换向量、重排、对话，供无 key 的机器跑通全链路 |
| 前端 | Jinja2 模板 + 原生 JS（`fetch` + `ReadableStream`），无框架、无构建工具 |
| 部署 | Docker Compose：`app` + `nginx`（反代，暴露 80/443）+ `certbot`（按需，profile 启动） |

> 运行时依赖 13 个（加 pytest 共 14 行）、装完约 420MB：fastapi uvicorn jinja2 sqlalchemy
> pydantic-settings httpx python-multipart PyMuPDF python-docx rapidocr-onnxruntime
> jieba rank-bm25 numpy（pytest 仅测试用）。体积大头是 OCR 的 opencv 108MB + onnxruntime 52MB
> 与 PyMuPDF 47MB。没有 torch、没有 paddle、没有 Postgres。
>
> 一处已知限制：`rapidocr-onnxruntime` 只支持 Python < 3.13。本机是 3.12，Docker 是 3.11，
> 都能装；如果谁用 3.13 直接 `pip install -r requirements.txt`，这一项会被跳过，
> 扫描件识别会报「未安装 RapidOCR」，其余功能不受影响（`ocr.py` 是惰性导入）。

## 3. 数据模型：六张表字段说明

ORM 定义位于 `app/models.py`，统一继承 `app/db.py` 的 `Base`。

### 3.1 `documents` —— 原始文档登记

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | Integer PK | 否 | 文档主键 |
| `title` | String(255) | 否 | 文档标题（取文件名主干） |
| `filename` | String(255) | 否 | 原始文件名 |
| `source_path` | String(512) | 否 | 相对 `data/raw` 的存放路径 |
| `sha256` | String(64) | 否 | 文件内容摘要，**唯一索引**，用于幂等去重 |
| `doc_type` | String(32) | 否 | 文档类型（pdf / png / jpg / docx） |
| `page_count` | Integer | 是 | 页数（图片型文档记为 1 页） |
| `ocr_pages` | Integer | 是 | 本次解析中走 OCR 的页数 |
| `status` | String(32) | 否 | 入库状态，默认 `pending`，取值见 3.8 |
| `failed_reason` | Text | 是 | 失败原因（异常类型 + 消息），成功时为空 |
| `created_at` | DateTime(tz) | 否 | 创建时间，服务端 `now()` |

### 3.2 `chunks` —— 检索单元

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | Integer PK | 否 | 分块主键 |
| `document_id` | Integer FK → `documents.id` | 否 | 所属文档，`ON DELETE CASCADE`，带索引 |
| `page_no` | Integer | 是 | 块起始页码，用于回答引用 |
| `section` | String(255) | 是 | 章节路径，如 `3.2 润滑系统 > 3.2.1 油压异常` |
| `content` | Text | 否 | 块正文，首行是小节标题，其后为段落与 markdown 表格 |
| `embedding` | LargeBinary | 是 | **归一化后的 float32 原始字节**（`np.float32.tobytes()`），由检索层写入。不用 pgvector：2 万块 × 512 维只有 40MB，numpy 全量点积 10~30ms，装 Postgres 在这个量级上的收益是零 |
| `embed_model` | String(128) | 是 | 生成该向量的模型名。换模型后旧向量不会被混进新矩阵（维度相同也不混），而是被跳过并提示重建 |
| `created_at` | DateTime(tz) | 否 | 创建时间，服务端 `now()` |

### 3.3 `embedding_cache` —— 文本向量缓存

存在的唯一理由：入库重跑、评测重跑、同一句 query 反复检索时不要重复付费。

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `key` | String(64) PK | 否 | `sha256(model + "\x00" + text)`，换模型后同一文本会重新算，缓存不串味 |
| `model` | String(128) | 否 | 生成该向量的模型名 |
| `dim` | Integer | 否 | 维度 |
| `vector` | LargeBinary | 否 | 归一化 float32 字节 |
| `text_chars` | Integer | 否 | 原始文本长度，便于排查异常长文本 |
| `created_at` | DateTime(tz) | 否 | 写入时间 |

命中缓存就**不发网络请求**——这一点由 `tests/test_mock_pipeline.py::test_second_identical_text_hits_local_cache` 用计数桩验证（不联网）。

### 3.4 `eval_runs` —— 一次评测运行

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | Integer PK | 否 | 评测运行主键 |
| `dataset_name` | String(255) | 否 | 所用评测集名称（`data/eval` 下用户提供的真实数据） |
| `config` | JSON | 是 | 本次检索 / 生成参数快照 |
| `status` | String(32) | 否 | 运行状态（running / finished / failed），默认 `running` |
| `created_at` | DateTime(tz) | 否 | 开始时间 |
| `finished_at` | DateTime(tz) | 是 | 结束时间 |

### 3.5 `eval_results` —— 逐题评测明细

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | Integer PK | 否 | 明细主键 |
| `run_id` | Integer FK → `eval_runs.id` | 否 | 所属运行，`ON DELETE CASCADE`，带索引 |
| `question` | Text | 否 | 故障现象 / 评测问题 |
| `expected_chunk_ids` | JSON | 是 | 标注的期望命中分块 ID 列表 |
| `retrieved_chunk_ids` | JSON | 是 | 实际召回的分块 ID 列表 |
| `answer` | Text | 是 | 生成的回答 |
| `metrics` | JSON | 是 | 该题的指标结果 |
| `created_at` | DateTime(tz) | 否 | 创建时间 |

### 3.6 `qa_logs` —— 问答调用日志

每次 `/api/chat` 调用写一行，用于核算 token 成本、定位慢请求、复盘幻觉。

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `id` | Integer PK | 否 | 日志主键 |
| `question` | Text | 否 | 用户问题 |
| `mode` | String(32) | 否 | 本次使用的检索模式 |
| `top_k` | Integer | 否 | 送入提示词的片段数 |
| `answer` | Text | 是 | 完整回答（失败时为已生成的部分） |
| `answer_status` | String(32) | 否 | `answered` / `refused` / `error` / `partial`（`partial` 指客户端中途断开） |
| `reference_count` | Integer | 否 | 回答中引用出现的次数（不是去重后的个数） |
| `invalid_citations` | JSON | 是 | 校验出的幻觉引用 `[{label, reason}]`，正常为 `[]` |
| `citations` | JSON | 是 | 本次检索到的片段快照 `[{chunk_id, doc, page, section, snippet, label}]` |
| `retrieved_chunk_ids` | JSON | 是 | 实际召回的块 ID 列表 |
| `prompt_tokens` | Integer | 是 | 输入 token（服务端返回或估算） |
| `completion_tokens` | Integer | 是 | 输出 token（服务端返回或估算） |
| `total_tokens` | Integer | 是 | 合计 token |
| `usage_estimated` | Boolean | 否 | 是否本地估算。估算值不得当实测值报给用户 |
| `retrieval_ms` | Float | 是 | 检索耗时（含向量编码、融合、重排） |
| `generation_ms` | Float | 是 | 生成耗时（首个 token 前的等待 + 全部增量） |
| `llm_model` | String(128) | 是 | 本次使用的模型名；MOCK 模式写 `stub-llm(mock)`，不冒充真实模型 |
| `llm_error` | Text | 是 | 结构化错误单行文本 `code(status, attempts): message` |
| `created_at` | DateTime(tz) | 否 | 创建时间，服务端 `now()` |

写日志是旁路：写入失败只告警，不影响已经流给用户的回答。

### 3.7 关系与约束

- `documents 1 ── N chunks`；`eval_runs 1 ── N eval_results`；两条外键均为级联删除。
- `documents.sha256` 唯一索引保证同一文件不会重复入库。
- `chunks.document_id`、`eval_results.run_id` 建索引。
- `qa_logs`、`embedding_cache` 不设外键：前者是独立证据，后者是跨文档共享的缓存。
- 表结构由启动时的 `create_all` 幂等创建，不使用迁移工具。
  `init_db()` 会检查 `chunks` 是否存在 `embed_model` 列，发现旧结构时打印
  「请删除 `data/kb.db` 后重新入库」的告警——`create_all` 不会改已有表。

### 3.8 `documents.status` 状态机

```
pending ──► parsing ──► indexed
                └────► failed（写入 failed_reason）
```

| 状态 | 含义 |
| --- | --- |
| `pending` | 已登记，尚未开始解析 |
| `parsing` | 正在解析 / 切块 / 写块，事务未提交 |
| `indexed` | 已入库，块已提交；重复扫描时直接跳过 |
| `failed` | 解析或入库失败，`failed_reason` 记录原因；下次运行会重建该记录并重试 |

块与状态在同一事务内提交：只有 `indexed` 才可能查到块；失败会回滚，不留半成品。

## 4. 对外接口清单

| 方法 | 路径 / 命令 | 职责 | 归属模块 | 状态 |
| --- | --- | --- | --- | --- |
| GET | `/health` | 返回数据库连通状态 | `app/main.py` | 已实现（P0） |
| CLI | `python -m app.ingest.pipeline data/raw` | 解析 → 切块 → 入库，输出统计 | `app/ingest/pipeline.py` | 已实现（P1） |
| CLI | `python -m app.retrieval.bench --queries data/eval/smoke_queries.jsonl` | 三种模式的 Recall@5 与耗时对比，可打印融合过程 | `app/retrieval/bench.py` | **已实现（P2）** |
| 内部 | `search(query, top_k, mode) -> list[Chunk]` | 检索层唯一入口，mode ∈ vector / hybrid / hybrid_rerank | `app/retrieval/__init__.py` | **已实现（P2）** |
| 内部 | `search_with_ranks(...) -> list[SearchHit]` | 同上，但保留向量排名、BM25 排名、融合排名、重排分 | `app/retrieval/__init__.py` | **已实现（P2）** |
| 内部 | `rebuild_indexes(session, force_embedding=False)` | 补齐向量 + 重建 BM25，供入库后与 `/api/admin/reindex` 调用 | `app/retrieval/__init__.py` | **已实现（P2）** |
| CLI | `python tests/check_citations.py` | 引用准确率校验 + 拒答题检查（可跑任意题库） | `tests/check_citations.py` | **已实现（P3）** |
| CLI | `python -m app.eval.generate_questions` | 从已入库分块批量出题（agent 出候选，人删） | `app/eval/generate_questions.py` | **已实现（P6）** |
| CLI | `python -m app.eval.run_eval` | 块级金标的三配置对比 + 失败案例 → `results/report.md` | `app/eval/run_eval.py` | **已实现（P6）** |
| CLI | `python -m app.eval.stress_ingest` | 跨厂商手册压测：独立库入库，只记失败模式，不调参 | `app/eval/stress_ingest.py` | **已实现（P6 补）** |
| 内部 | `prepare(question, mode, top_k) -> AnswerPlan` | 检索 + 组织提示词，生成阶段的唯一入口 | `app/llm/pipeline.py` | **已实现（P3）** |
| 内部 | `stream_answer(plan) -> Iterator[dict]` | 流式生成 + 引用校验 + 写 `qa_logs` | `app/llm/pipeline.py` | **已实现（P3）** |
| 内部 | `encode_documents(texts)` / `encode_query(text)` | 远端 `/embeddings` + `embedding_cache` 缓存 | `app/llm/embed.py` | **已实现（P2，v0.5.0 改为 HTTP）** |
| 内部 | `rerank(query, candidates, top_k) -> RerankOutcome` | 远端 `/rerank`，失败降级为原序 | `app/llm/rerank.py` | **已实现（P2，v0.5.0 改为 HTTP）** |
| 内部 | `stub_embed` / `stub_rerank_scores` / `stream_stub` | 无 key 时的本地桩模型（`MOCK_MODE=1`） | `app/llm/mock.py` | **已实现（v0.5.0）** |
| GET | `/` | 问答页：输入框、流式回答、引用卡片、模式切换 | `app/web/pages.py` + `web/templates` | **已实现（P5）** |
| GET | `/documents` | 文档管理页：上传、列表、删除、重新解析 | 同上 | **已实现（P5）** |
| GET | `/eval` | 评测页：报告 Markdown → 表格 | 同上 | **已实现（P5）** |
| POST | `/api/documents` | 上传文档（落盘 + 后台解析） | `app/api/documents.py` | **已实现（P5）** |
| GET | `/api/documents` | 文档列表：状态、页数、块数、已向量化块数 | `app/api/documents.py` | **已实现（P5）** |
| DELETE | `/api/documents/{document_id}` | 删除文档、分块与磁盘文件 | `app/api/documents.py` | **已实现（P5）** |
| POST | `/api/documents/{document_id}/ingest` | 重新解析（状态改回 pending） | `app/api/documents.py` | **已实现（P5）** |
| GET | `/api/chunks/{chunk_id}` | 取单个块的完整正文，供引用卡片展开 | `app/api/chunks.py` | **已实现（P5）** |
| POST | `/api/chat` | 输入故障现象 → 检索 + SSE 流式带引用回答 | `app/api/chat.py` | **已实现（P3）** |
| POST | `/api/admin/reindex` | 重建 BM25 / 向量索引 | `app/api/admin.py` | 空壳（P5 用 CLI 代替） |
| GET | `/api/admin/stats` | 文档数、分块数、索引状态 | `app/api/admin.py` | 空壳（文档页已展示这些数字） |

CLI 参数：

```bash
python -m app.ingest.pipeline [目录] [--sample N]
```

- `目录` 默认 `data/raw`，递归扫描 `.pdf / .png / .jpg / .jpeg / .docx`。
- `--sample N` 结束后随机打印 N 个块，用于人工核对切块质量。
- 退出码：正常完成 0；目录不存在 2；数据库初始化失败 1。单个文件失败不改变退出码，失败会列入统计。

评测 CLI 参数：

```bash
python -m app.retrieval.bench [--queries 路径] [--top-k 5] [--modes vector,hybrid,hybrid_rerank] \
                              [--verbose] [--trace q02] [--rebuild] [--save]
```

- `--queries` 默认 `data/eval/smoke_queries.jsonl`，**只读**，评测过程不会修改评测集。
- `--verbose` 逐题打印命中情况（✓ 页码命中 / ◐ 仅文档命中 / ✗ 未命中）。
- `--trace` 打印指定题目的两路排名与融合排名（可用题号或 id）。
- `--rebuild` 评测前强制重算向量并重建 BM25；`--save` 把结果写入 `eval_runs` / `eval_results`。
- 退出码：正常 0；评测集不存在或格式不合法 2。

引用校验 CLI 参数：

```bash
python tests/check_citations.py [--queries 路径] [--mode hybrid] [--top-k 5] \
                                [--mock [--hallucinate]] [--skip-unanswerable] [--out 报告.json]
```

- 默认用 `data/eval/smoke_queries.jsonl` 的 10 题 + 2 个库里确实没有的问题。
- `--mock` 用本地桩模型替代真实对话模型，**只用于验证校验脚本本身**；真实准确率必须不带 `--mock` 跑。
- `--hallucinate` 配合 `--mock`，故意编一个不存在的来源，用来确认脚本能把它抓出来。
- `--out` 把逐题明细写成 JSON，便于留档。
- 整个脚本也支持 `MOCK_MODE=1`：此时向量与对话都是桩，用于在无 key 机器上验证管道。
- 退出码：0 全部通过；1 存在幻觉引用 / 该拒答而未拒答 / 有题目报错；2 库空、缺向量或模型未配置。

## 5. 目录职责

```
.
├── app/
│   ├── main.py            FastAPI 应用装配 + /health
│   ├── config.py          唯一的配置读取入口
│   ├── db.py              引擎、声明基类、会话工厂、建表、连通性探测
│   ├── models.py          六张表的 ORM 定义
│   ├── api/               HTTP 接口层：chat 已实现，documents / admin 仍为空壳
│   ├── ingest/
│   │   ├── loader.py      PDF / 图片 / DOCX → 页 + 段落/表格块（已实现）
│   │   ├── ocr.py         RapidOCR（onnxruntime）封装：文本 + 置信度 + 低置信度标记
│   │   ├── chunker.py     标题层级优先切块，400–800 字，100 字重叠（已实现）
│   │   └── pipeline.py    命令行入库：解析 → 切块 → 入库 + 统计（已实现）
│   ├── retrieval/         检索层（P2）
│   │   ├── vector.py      远端向量编码 + numpy 全量点积（float32 BLOB）
│   │   ├── bm25.py        jieba 分词 + BM25，内存索引
│   │   ├── hybrid.py      RRF 融合，保留两路原始排名
│   │   ├── __init__.py    统一入口 search / search_with_ranks
│   │   └── bench.py       三模式评测 CLI
│   ├── llm/               模型接入 + 生成（P2 向量/重排、P3 对话）
│   │   ├── embed.py       OpenAI 兼容 /embeddings + embedding_cache 缓存
│   │   ├── rerank.py      远端 /rerank，失败降级为原序并标 degraded
│   │   ├── client.py      /chat/completions 流式客户端：30s 超时、2 次退避重试、结构化错误
│   │   ├── prompt.py      约束型系统提示词 + 片段渲染 + 引用标签
│   │   ├── citations.py   引用的领域对象与真伪校验（纯函数）
│   │   ├── pipeline.py    编排：检索 → 提示词 → 流式生成 → 校验 → 写 qa_logs
│   │   └── mock.py        桩模型（StubEmbedder / StubRerank / StubLLM），MOCK_MODE=1 启用
│   ├── web/               页面层（P5）
│   │   ├── pages.py       三个页面的路由、报告发现与选择
│   │   └── markdown.py    极简 Markdown → HTML（只支持报告用到的语法）
│   └── eval/              评测层：dataset（评测集加载）、metrics（Recall/分位）
├── web/templates/         base / chat / documents / eval 四个模板
├── web/static/            style.css + common.js / chat.js / documents.js
├── deploy/
│   ├── nginx.conf         HTTP 反代（SSE 不缓冲）
│   ├── nginx.https.conf   HTTPS 版（含 301 跳转与 HSTS）
│   └── certbot/           webroot 与证书目录（不进仓库）
├── results/               评测报告（默认读 results/report.md，不进仓库）
├── data/raw/              真实文档（不提交仓库）
├── data/kb.db             SQLite 库文件（不提交仓库，删了重跑入库即重建）
├── data/dict.txt          jieba 自定义词典：设备型号 + 专业术语
├── data/eval/             真实评测集（smoke_queries.jsonl）与验收记录
├── tests/                 pytest 用例；check_citations.py 为引用准确率校验脚本
├── README.md              对外首页：一句话定位、架构图、评测数字、10 分钟跑通
├── docs/
│   ├── DESIGN.md          四个设计问题的实话（收益、代价、失败模式）
│   ├── DEMO.md            三分钟演示脚本（按秒）+ 答辩问答 + 故障预案
│   ├── RESUME.md          简历三条 bullet（数字用 {X} 占位）+ 取值出处
│   └── images/            截图（占位，由作者补图）
├── DEPLOY.md              从空 VPS 到 HTTPS 的完整部署手册
├── requirements.txt       依赖清单（14 行，无 torch / paddle）
├── Dockerfile / docker-compose.yml
└── PROJECT_SPEC.md        本文
```

| 文件 | 单一职责 |
| --- | --- |
| `app/ingest/loader.py` | 只把文件变成结构化的页与块，不判定章节、不切块 |
| `app/ingest/ocr.py` | 只做图片 → 文本 + 置信度，不关心来源是 PDF 还是图片文件 |
| `app/ingest/chunker.py` | 只做块切分与章节路径维护，不读文件、不写库 |
| `app/ingest/pipeline.py` | 只做编排、状态流转与统计输出 |
| `app/retrieval/vector.py` | 只做语义召回（调 `embed` 编码 + 内存点积），不关心融合与重排 |
| `app/retrieval/bm25.py` | 只做词法召回，索引的生命周期（构建 / 复用 / 失效）也在这里 |
| `app/retrieval/hybrid.py` | 只做排名融合，不打分、不查库 |
| `app/retrieval/__init__.py` | 只做编排与结果装配，不实现任何检索算法 |
| `app/llm/embed.py` | 只管「文本 → 归一化向量」，含批量、重试与缓存；缺配置就报错，不返回假向量 |
| `app/llm/rerank.py` | 只管「候选重排」，失败必须降级且可见，不抛到上层 |
| `app/llm/mock.py` | 只提供与真实模型同形状的桩，输出自我标记，不参与任何效果评估 |
| `app/llm/client.py` | 只负责发出请求、解析 SSE、把失败表达成结构化错误；不含提示词与检索 |
| `app/llm/prompt.py` | 只负责把片段与约束组织成消息；不调模型、不查库 |
| `app/llm/citations.py` | 只做引用的解析与真伪校验（纯函数），不接触数据库与模型 |
| `app/llm/pipeline.py` | 只做编排与记账（耗时、用量、日志），不实现 HTTP 细节或检索算法 |
| `app/api/chat.py` | 只做 HTTP 层：参数校验、SSE 分帧、把同步链路桥接进线程池 |
| `app/api/documents.py` | 只做上传/列表/删除/重解析的编排；解析本身交给 `ingest.pipeline` |
| `app/api/chunks.py` | 只按 id 取一个块的完整正文，不参与检索与生成 |
| `app/web/pages.py` | 只把数据放进模板；不含业务判断 |
| `app/web/markdown.py` | 只做 Markdown → HTML，先转义再拼标签 |
| `app/eval/dataset.py` | 只解析评测集文件，不做检索也不算指标 |
| `app/eval/metrics.py` | 只算指标，不接触数据库与模型 |
| `tests/check_citations.py` | 只跑题目并核对引用真伪，不改语料、不改金标 |

## 6. 编码约定

1. **注释与文档字符串一律中文**；标识符、文件名、表名使用英文 `snake_case`。
2. **全量类型标注**：所有函数/方法的参数与返回值都要标注；模块统一 `from __future__ import annotations`。
3. **每个模块一个职责**，跨职责逻辑放进对应模块，不写「大杂烩」工具模块。
4. **ORM 之外不加抽象层**：不引入 Repository、Service、UnitOfWork、DI 容器；直接用 SQLAlchemy 2.0 的 `DeclarativeBase` + `Mapped` / `mapped_column`。
5. **配置集中**：一律经 `app.config.settings` 读取，禁止在业务模块里散落 `os.getenv`。
6. **切块参数集中**：`MIN_CHARS / MAX_CHARS / OVERLAP_CHARS` 只在 `chunker.py` 顶部定义。
7. **向量维度由 `EMBED_DIM` 决定**，换模型必须同步改该配置并重建索引（`embed_model` 不一致的行会被跳过）。
8. **表结构即文档**：任何字段变更必须同步更新本文第 3 节。
9. **前端零框架**：仅 Jinja2 模板 + 原生 JS。
10. **不造数据**：不编造示例手册、故障案例或评测集；真实文档由用户放入 `data/raw`。
11. **异常不静默**：解析异常必须写入 `failed_reason` 并继续处理下一个文件；日志与统计都要能看到。
12. **禁止深度学习依赖**：代码里不许 `import torch / transformers / sentence_transformers / paddle*`，
    连 `try: import` 兜底也不行——那等于把 5GB 依赖重新塞回部署链路。
13. **远端调用必须能被替换与降级**：向量缺配置报错、重排失败降级、对话失败结构化返回；
    三者都要有 `MOCK_MODE` 的桩实现，且桩输出必须自我标记，不得混进任何效果数字。

## 7. 入库流水线（P1）

### 7.1 流程

```
扫描目录 → 计算 sha256 → 查重 → 登记(parsing) → 解析 → 切块 → 写 chunks → 提交(indexed)
                                                     └─ 任一环节异常 → 回滚 → failed + failed_reason
```

### 7.2 解析规则（`loader.py`）

- **PDF**：逐页 `get_text("text")`；字符数 **< 30 且页面含图片** 才判定为扫描页并送 OCR（纯空白页、无图分隔页不浪费 OCR）。
- **页眉页脚**：位于页面上下 8% 且跨页重复出现的行会被剔除，页码行按正则剔除。判定有三条补充规则，都来自真实手册踩坑：
  1. **字号豁免**：字号 ≥ 正文字号 + 3pt 的行一律不剔除。CP600 的章标题「2 产品信息」排在 y=37.6（正好落在 8% 页眉带内）且与页眉文字相同，只按「重复」判定会把真标题删掉，导致 2~5 章全部挂到第 1 章下面。
  2. **短文本 2 次即算重复**：手册奇偶页常交替印「章节名 / 手册名」，每个变体全书只出现 2 次，高阈值会让它逃过剔除、混进正文块。规则为 `≥3 次` 无条件判定，`==2 次` 且文本 ≤24 字也判定。
  3. **页码先于字号豁免**：页码常是 11pt（大于正文），必须先按正则判定再走字号豁免，否则 `- 3 -` 会被当成大字保留。
- **行内切分**：同一行里两个 span 的水平间距 > 36pt 视为两段独立内容（页眉左右栏、多栏排版），先切开再各自成行。否则「附录A参数表 + 手册名」会被拼成一个每章都不同的字符串，重复计数被打散，页眉剔不干净。
- **项目符号**：PDF 常把「◆」画两遍（一次独立成行、一次在正文前），加载时剥掉前导符号、整行都是符号则丢弃，避免 `◆ ◆` 混入正文与向量化文本。
- **段落重建**：按行距、字号、缩进把文本行合并成自然段落，避免 PDF 里「一行一个块」导致碎片化。
- **表格**：`find_tables()` 的结果先做结构校验（至少两列有内容、单元格中位长度 ≤ 40 字），通过后整表转 markdown；未通过的（例如「警告/危险」提示框）交回普通文本，避免整页文字被吞掉。
- **DOCX**：按正文顺序遍历段落与表格；标题层级取 Word 样式（`Heading 1` / `标题 1`）；页号无法获得，`page_no` 置空。
- **图片**：整图 OCR，视为 1 页。

### 7.3 OCR 规则（`ocr.py`）

- **RapidOCR（onnxruntime 后端）**中文模型，按需加载；未安装或加载失败直接抛 `OcrUnavailableError`，不返回空字符串冒充成功。
  换掉 PaddleOCR 的原因：paddlepaddle 体积大、与 onnxruntime 常冲突，而本项目只需要「扫描页 → 中文文本」。
- 输出文本 + 逐行置信度，按行距聚成段落；**段落平均置信度 < 0.6 时**在正文前加 `【OCR低置信度 0.52】` 标记。
- 接口保持 `recognize(path) -> list[OcrParagraph]`（含 `text` / `confidence` / `low_confidence`），换引擎不影响 loader。
- 已知限制：`rapidocr-onnxruntime` 只支持 Python < 3.13，3.13 环境下这一功能不可用（其余功能正常）。

### 7.4 切块规则（`chunker.py`）

| 规则 | 实现 |
| --- | --- |
| 标题识别 | 编号正则 `1 / 1.1 / 1.1.1`；无编号时用字号：正文字号取「20 字以上段落的众数字号」，明显更大（≥ +1.0pt）且为中文短句才判为标题 |
| 标题层级 | 编号点号个数即层级；纯字号标题按字号从大到小映射 1..N 级 |
| 章节路径 | 维护层级栈，`section` 记录完整路径，如 `2 安装与接线 > 2.2 接线 > 2.2.2 控制端口分布` |
| 块大小 | 目标 400–800 字；到 800 就切，不足 400 的块尽量与相邻块合并 |
| 切分位置 | 只在段落/句子边界切，超长段落按 `。！？；` 再拆，绝不按固定字数硬切 |
| 重叠 | 同一小节被拆开时，后一块带前一块末尾 100 字，并从句子边界开始 |
| 表格 | 表格是原子单位，永不拆行；能塞进当前块就带上文，放不下就独立成块（可能超过 800 字） |
| 标题不脱离正文 | 每个块的 `content` 首行都是所属小节标题，标题不会单独成块 |
| 噪声过滤 | 目录行（点线引导符 + 页码）不参与切块；全文反复出现的「危险/警告/注意」标签不算标题 |
| 引用信息 | 每块都带 `page_no`（块起始页）与 `section` 路径 |

### 7.5 幂等与失败处理

- `documents.sha256` 唯一索引 + 入库前查重：已 `indexed` 的文件直接跳过并打印提示，块数不增加。
- 状态为 `failed / parsing / pending` 的旧记录会被删除重建，便于重跑修复。
- 单文件异常：回滚该文件的所有块，写入 `failed_reason`，继续处理下一个文件；统计里单列失败清单。
- 不存在的目录、数据库初始化失败会直接报错退出，不产生「看起来成功」的结果。

## 8. 检索层（P2）

### 8.1 三种模式

| 模式 | 组成 | 说明 |
| --- | --- | --- |
| `vector` | `vector.py` | 远端 `/embeddings` 编码 query，numpy 全量点积取 top_k |
| `hybrid` | `vector.py` + `bm25.py` + `hybrid.py` | 两路各取 `RETRIEVAL_CANDIDATES`（默认 50）个候选，RRF（k=60）融合后取 top_k |
| `hybrid_rerank` | 上述 + `app/llm/rerank.py` | 融合结果的前 `RERANK_CANDIDATES`（默认 10）个候选送远端 `/rerank` 重排；关闭或调用失败时**原样透传并标 `degraded`** |

### 8.2 向量检索（`vector.py`）

- 编码交给 `app/llm/embed.py`：OpenAI 兼容 `/embeddings`，本机**不加载任何模型**。
- 编码文本 = `section` + 正文（正文已以标题开头时不重复拼接），让章节路径参与语义匹配。
- 返回的向量在 `embed.py` 里 L2 归一化，入库前又按 float32 落字节；检索侧点积即余弦，不必每次算模长。
- 索引常驻内存（`ids` + `matrix`），按「有向量的块数」失效；`index_chunks()` 写完主动清缓存，避免刚补完向量却检索到旧矩阵。
- `chunks.embed_model` 与当前模型不一致的行会被跳过并告警，绝不把不同模型的向量混进一个矩阵。
- `index_chunks(session, force=False)` 只补「缺向量 / 模型不一致」的块；`force=True` 全量重算，用于换模型。
- 超过 `VECTOR_WARN_CHUNKS`（默认 5 万块）打印一次告警：该换 pgvector / FAISS 了。

### 8.3 词法检索（`bm25.py`）

- jieba 精确模式分词，统一小写，只保留中文/字母/数字（`_TOKEN_RE`），标点与空白不进倒排。
- **自定义词典 `data/dict.txt`**：设备型号（CP600-4T22G）、专业术语（母线电压、载波频率、休眠唤醒）。不加词典时 jieba 会把「母线电压」切成「母线/电压」，型号与术语的召回直接失效。
- 索引在首次检索时从数据库构建并常驻内存；每次检索先 `count(*)` 比对块数，变了就自动重建；`rebuild()` / `invalidate()` 供入库后显式刷新。
- 只返回正分结果——BM25 分数为 0 表示没有任何词命中，塞进候选只会污染融合。

### 8.4 融合（`hybrid.py`）

- RRF：`score = Σ 1 / (k + rank)`，k=60。只看排名不看分数，避开「余弦 0~1 与 BM25 0~50 尺度不同、直接相加会被某一路主导」的问题。
- 每个候选保留 `vector_rank` / `bm25_rank` / `sources`（V / B / VB），用于判断这题是哪一路救回来的。
- 分数相同时按 `chunk_id` 升序，保证多次运行结果一致。

### 8.5 重排（`app/llm/rerank.py`）

- 开关：`RERANK_ENABLED`，默认 `false`。关闭时直接透传，`rerank_score=None`。
- 开启时把 `(query, 候选正文)` POST 到远端 `/rerank`，兼容三种返回格式：`results[].relevance_score`、`results[].score`、`scores[]`。
- **失败一律降级而不是报错**：重排只是给已召回的候选换序，原序本身可用，不该因为一次网络抖动让整条问答失败。但降级必须可见——日志记 `degraded` 与原因，结果里 `SearchHit.extra["rerank_degraded"]=1`。
- 退避策略比对话客户端弱（只重试 1 次）：它是可选优化，不值得让用户多等 3 秒。
- `MOCK_MODE=1` 时用字符 n-gram 重合率打分，只为让这条链路在无 key 时也能被执行到。

### 8.6 统一接口（`__init__.py`）

```python
search(query: str, top_k: int = 5, mode: str = "hybrid") -> list[Chunk]
search_with_ranks(query, top_k=5, mode="hybrid", *, session=None, candidate_k=None) -> list[SearchHit]
rebuild_indexes(session=None, *, force_embedding=False) -> dict[str, int]
```

- `search` 自己开短会话，调用方不接触数据库；返回的块已预加载所属文档（`lazy="joined"`），会话关闭后仍可读，`SearchHit.citation` 可直接产出「文档名 第 N 页 章节」。
- 模式非法在**加载模型之前**抛 `InvalidModeError`。

### 8.7 评测口径

评测集 `data/eval/smoke_queries.jsonl`，每行 `{id, query, gold:[{doc,page}], note}`。金标页由**入库块的正文**反查确定（答案实际落在哪一页），不是在拿到检索结果后回头调整。

| 指标 | 定义 |
| --- | --- |
| Recall@K（文档级） | top_k 中存在一个块的文档名属于金标文档集合 |
| Recall@K（页码级） | top_k 中存在一个块的 `(文档名, 页码)` 命中金标页——**这才是真指标** |

三份手册同源、内容高度重叠（同一张故障码表在 CP320 与 CP650 里都有一份），只报文档级会虚高，因此两级必须同时报。

**v0.6.0 实测（硅基流动 BAAI/bge-m3，203 块，本机）**：

| 模式 | Recall@5（文档） | Recall@5（页码） | 平均耗时 | 说明 |
| --- | --- | --- | --- | --- |
| `vector` | 0.854 | 0.854 | 约 340ms | 48 题，块级金标，冷查询（中位数） |
| `hybrid` | **0.979** | 0.979 | 约 360ms | 含强匹配通道 |
| `hybrid_rerank` | 0.979 | 0.979 | 约 330ms | 重排关闭，透传 |

两条读法，别读错：

- 这是 **`app/eval/run_eval.py` 在 59 条题库（48 条可评分）上的块级指标**，
  比页级严格：一页多个块时，命中「接线示意图」不再算成功。
- **耗时几乎全是远端 embedding 往返**，且抖动很大：同一套代码两次运行能差 30%~70%。
  要复现冷查询需 `EMBED_CACHE_ENABLED=false`；命中缓存时是 7ms 量级，那只发生在重复问同一句话时。
- 唯一权威数字在 `results/report.md`（自动生成）；`data/eval/p2_bench_report.md` 是换模型前的
  历史记录（10 题、页级、本地 bge-small），已加抬头标注「已被取代」。
- **0.938 → 0.979 是一次单变量改动**：查询含确切标识符（`F4-01`/`CP600`/`DI2`）时，
  同时包含这些标识符的块（命中 ≤3 才生效）提到融合最前。诊断过程与 A/B 记录见
  `docs/DESIGN.md` 第一节；切块没动，因为诊断已证明金标块本身完整。
- **页码级 Recall@5 = 1.000 会高估能力**。它的定义是「top-5 里有块落在金标页」，
  而一页可能有 5 个块：命中「接线示意图」也算命中，但真正回答问题的「选型表」可能排在
  第 8 名之外。P3 的引用评测暴露过这一点（见 §11.2）。要看答案可达性，得用「答案块」做金标。

> `data/eval/p2_bench_report.md` 里的数字是**旧栈**（本地 bge-small-zh-v1.5 + 内存余弦）测的，
> 已被上表取代，不要再用。`MOCK_MODE=1` 下的数字只反映 BM25 与管道，**不得**当检索质量结论。

## 9. 模型接入与生成（P2 向量 / 重排 + P3 对话）

### 9.1 三条远端调用

| 用途 | 模块 | 接口 | 失败时的行为 |
| --- | --- | --- | --- |
| 向量 | `app/llm/embed.py` | `POST {EMBED_BASE_URL}/embeddings` | 抛 `EmbeddingUnavailableError`，入库记 `failed_reason`，问答返回 `retrieval_failed` 事件 |
| 重排 | `app/llm/rerank.py` | `POST {RERANK_BASE_URL}/rerank` | 降级为原序 + `degraded` 标记 |
| 对话 | `app/llm/client.py` | `POST {LLM_BASE_URL}/chat/completions` | 结构化 `LlmError`（`config_missing` / `timeout` / `network` / `http_status` / `protocol`） |

共同约定：

- `base_url` 写到 `/v1` 或写到完整路径都能拼对；三项配置任一为空即视为未配置。
- 超时 30s、重试 2 次、指数退避 1s / 2s；只重试超时、连接失败与 408/409/429/5xx。
- **向量有本地缓存**：key = `sha256(model + text)`，命中 `embedding_cache` 就不再发请求。
  入库重跑、评测重跑、同一句 query 反复检索都不再花钱；同一批文本只按 32 条分批发。
- **维度必须自证**：接口返回维度与 `EMBED_DIM` 不一致直接报错，而不是拿两个不同模型的
  向量去算内积——那种错会安静地毁掉整个检索。

### 9.2 MOCK 模式（无 key 跑通全链路）

`MOCK_MODE=1` 时 `app/llm/mock.py` 接管三个模型：

| 桩实现 | 做法 | 目的 |
| --- | --- | --- |
| `stub_embed` | 字符 2~3-gram 的 blake2b 哈希累加成 512 维并归一化 | 让「词面相近的块排在前面」，检索链路可跑 |
| `stub_rerank_scores` | 字符 n-gram 重合率 | 让 `hybrid_rerank` 分支被执行到 |
| `stub_llm` | 复述提示词里的 `<cite>` 标签，拼成「结论 → 排查步骤 → 来源」 | 让 SSE、引用校验、日志全链路可跑 |

三条硬规矩：

1. **必须自我标记**。桩回答末尾带「MOCK 模式…不代表真实模型效果」，`qa_logs.llm_model`
   写 `stub-llm(mock)`，`meta` 事件里 `mock_mode: true`。任何一屏都不许让人以为是真实模型。
2. **完全确定**。全部用 `hashlib`，不用 `hash()`（受 `PYTHONHASHSEED` 影响），
   换机器、重启都得到同一向量，缓存与测试才可复现。
3. **不降低校验强度**。桩模型同样只能引用提示词给出的标签，编造照样会被
   `citations.validate_answer` 抓出来（`--mock --hallucinate` 就是用来证明这一点的）。

它能证明的是「管道与模型解耦」；**不能**证明模型效果——那必须配好 `.env` 实测。

### 9.3 调用链

```
POST /api/chat
  → pipeline.prepare      检索 top_k（记 retrieval_ms）→ 片段转 Citation → 组装 system + user 消息
  → client.stream_chat    直连 /chat/completions，SSE 逐块解析（30s 超时，最多 2 次退避重试）
  → citations.validate_answer   逐条核对回答里的 [文档名 p.页码] 是否落在本次片段上
  → 发 citations 事件（引用列表 + 校验结果 + 用量 + 耗时）
  → save_call_log         写 qa_logs（失败只告警）
```

### 9.4 对话客户端（`client.py`）

- **不用 openai SDK**：直连 `{base_url}/chat/completions`，`base_url` 写到 `/v1` 或写到完整路径都能拼对。少一层版本耦合，超时与重试的语义完全由本模块掌握。
- **超时 30s**：连接、读取、整体截止共用一个 `LLM_TIMEOUT_SECONDS`。流式响应两个 token 之间也可能卡死，所以除 httpx 读超时外还按整体截止时间兜底。
- **重试最多 2 次，指数退避 1s、2s**：只重试超时、连接失败与 408/409/429/5xx。**已经吐出过 token 就不再重试**——重来一遍会让前端看到重复文本，也白烧额度。
- **结构化错误**，不把异常抛给前端：

| `code` | 触发场景 | `retryable` |
| --- | --- | --- |
| `config_missing` | `.env` 缺 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | 否 |
| `timeout` | 连接/读取超时，或超过整体截止时间 | 是 |
| `network` | 连接被拒、DNS、TLS 等传输层错误 | 是 |
| `http_status` | 非 200，附状态码与截断后的响应体（≤300 字） | 429/5xx 是，其余否 |
| `protocol` | HTTP 200 但流里带 `error` 字段 | 否 |
| `interrupted` | 客户端中途断开，生成未完成 | 是 |
| `config_invalid` | `LLM_EXTRA_BODY` 不是合法 JSON 对象 | 否 |

- **`LLM_EXTRA_BODY`**：厂商特有字段（JSON 字符串）原样并入请求体，最后合并、可覆盖默认值。
  不同兼容实现的扩展字段各不相同，硬编码任何一家都会绑死；放开一个 JSON 口子比在代码里写 if 更好排错。
  实测用途：DeepSeek 默认开思考模式，本任务是「照格式抄片段」，关掉更快更稳
  （`{"thinking": {"type": "disabled"}}`）；**顺带解决了一个真实故障**——思考型模型会把推理
  算进 `max_tokens`，1200 的限制被推理吃光，正文直接为空。因此 `LLM_MAX_TOKENS` 默认 4000。

- **用量统计**：优先取服务端 `usage`（`stream_options.include_usage`）。个别兼容实现不认这个参数，被 400 拒绝一次后本进程内关闭并改用字数估算（中文约 1.5 字/token），同时把 `usage_estimated` 标为 `true`——估算值不能当实测值汇报。

### 9.5 提示词（`prompt.py`）

系统提示词是**静态文本**（不带占位符、不拼语料），片段与问题放 user 消息。约束逐条对应可校验的行为：

1. 只依据 `<片段>` 作答，片段以外的知识一律不许使用；
2. 每条结论、每个排查步骤末尾必须带来源，格式 `[文档名 p.页码]`，且必须**原样复制**片段自带 `<cite>` 标签的内容；
3. 片段不足时只回「资料中未找到相关内容」+「缺少的信息：……」，不给步骤、不引用；
4. 片段互相矛盾时指出矛盾并分别列出各自来源，不许替它们选一个；
5. 只覆盖部分问题时只答被覆盖的部分；
6. 中文，结构固定为 `结论 → 排查步骤（分条编号） → 来源`；
7. 不寒暄、不写「作为专家」这类无依据的话。

不写人设。「你是资深工业运维专家」只会提高编造概率，约束与格式才可校验。

**送模型的是完整块正文（`Citation.content`），不是给前端看的 120 字摘要（`Citation.snippet`）。**
这一条是真实评测逼出来的：早期两者混用，故障码表只把前 120 字喂给模型，模型看到的全是表头，
10 题里 6 题只能拒答；分开之后同样 10 题全部作答。引用列表回给前端时仍只给摘要 + `chars`。

user 消息结构固定为三段：`<片段>`（每段带 `<cite>` 标签与完整正文）→ `<合法来源清单>`
（把本次所有合法标签集中列一遍）→ `【问题】`。清单不是冗余：两份手册的故障表几乎一样，
模型会把 CP320 的 p.30 安到 CP650 上，集中列一遍后这种拼装错误消失。

片段正文按 `CONTEXT_MAX_CHARS`（默认 3000）截断。取 3000 是因为库内最长块 2922 字
（故障码表）；早期设 1200 时 203 块里有 58 块被截断，等于把表格腰斩。

### 9.6 引用校验（`citations.py`）

唯一合法来源集合 = **本次检索到的片段**，模型无权自行决定引用哪些块。

| 判定 | 条件 |
| --- | --- |
| 有效 | `[文档名 p.页码]` 的文档名与页码都命中本次片段 |
| 幻觉 | 文档名不在本次片段中，或页码不在该文档的片段页里 |
| 畸形 | 看着像引用（含 `p.` / `第N页` / `.pdf` 后缀）但格式不合要求，如 `[手册 p.24-25]` |

两条防空子：
- **畸形必须算无效**，否则 `[手册 p.24-25]` 会因为「没被识别」把准确率虚报成 100%；
- **带文件后缀却只写文档名也要算一条**（PDF 片段必须给页码，只有 DOCX 这类 `page_no` 为空的片段才允许只写文档名），否则编一个不存在的手册名就能完全不计入分母。

同一来源重复出现按一条去重；文档名容差只到「空白与 `.pdf` / `.docx` 后缀可否省略」，页码严格。正文里的 `[片段1]`、`[图4-2]` 不是引用，不计入分母。

### 9.6.1 支撑性校验（`support.py`）

`citations.py` 只证明「来源存在」，`support.py` 补第二道闸：**这段原文到底支不支持这句话**。
纯规则、零成本、可解释：

1. 按句切出「带引用的结论」，先把引用标签从句子里抠掉（否则 `[CP600…p.24]` 里的型号会被当成结论事实）；
2. 抽事实 token：数值+单位、`Err\d+`、`F4-01`、`DI2`、`CP600-4T22G`；
3. 到引用块正文里找；归一化时去掉竖线/空格（表格里常写成 `| 16 | mm2 |`），
   完整 token 不中再退一步只比数字（避免 `59.0A` vs `59 A` 误判）；
4. 抽不出 token 的定性句记 `unknown`，**不计入准确率**——不拿规则判不了的东西充数。

局限写在 `docs/DESIGN.md` 第四节：同义改写、单位换算、定性结论都判不了；
它还会放过「数字对、单位错」。所以对外表述是「可机械判定的 N 条里 M 条确认」，
不是「结论全对」。

### 9.6.2 拒答判定（`citations.is_refusal`）

拒答的定义是「**结论段只有未找到，且全篇没有任何引用**」，不是「全文出现一次标记」。

```python
head = re.split(r"\n\s*#*\s*排查步骤", answer)[0]
conclusion = head.split("结论", 1)[-1].strip()
return conclusion.startswith(REFUSAL_MARKER) and not _ANY_CITATION.search(answer)
```

原因：模型常「先答完、再补一句片段里没有 X」，例如 q008 答了「AC380V~480V、50/60Hz」并给了引用，
末尾补一句「片段中未给出 CP600 对应的规格」。用全文包含判定会把它算成拒答，
于是作答率在 45/46/47 之间假波动。改判后三轮稳定 47/48，问题项只剩 q018（接线图被压平）。

**通用教训**：指标异常波动时，先怀疑度量，再怀疑被度量的对象。

### 9.6.3 门禁分两级（v1.3.1）

`check_citations.py` 的退出码**只看「有没有编造」**，误拒只记录不阻塞：

| 级别 | 内容 | 影响 |
| --- | --- | --- |
| 硬失败 | 幻觉引用、该拒答却作答、作答但零引用、调用/生成异常 | 退出码 1，必须修 |
| 质量项 | 该答没答（误拒）、回答被 max_tokens 截断 | 逐条列出，退出码仍为 0 |

理由：把「编造」和「该答没答」混成一个布尔值，CI 会因为 1/48 的误拒长期红；
红一个月就没人看了，门禁等于没有。当前状态：**0 条编造，误拒 1/48（q018 接线图被压平）**。

### 9.7 接口（`api/chat.py`）

`POST /api/chat`，请求体 `{question, mode, top_k}`（`question` 1–500 字，`top_k` 1–20，`mode` ∈ 三种模式）。`mode` 非法或问题为空在流开始前返回 422。

响应是 `text/event-stream`，事件顺序固定：

| 事件 | 负载 |
| --- | --- |
| `meta` | 问题、模式、`retrieval_ms`、片段数、引用列表、`llm_configured`、`model` |
| `citations`（含 `support_check`） | 除引用列表与 `citation_check` 外，还带「可机械判定的结论句数 / 被支撑 / 不被支撑 / 判不了」 |
| `delta` | `{"text": "..."}`，可有多条 |
| `citations` | `answer_status`、引用列表（文档名/页码/章节/块内容摘要 `snippet`/完整长度 `chars`/`label`）、`citation_check`、`usage`、`timings`、`finish_reason`、`error` |
| `done` | `{}` |

- 检索失败或模型失败时插入 `event: error`，负载是 `{code, message, retryable, attempts}`，随后仍发 `citations`（说明本来检索到了什么）与 `done`。
- 每帧一行 JSON（`ensure_ascii=False`、`separators=(",", ":")`），禁止裸换行；响应头带 `Cache-Control: no-cache` 与 `X-Accel-Buffering: no`，避免反代把流攒成一整块。
- 检索（CPU 编码）与生成（同步 httpx）都是阻塞代码，统一走线程池（`run_in_threadpool` / `iterate_in_threadpool`），不占用事件循环。
- 不做多轮记忆：没有 history 字段，永远单轮。

### 9.8 校验脚本（`tests/check_citations.py`）

跑评测集 10 题 + 2 个库里确实没有的问题，输出逐题与汇总：

- **引用准确率** = 有效引用 / 去重后引用总数；只要出现一条幻觉引用就判失败，不看平均值；
- 有作答却一个引用都没有 → 失败（无法溯源）；
- `finish_reason=length`（被 `LLM_MAX_TOKENS` 截断）→ 失败，回答尾部可能刚好切在引用中间；
- 评测集里有答案的题被拒答 → 失败（检索或提示词有问题）；
- 库里确实没有的问题未拒答 → 失败；
- 库空或缺向量直接退出码 2，避免「检索恒为空 → 全拒答 → 看起来 100%」的假通过。

## 10. 配置与部署

- 将 `.env.example` 复制为 `.env` 后按需修改。`.env` 已在 `.gitignore` 里，**永远不要提交 key**。
- 三档用法：`MOCK_MODE=1`（无 key，全链路可跑）／填 `EMBED_*` + `LLM_*`（真实模型）／第六周再上 VPS。
- **本机零模型**：`pip install -r requirements.txt` 装完约 420MB（opencv + onnxruntime 约 160MB 是大头），
  不含 torch / paddle / Postgres；同功能的本地方案（torch + sentence-transformers + PaddleOCR）要 5GB 起。
- 默认库是 `data/kb.db`（SQLite，WAL）。`data/` 已被 git 忽略 `.db` 文件，删了重跑入库即可。
- `docker compose up` 起 `app` + `nginx` 两个常驻服务（不再有 postgres 容器）：
  `data/` 挂载到 `/app/data`，`results/` 挂载到 `/app/results`；容器内
  `DATABASE_URL` 默认 `sqlite:////app/data/kb.db`。
- 端口：nginx 发布 `80/443` 到公网；app 只发布到 `127.0.0.1:8000`，供本机调试。
- `certbot` 走 `profiles: ["certbot"]`，平时不启动，只在签发/续期时
  `docker compose --profile certbot run --rm certbot ...`。
- 部署从零到 HTTPS 的全部命令见 `DEPLOY.md`（装 docker → 拉代码 → 配 .env → 起服务 →
  入库 → 证书 → 验收 → 常见故障对照）。
- app 启动时 `create_all` 建表；发现旧结构（`chunks` 无 `embed_model` 列）会打印告警，
  这时删掉 `data/kb.db` 重新入库即可——本项目不用迁移工具。
- 入库：`python -m app.ingest.pipeline data/raw`；补建向量与 BM25：`python -m app.retrieval.bench --rebuild`。
- 向量接口有本地缓存，重复入库、重复评测不会重复计费；换 `EMBED_MODEL` 后会重新编码（`embed_model` 不一致的行会被跳过并告警）。
- 未配置 `LLM_*` 时服务照常启动，`/api/chat` 返回 `config_missing` 事件而不是 500——先能起服务，再谈能不能生成。
- OCR 依赖 `rapidocr-onnxruntime`，只支持 Python < 3.13：Docker（3.11）与本机（3.12）都能装，
  3.13 会被 `requirements.txt` 的环境标记跳过，扫描件识别会报「未安装 RapidOCR」。

## 11. 验收方式

验收拆成两层，**不要混着写**：第一层不需要任何 key，第二层必须有真实模型。

### 11.1 第一层：管道契约（无需 key，MOCK_MODE=1）

```bash
# 1) 单元与全链路测试：入库 → 检索 → 生成 → 引用校验
MOCK_MODE=1 python -m pytest            # 期望 52 passed, 1 skipped（真实模型那条被跳过）

# 2) 真实语料入库 + 建索引 + 检索评测
MOCK_MODE=1 python -m app.ingest.pipeline data/raw
MOCK_MODE=1 python -m app.retrieval.bench --rebuild

# 3) SSE 流：meta → delta… → citations → done
MOCK_MODE=1 uvicorn app.main:app --port 8000
curl -N -X POST localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question":"4T22G 主功率线缆选多粗","mode":"hybrid","top_k":5}'

# 4) 校验器能抓出编造的来源
python tests/check_citations.py --mock --hallucinate     # 期望退出码 1，并指出是哪条假来源
```

第一层通过意味着：引用只能来自本次检索到的片段、空语料必须拒答、日志有耗时与用量。
它**不**意味着模型效果好，`--mock` 与 `MOCK_MODE=1` 下的 100% 只说明校验器与管道正确。

注意：`MOCK_MODE=1` 跑真实语料时，`check_citations.py` 的「拒答题」会失败（桩模型没有
判断力，只要有片段就作答），这是**预期**的：

```
引用：去重后 20 个，有效 20 个，不存在 0 个 → 引用准确率 100.0%
拒答：0/2 题正确拒答
[结论] 未通过
```

### 11.2 第二层：模型效果（需真实接口）

```bash
# .env 里配好 EMBED_* 与 LLM_*
python -m app.retrieval.bench --rebuild              # 用真实向量重建索引
python tests/check_citations.py --out data/eval/p3_citations.json
python -m pytest                                     # 同一套测试改走真实模型，期望 53 passed
```

第二层通过才允许写「10 题引用准确率 100%」。在这之前，任何材料里都只能写
「引用校验器单测通过；真模型验证待 API key 就位后执行」。

**已通过（2026-11-14，DeepSeek `deepseek-flash` + 硅基流动 `BAAI/bge-m3`）**：

```
q01..q10  answered  引用合计 26 个，无效 0 个 → 引用准确率 100.0%
拒答：2/2 题正确拒答
[结论] 通过（退出码 0）
明细：data/eval/p3_citations.json
pytest：53 passed
```

这一轮不是一次就过的，改了三处才到 100%，过程本身就是证据：

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| 10 题里 6 题拒答 | 送模型的是 120 字摘要，故障码表只剩表头 | `Citation.content`（完整正文）与 `snippet`（摘要）拆开 |
| 一题引用 `[CP650… p.30]`，而 CP650 只召回到 p.34/35 | 两份手册的故障表几乎一样，模型把 CP320 的页号安到 CP650 上 | user 消息里加 `<合法来源清单>`，并明令「禁止把 A 文档页码安到 B 文档」 |
| 一题正文为空、`finish_reason=length` | 思考模式的推理算进 `max_tokens=1200`，被推理吃光 | `LLM_MAX_TOKENS=4000` + `LLM_EXTRA_BODY` 关思考 |

### 11.3 第三层：界面（无需 key 也能看结构，作答需要 key）

```bash
python -m pytest tests/test_web.py -q      # 页面渲染、上传/删除/重解析、Markdown 表格
uvicorn app.main:app --port 8000           # 打开 http://127.0.0.1:8000
```

浏览器要过的四条：

- 三个页面都能打开，空状态有下一步提示（不是白屏）；
- 提问时文字逐个出现，回答下方有引用卡片，点开能看到该块原文与页码；
- 把后端停掉再提问，页面显示「无法连接后端（网络中断或服务未启动）」，不是一直转圈；
- 手机（或 390px 视口）下三页都没有横向滚动条。

第二层的验收口径：

- 10 题的引用准确率必须 100%，出现一条不存在的来源就算未通过；
- 问「CP600-4T22G 的出厂序列号编码规则」这类库里没有的问题，必须回「资料中未找到相关内容」并说明缺哪类信息；
- `qa_logs` 里每行都有 `retrieval_ms` / `generation_ms` / token 用量，且 `usage_estimated` 如实标记。

## 12. 前端与部署（P5）

### 12.1 三个页面

| 路径 | 内容 | 交互（原生 JS） |
| --- | --- | --- |
| `/` | 输入框、检索模式下拉（三种）、流式回答、引用卡片 | `fetch` + `ReadableStream` 手写 SSE 解析；引用卡片点「查看原文」才去拉 `/api/chunks/{id}` |
| `/documents` | 上传、列表（状态 / 页数 / 块数 / 已向量化）、删除、重新解析 | 上传后每 2.5s 轮询，状态进终态就停 |
| `/eval` | 报告 Markdown 渲染成表格，可切换报告文件 | 服务端渲染，只有下拉切换走一次 GET |

### 12.2 几条实现决定

- **不用 `EventSource`**：它只能 GET，而 `/api/chat` 是 POST。手写 `getReader()` 循环，
  按 `\n\n` 切帧、认 `event:` / `data:` 两行——这比为了流式把问题塞进 query string 干净得多。
- **90 秒兜底 `AbortController`**：后端卡在模型调用上时，页面不能永远转圈。
- **错误必须是人话**：`fetch` 直接 reject 抛的是 `TypeError: Failed to fetch`，
  浏览器原文不能直接贴给用户。统一翻成「无法连接后端（网络中断或服务未启动）」，
  再附一行排查命令；且此时**不保留空白回答框**，避免"看起来像白屏"。
- **三态齐全**：空状态给下一步动作（「输入一段故障现象」/「还没有文档，先上传」）、
  加载状态给具体阶段（参考「正在检索片段…」→「正在生成…」）、错误状态给原因 + 排查线索。
- **样式**：白底 `#fff`、深色字 `#1b1f24`、无阴影无渐变无动画；正文 16px，最小 14px；
  按钮 `min-height: 44px`（触控目标）；窄屏（≤600px）把文档表格从"表格"变成"卡片"，
  用 `td[data-label]::before` 补字段名——横向滚动的表格在手机上会把「操作」列藏起来，比挤压更糟。

### 12.3 部署拓扑

```
浏览器 ── 80/443 ──> nginx（容器）──> app:8000（只绑 127.0.0.1）
```

- nginx 的 `location /api/chat` 必须 `proxy_buffering off` + 300s 读超时，否则流式会被攒成一整块。
- `client_max_body_size` 必须与 `MAX_UPLOAD_MB` 对齐：不一致时 413 由 nginx 先返回，
  后端收不到请求，前端只能看到一个没有原因的失败。
- app 的 8000 端口只发布到 `127.0.0.1`，公网入口只有 nginx，避免绕过反代直连。
- certbot 用 webroot 方式签证书，`deploy/nginx.https.conf` 是切 HTTPS 后的配置；
  `DEPLOY.md` 第 6 步给出完整命令与续期 cron。

### 12.4 实测（本机，390×844 手机视口）

用 CDP 的 `Emulation.setDeviceMetricsOverride` 把视口压到真机宽度（Edge 命令行
`--window-size` 在 Windows 上有 504px 下限，测不出手机布局）：

| 检查 | 结果 |
| --- | --- |
| 三页横向溢出 | `/`、`/documents`、`/eval` 均为 `scrollWidth == clientWidth == 390` |
| 提问 → 流式 | 回答 380 字、5 张引用卡片，meta 行显示检索 12ms / 生成 1994ms / 5526 tokens |
| 引用展开 | 点「查看原文」拉到完整块正文（表格原文可读），按钮变「收起原文」 |
| 后端不可达 | 拦截 `/api/chat` 后显示「无法连接后端（网络中断或服务未启动）」+ 排查命令，无残留转圈 |
| 上传链路 | curl 上传 1 页 PDF → 12s 后 `status=indexed, chunks=1, chunks_vectorized=1` |

未验证的两项，写清楚免得被当成已验：**真机**没测（用的是视口模拟），
**真实 nginx** 没跑（本机无 Docker），故 `deploy/nginx.conf` 的语法与 SSE 不缓冲效果
需在 VPS 上按 `DEPLOY.md` 第 7 步自检。

## 13. 题库与块级评测（P6）

### 13.1 为什么要有第二套题库

`smoke_queries.jsonl` 只有 10 条、金标标到「页」，两个问题：样本太小、页级指标会把
「同页的接线图命中」算成成功。于是补一套 **块级金标**题库：每条题的金标是**出题时依据的那个 chunk**，
出题时还不知道检索结果，因此不含「照着答案找证据」的偏差。

### 13.2 生成方式（`app/eval/generate_questions.py`）

| 步骤 | 做法 |
| --- | --- |
| 抽样 | 按章节去重（同一章节最多 3 题），表格块→查表型，叙述块→推理型；安全注意事项类套话排除 |
| 出题 | 每块一次调用，要求「只能靠这段原文回答、只问一个事实、答案必须完整落在原文里」 |
| 跨文档 | 按「同章节标题」在两份不同手册间配对，要求问出差异 |
| 无答案 | 人工给主题（价格/库存/证书/工期/序列号…），再用**决定性关键词**核验语料确实没有；命中则丢弃 |
| 产物 | `data/eval/questions.jsonl`，`gold: [{chunk_id, doc, page}]`，无答案题标 `expect: "refusal"` |

两条踩过的坑，已写进提示词与流程：

1. **关键词核验不能用宽泛词**。第一轮用「柜」「培训」做探针，把本来无答案的题误杀，
   11 条只凑到 6 条（无答案占比 11%，低于 15% 的下限）。
2. **模型会出复合问题**。「两种设备的端子分别在哪」这类题需要多段原文，金标只覆盖其中一段，
   结果模型正确地说「另一部分资料里没有」却被判成拒答错误。现在提示词明令「只问一个事实」，
   已生成的复合题在人工复核时删除。

### 13.3 评测方式（`app/eval/run_eval.py`）

- 指标：**块级 Recall@K**（金标 chunk 是否进 top-K）为主，页码级/文档级仅作对照；
- 分题型统计 + 最多 10 条失败案例，每条标注是「页码命中块未命中」「文档命中页码不同」还是「完全未命中」；
- 输出 `results/report.md`——**唯一权威数字来源**，README 与 DESIGN 引用它，不再手工抄第二份；
- 生成侧数字从 `data/eval/p3_citations.json` 读取，与检索侧拼进同一份报告。

首条查询的冷启动不计入耗时统计（否则「平均」会大于「P95」）；不同模式的延迟不可横向比较——
第一个跑的模式要付远端 embedding 往返，后面的模式命中缓存。

### 13.4 CI 与压测（v1.1.1）

**CI 不是装饰。** 三道闸：

| 闸 | 触发条件 | 内容 |
| --- | --- | --- |
| 1 | 每次 push | `MOCK_MODE=1` 跑全量 pytest（不需要 key） |
| 2 | 配了 `EMBED_API_KEY` | `tests/fixtures/generate_smoke_corpus.py` 现场生成 1 页 PDF + 3 道题 → 入库 → 建索引 |
| 3 | 配了 LLM + EMBED key | `check_citations.py --queries data/eval/ci_smoke.jsonl`，**不挂 `\|\| true`** |

为什么语料要现场生成而不是提交进仓库：二进制不进仓库，也避免「示例文档算不算侵权」的争议。
本地已按 CI 的步骤跑通一次：3 题（2 答 1 拒）、引用 2/2 有效、退出码 0。

**压测的表格判定分两档**，因为混在一起会毁掉清单可信度：

- `列错位`：多数数据行 N 列、少数行列数差 ≥2 → 真会读错，值得修；
- `表头合并被拍平`：表头出现连续空单元格，数据行仍对齐 → 轻度，先记录不修。

报告留「人工复核」空列，工具判断与人工结论分开记。

## 14. 交付材料（P6）

| 文件 | 内容 | 写作纪律 |
| --- | --- | --- |
| `README.md` | 一句话定位、mermaid 架构图、检索/生成两张结果表、三种跑通方式、目录说明、截图占位、已知限制 | 不写「基于大模型的智能运维平台」这类空话；每个数字都能在仓库里复现 |
| `docs/DESIGN.md` | 四个问题：为什么混合检索 / 切块为什么这么定 / 重排值不值 / 什么时候会答错 | 每个答案带实测数字或具体案例；没测的地方明说「没测」 |
| `docs/DEMO.md` | 三分钟脚本（按秒：0:00 说什么、0:15 点什么）、演示前检查清单、四个高频追问、四种故障预案 | 台词按「结论 → 证据 → 边界」排；拒答当卖点而不是 bug |
| `docs/RESUME.md` | 三条简历 bullet（≤2 行，数字 `{X}` 占位）+ 每个占位的取值出处表 | 只写实测值；「提升 10%」这类换算一律不许用 |

全仓库体检（P6 要求）：

```bash
grep -rn "TODO\|FIXME" app tests          # 0 条
grep -rn "sk-[A-Za-z0-9]\{16,\}" --include=*.py .   # 0 条（key 只在 .env，且已 gitignore）
grep -rn "^\s*#\s*\(import\|def \|class \|return \)" app   # 0 条（无注释掉的死代码）
```

README 的「10 分钟跑通」是**实测过的**：在空库 + 1 页 PDF + `MOCK_MODE=1` 下走
「建 venv → 装依赖 → 入库 → 建索引 → 起服务 → 提问」，首页 200、回答带一条合法引用
（`[演示手册.pdf p.1]`，accuracy 1.0）。仓库不附带真实手册与评测集，README 第 0 步明确要求自备。

## 15. 阶段边界与已知风险

- 检索、评测与生成已实现；前端与 `documents` / `admin` 两组 `/api/*` 仍为空壳（P4）。
- 语料只有 3 份同厂商手册、203 个块，top-5 就覆盖了 2.5% 的库，Recall@5 容易饱和；扩语料后这个指标才有区分度，届时应同步扩评测集。
- 表格为原子单位，超长表格会让单块超过 800 字；这类超长块在向量空间里方向偏「平均」，语义检索容易被带偏——页码级指标的失分多来自这里。
- 标题识别依赖字体与编号，扫描版文档、无编号且同字号的标题可能识别不到，会退化为按字号/段落切分。
- OCR 质量受扫描清晰度影响，低置信度段落只做标记，不做人工复核。
- 页眉页脚按「位置 + 跨页重复」剔除；页数很少（< 3 页）的文档、或每页页眉都改写（如带页码递增章节号）的文档仍可能残留少量页眉文字。
- 合并单元格表格、图形化端子排、嵌套表格转 markdown 会串行或重复渲染（例如表头「额定输入电流」被拆成「额 型号 入」错位），纯文本抽取无法根治。
- 图注与紧随其后的表题可能被拼成一行（如「…示意图表4-2 …」），不影响检索但影响可读性。
- DOCX 无法取得页码，`page_no` 为空，引用时只能给章节。
- 重排默认关闭：开启后单条查询会增加到数百毫秒，10 条候选以上就会突破 500ms 预算；生成链路下首 token 延迟会把它进一步放大，因此仍默认关闭。
- **引用准确率只证明「来源真实」，不证明「来源支撑该结论」**。`[文档名 p.X]` 存在只能排除编造文献，答非所引仍需人工抽查；本阶段不做语义蕴含判定（要有第二个模型才有意义，代价与误判都要另算）。
- 拒答依赖模型遵守提示词。片段与问题「似是而非」时（例如问 CP600 却只召回 CP650 的内容）仍可能被强行作答；观测手段只有 `qa_logs.answer_status` 与人工抽检，没有自动判定。
- 流式响应的 `usage` 只有部分兼容实现会返回；拿不到就是字数估算，`usage_estimated=true` 的行不能用于精确计费。
- 生成层单次调用含首 token 等待，端到端耗时主要取决于所接模型；`qa_logs` 已分别记 `retrieval_ms` 与 `generation_ms`，便于区分是谁慢。
- **桩模型的输出没有语义价值**。`stub_embed` 只是字符 n-gram 哈希，「油压偏低」能召回「油压低于 0.15MPa」纯属词面重合；用它跑出来的 Recall 不能当检索质量，只能证明链路通。
- **向量改为远端后多了一次网络往返**：入库要按批发请求（203 块约 7 批），首次入库比本地模型慢；好处是不再需要下载模型、也不再吃 1~2GB 内存。缓存命中后重复入库接近零成本。
- 外部接口有配额与限流风险：429 会让入库变慢（`embed.py` 重试 2 次后失败，该文件记 `failed_reason`，可重跑续上，不会产生重复块）。
- `embedding_cache` 会随语料增长（每块一行、512 维约 2KB）；10 万块约 200MB，够用，但清理策略目前只有「删库重建」。
- SQLite 单文件适合单机与演示：并发写（多进程同时入库）会退化成排队。真上多实例时要换 Postgres，检索入口仍是 `search()` 一个函数。
- `rapidocr-onnxruntime` 只支持 Python < 3.13；本机 3.12 / Docker 3.11 可用。扫描件是这三份手册的主要来源之一，**换 3.13 会让入库大面积失败**，这一点比任何参数调优都优先。
- **bench 的多模式对比会互相「加热」**：第一个模式承担冷查询（每次一次 embedding 往返），
  后两个模式命中 `embedding_cache`，于是显示 6~7ms。这个数字只能说明融合与 BM25 很快，
  不能当首问延迟。要测冷启动得关缓存或单独跑一个模式。
- **页码级 Recall@5 = 1.000 会高估能力**：它只要求「top-5 里有一块落在金标页」，
  一页有多个块时，「接线示意图」命中而「选型表」排在第 8 名也算满分。
  P3 的引用评测就是这么暴露出「召回页对了但答案块没进上下文」的（早期 6/10 拒答里有一半是这个原因）。
  想量化答案可达性，得把金标改成「答案所在块」而不是「页」。
- 本机没有 Docker；P1/P2/P3/P5 的验证都在本机 Python 3.12 + SQLite 上完成
  （`pytest` 真实模型下 82 passed；`MOCK_MODE=1` 下 81 passed / 1 skipped）。
- 真实模型下的引用准确率与拒答能力必须在配好 `.env` 的环境跑 `tests/check_citations.py` 才能确认；
  该记录已于 2026-11-14 在 59 题题库上重跑（`data/eval/p3_citations.json`，92 个引用 0 幻觉，
  11/11 拒答；48 条可答题里 46 条作答，2 条因检索未命中而拒答）。但**换模型、换语料、
  改提示词之后这份记录就作废**，必须重跑。
- **不做登录 = 任何拿到域名的人都能上传、删除、提问**。演示期的补偿只有「8000 只绑本机 +
  公网只走 nginx + 关掉其它端口」；真要长期公开，至少加 nginx `allow/deny` 白名单，
  或者在做 P6 时补一层 Basic Auth（本阶段按约定没做）。
- **解析是进程内的后台任务**（FastAPI `BackgroundTasks`）：进程崩了会留下 `pending/parsing`
  记录，页面表现为一直「解析中」。恢复方式：点「重新解析」（`ingest_file` 会删旧记录重建）。
  也正因为任务与锁都在单进程内，**app 不要开多 worker**（`--workers 2` 会让入库串行锁失效）。
- 1GB 内存的机器处理 160 页手册（含 OCR）可能被 OOM 杀掉，`DEPLOY.md` 第 1 步要求加 2G swap
  就是这个原因；内存更小的话建议先只上传关键章节。
- 评测页只负责**渲染报告文件**，不判断报告是否过期。`data/eval/p2_bench_report.md` 里的数字
  来自旧栈（本地 bge-small），页面上仍会照样显示——看之前先确认文件时间。
- 静态资源没有加版本号哈希，`nginx` 又给 `/static/` 设了 1 小时缓存：改了 CSS/JS 后
  手机上可能要强制刷新才看到新样式。
- `deploy/nginx.conf` 与 `nginx.https.conf` **没有在本机跑过**（无 Docker），
  只做了语法层面的检查；上线时必须按 `DEPLOY.md` 第 7 步逐条自检，尤其是流式是否被缓冲。
- README 里的三张截图是**占位**（`docs/images/` 只有 `.gitkeep`），补图前 README 会显示破图；
  提交前要么补图，要么把那三行删掉——破图比没有图更减分。
- `docs/DESIGN.md` 里引用的 Recall 与重排耗时来自**旧栈**（本地 bge-small / bge-reranker）与
  **新栈**（bge-m3 远端）两次不同阶段的实测，文中都标了来源。换模型后这些数字要重测，
  否则材料与仓库会不一致。
- 简历 bullet 里的数字是 `{X}` 占位，取值出处写在同文件的表里。**填之前请重跑一次**
  `pytest` 与 `check_citations.py`，避免提交的数字和最后一次运行对不上。
- 目前没有迁移工具，字段变更需手动处理既有库（`chunks.embedding` 从 pgvector/JSON 改为 BLOB、新增 `embed_model`，都是靠删除 `data/kb.db` 重建完成的；`init_db()` 会检测并提示）。
