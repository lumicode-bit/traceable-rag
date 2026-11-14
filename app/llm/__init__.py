"""远端模型接入层：本机不装任何深度学习框架，所有模型调用都是 HTTP。

- `embed.py`    向量：OpenAI 兼容 `/embeddings` + 本地缓存（同一文本只发一次请求）
- `rerank.py`   重排：`/rerank`，未配置或失败时降级为原序并标记 degraded
- `client.py`   对话：`/chat/completions` 流式，超时/重试/结构化错误
- `prompt.py`   提示词：约束「只能依据片段、必须带来源、不足就拒答」
- `citations.py` 引用校验：纯函数，可离线跑
- `pipeline.py` 编排：检索 → 提示词 → 生成 → 校验 → 写 qa_logs
- `mock.py`     桩模型：`MOCK_MODE=1` 时替换上面三者，让无 key 的机器也能跑通全链路

分工的原则：**谁负责调用模型，谁负责处理失败**；上层只看到事件与结构化错误。
"""
