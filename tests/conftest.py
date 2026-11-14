"""pytest 全局前置：把数据库指向临时 SQLite 文件，并按需开启 MOCK_MODE。

两件事必须在**任何 `app.*` 导入之前**完成：

1. `DATABASE_URL` 指向临时目录里的单文件库。绝不能让测试碰到 `data/kb.db`——
   测试会清表、会写入假文档，跑一次就把真实语料洗了。
2. 没有任何真实模型配置时，把 `settings.mock_mode` 置为 True。这样 `pytest`
   在无 key 的机器上也能跑通全链路；配好 `.env` 后同一套测试走真实模型。

用临时文件而不是 `:memory:`：SQLite 的内存库是「每连接一份」，入库和检索
拿到的是两个不同的库，全链路测试会莫名其妙查不到数据。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_DIR = Path(tempfile.mkdtemp(prefix="kb-pytest-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_DIR / 'test.db').as_posix()}"

from app.config import settings  # noqa: E402  必须在上面设置环境变量之后导入

# 真实模型可用的判据：向量与对话两项接口都配齐了
REAL_MODELS_READY = bool(
    settings.embed_base_url
    and settings.embed_api_key
    and settings.embed_model
    and settings.llm_base_url
    and settings.llm_api_key
    and settings.llm_model
)

if not REAL_MODELS_READY:
    # 桩模型只影响「模型输出」，管道、校验、存储的行为完全一致
    settings.mock_mode = True
