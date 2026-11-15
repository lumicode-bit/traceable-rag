"""配置解析的容错：空字符串必须回落到默认值，不能把整个 Settings 构造炸掉。

这不是假想问题——GitHub Actions 里**未设置的 secret 会变成空字符串**，
`EMBED_DIM=` 会让 pydantic 报 `int_parsing`，pytest 连用例都收集不了（退出码 4）。
用户在 `.env` 里把某项留空也是同一个错。
"""
from __future__ import annotations

from app.config import Settings


def test_blank_numeric_values_fall_back_to_defaults() -> None:
    # `_env_file=None` 不读 .env：否则比的是「本机 .env 里的值」，不是代码默认值
    blank = Settings(_env_file=None, embed_dim="", chat_top_k="", context_max_chars="")
    default = Settings(_env_file=None)

    assert blank.embed_dim == default.embed_dim
    assert blank.chat_top_k == default.chat_top_k
    assert blank.context_max_chars == default.context_max_chars


def test_blank_string_values_are_allowed() -> None:
    settings = Settings(embed_api_key="", llm_api_key="", llm_extra_body="")

    assert settings.embed_api_key == ""
    assert settings.llm_extra_body == ""


def test_real_values_still_parse() -> None:
    settings = Settings(embed_dim="1024", chat_top_k="3", mock_mode="1")

    assert settings.embed_dim == 1024
    assert settings.chat_top_k == 3
    assert settings.mock_mode is True
