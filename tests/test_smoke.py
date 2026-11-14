"""冒烟测试：验证 /health 的路由与响应形状。

本阶段不启动真实 PostgreSQL，通过 monkeypatch 替换建表与连通性探测，
使测试可在纯 Python 环境中运行；真实数据库连通性由 compose 环境下的 curl 验收。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """数据库探测成功时，/health 应返回 200 与 {"db": "ok"}。"""
    monkeypatch.setattr("app.main.init_db", lambda: None)
    monkeypatch.setattr("app.main.check_db", lambda: None)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"db": "ok"}
