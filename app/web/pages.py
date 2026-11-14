"""页面路由：三个页面，服务端渲染骨架 + 原生 JS 处理交互。

- `/`          问答页：输入框、流式回答、引用卡片、三种检索模式
- `/documents` 文档管理：上传、列表、删除、重新解析
- `/eval`      评测页：把报告 Markdown 渲染成表格

模板目录用**绝对路径**定位（相对 `__file__`），这样从任意工作目录启动 uvicorn
都能找到模板；只在读 `.env` 时需要 cwd，那是另一回事。
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.retrieval import MODES
from app.web import markdown

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = PROJECT_ROOT / "web" / "templates"
STATIC_DIR = PROJECT_ROOT / "web" / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
router = APIRouter(tags=["pages"])

MODE_LABELS = {
    "vector": "向量（语义）",
    "hybrid": "混合（语义 + 关键词）",
    "hybrid_rerank": "混合 + 重排",
}


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    """问答页：模式来自后端，前端不允许自己编一个模式出来。"""
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "nav": "chat",
            "modes": [(mode, MODE_LABELS.get(mode, mode)) for mode in MODES],
            "default_mode": settings.chat_mode if settings.chat_mode in MODES else MODES[0],
            "default_top_k": settings.chat_top_k,
            "mock_mode": settings.mock_mode,
        },
    )


@router.get("/documents", response_class=HTMLResponse)
def documents_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "documents.html",
        {"nav": "documents", "max_upload_mb": settings.max_upload_mb},
    )


@router.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request, report: str = "") -> HTMLResponse:
    """评测页：渲染报告 Markdown；没有报告时给出「怎么生成」的指引，不留白屏。"""
    reports = _list_reports()
    selected = _select_report(reports, report)
    body = ""
    error = ""
    if selected is not None:
        try:
            body = markdown.render(selected["path"].read_text(encoding="utf-8"))
        except OSError as exc:
            error = f"读取报告失败：{exc}"
    elif reports:
        error = f"指定的报告不存在：{report}"
    return templates.TemplateResponse(
        request,
        "eval.html",
        {
            "nav": "eval",
            "reports": reports,
            "selected": selected,
            "body": body,
            "error": error,
            "candidates": [settings.eval_report_path, f"{settings.eval_report_dir}/*.md"],
        },
    )


def _list_reports() -> list[dict]:
    """可渲染的报告：配置指定的那个 + `data/eval` 下所有 .md（按修改时间倒序）。"""
    items: list[dict] = []
    seen: set[Path] = set()

    configured = Path(settings.eval_report_path)
    if configured.is_file():
        items.append(_describe(configured))
        seen.add(configured.resolve())

    directory = Path(settings.eval_report_dir)
    if directory.is_dir():
        for path in sorted(
            directory.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True
        ):
            resolved = path.resolve()
            if resolved in seen:
                continue
            items.append(_describe(path))
            seen.add(resolved)
    return items


def _describe(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": path,
        "size_kb": round(stat.st_size / 1024, 1),
        "modified": _format_time(stat.st_mtime),
    }


def _select_report(reports: list[dict], requested: str) -> dict | None:
    """按文件名挑选；只认已经列举出来的文件，天然挡住路径穿越。"""
    if not reports:
        return None
    if requested:
        for item in reports:
            if item["name"] == requested:
                return item
        return None
    return reports[0]


def _format_time(timestamp: float) -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
