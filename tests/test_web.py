"""Web 层测试：三个页面能渲染、文档接口能增删、报告能转成表格。

不启动真实解析：`_ingest_document` 在测试里被替换成记录器。解析本身（PDF/OCR/切块）
由 `test_mock_pipeline.py` 用真实 PDF 覆盖，这里只测 HTTP 层与状态流转。
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.api import documents as documents_api
from app.config import settings
from app.db import SessionLocal, init_db
from app.main import app
from app.models import Chunk, Document
from app.web import markdown


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """把上传目录指到临时目录，避免测试往仓库的 data/raw 里写文件。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(settings, "raw_dir", str(raw_dir))
    init_db()
    with SessionLocal() as session:
        session.execute(delete(Chunk))
        session.execute(delete(Document))
        session.commit()
    with TestClient(app) as test_client:
        yield test_client


def _make_document(name: str = "手册.pdf", status: str = "indexed", chunks: int = 2) -> int:
    """直接写库造一份文档，省掉解析过程。"""
    with SessionLocal() as session:
        document = Document(
            title=Path(name).stem,
            filename=name,
            source_path=name,
            doc_type="pdf",
            sha256=f"sha-{name}",
            status=status,
            page_count=12,
        )
        session.add(document)
        session.commit()
        for index in range(chunks):
            session.add(
                Chunk(
                    document_id=document.id,
                    page_no=index + 1,
                    section="4.4 主功率线缆",
                    content=f"第 {index + 1} 块正文：按表4-2 选择线径。",
                )
            )
        session.commit()
        return document.id


# ---------------- 页面 ----------------


def test_chat_page_renders_form_and_all_modes(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert 'id="qa-form"' in body
    assert 'id="question"' in body
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        assert f'value="{mode}"' in body
    assert "还没有提问" in body  # 空状态提示必须在
    assert "/static/chat.js" in body


def test_documents_page_renders_upload_form(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert 'type="file"' in response.text
    assert "上传并解析" in response.text


def test_eval_page_without_report_shows_guidance(client: TestClient, monkeypatch) -> None:
    """没有报告时不能白屏，要告诉人怎么生成。"""
    monkeypatch.setattr(settings, "eval_report_path", "results/不存在.md")
    monkeypatch.setattr(settings, "eval_report_dir", "data/eval/不存在")

    response = client.get("/eval")

    assert response.status_code == 200
    assert "还没有可读取的评测报告" in response.text
    assert "app.retrieval.bench --save" in response.text


def test_eval_page_renders_markdown_table(client: TestClient, tmp_path, monkeypatch) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "# 检索评测\n\n| 模式 | Recall@5 |\n| --- | --- |\n| vector | 0.900 |\n| hybrid | 1.000 |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "eval_report_path", str(report))
    monkeypatch.setattr(settings, "eval_report_dir", str(tmp_path / "empty"))

    response = client.get("/eval")

    assert response.status_code == 200
    assert "<table>" in response.text
    assert "<th>Recall@5</th>" in response.text
    assert "<td>1.000</td>" in response.text
    assert report.name in response.text  # 标明读的是哪个文件


def test_eval_page_rejects_path_traversal(client: TestClient, tmp_path, monkeypatch) -> None:
    """报告名只从已列举的文件里选，`../` 这类参数直接当不存在。"""
    report = tmp_path / "report.md"
    report.write_text("# 标题\n", encoding="utf-8")
    monkeypatch.setattr(settings, "eval_report_path", str(report))
    monkeypatch.setattr(settings, "eval_report_dir", str(tmp_path / "empty"))

    response = client.get("/eval", params={"report": "../../etc/passwd"})

    assert response.status_code == 200
    assert "指定的报告不存在" in response.text


# ---------------- Markdown ----------------


def test_markdown_table_and_inline_formatting() -> None:
    html = markdown.render(
        "## 标题\n\n| 项 | 值 |\n| --- | --- |\n| **粗体** | `code` |\n\n- 一\n- 二\n"
    )

    assert "<h2>标题</h2>" in html
    assert "<th>项</th>" in html
    assert "<strong>粗体</strong>" in html
    assert "<code>code</code>" in html
    assert html.count("<li>") == 2


def test_markdown_escapes_html() -> None:
    """报告里的 HTML 只当文本显示，`<script>` 不能被浏览器执行。"""
    html = markdown.render("<script>alert(1)</script>")

    assert "&lt;script&gt;" in html
    assert "<script>" not in html


# ---------------- 文档接口 ----------------


def test_list_documents_reports_counts(client: TestClient) -> None:
    document_id = _make_document("CP600空压机一体机手册.pdf", chunks=3)

    data = client.get("/api/documents").json()

    assert data["total"] == 1
    item = data["documents"][0]
    assert item["id"] == document_id
    assert item["status"] == "indexed"
    assert item["pages"] == 12
    assert item["chunks"] == 3
    assert item["chunks_vectorized"] == 0  # 造数据没建向量


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "不支持的文件类型" in response.json()["detail"]


def test_upload_saves_file_and_schedules_parse(client: TestClient, monkeypatch, tmp_path) -> None:
    """上传只负责落盘 + 排后台任务，解析不在请求里做。"""
    calls: list[Path] = []
    monkeypatch.setattr(documents_api, "_ingest_document", lambda path, **kwargs: calls.append(path))

    response = client.post(
        "/api/documents",
        files={"file": ("CP600空压机一体机手册.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "CP600空压机一体机手册.pdf"
    assert "后台解析" in payload["message"]
    assert calls and calls[0].name == "CP600空压机一体机手册.pdf"
    assert calls[0].read_bytes() == b"%PDF-1.4 fake"


def test_upload_same_name_does_not_overwrite(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(documents_api, "_ingest_document", lambda path, **kwargs: None)
    body = {"file": ("手册.pdf", b"%PDF-1.4 a", "application/pdf")}

    first = client.post("/api/documents", files=body)
    second = client.post("/api/documents", files={"file": ("手册.pdf", b"%PDF-1.4 b", "application/pdf")})

    assert first.json()["filename"] == "手册.pdf"
    assert second.json()["filename"] == "手册-1.pdf"


def test_delete_document_removes_rows_and_file(client: TestClient, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(documents_api, "_ingest_document", lambda path, **kwargs: None)
    upload = client.post(
        "/api/documents",
        files={"file": ("待删.pdf", b"%PDF-1.4 x", "application/pdf")},
    )
    document_id = _make_document("待删.pdf")

    response = client.delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["deleted"] == document_id
    assert client.get("/api/documents").json()["total"] == 0
    assert not (Path(settings.raw_dir) / upload.json()["filename"]).exists()
    # 分块必须一起消失：SQLite 默认不校验外键，漏了就是「删了文档但块还能被检索到」
    with SessionLocal() as session:
        left = session.execute(
            select(Chunk.id).where(Chunk.document_id == document_id)
        ).all()
    assert left == []


def test_delete_missing_document_is_404(client: TestClient) -> None:
    assert client.delete("/api/documents/9999").status_code == 404


def test_reingest_missing_file_is_410(client: TestClient) -> None:
    """源文件被手动删掉时，重新解析要明确说「文件不在了」，不是 500。"""
    document_id = _make_document("没有文件.pdf")

    response = client.post(f"/api/documents/{document_id}/ingest")

    assert response.status_code == 410
    assert "源文件已不存在" in response.json()["detail"]


def test_reingest_sets_pending_and_schedules(client: TestClient, monkeypatch) -> None:
    document_id = _make_document("重解析.pdf")
    (Path(settings.raw_dir) / "重解析.pdf").write_bytes(b"%PDF-1.4 y")
    calls: list[Path] = []
    monkeypatch.setattr(documents_api, "_ingest_document", lambda path, **kwargs: calls.append(path))

    response = client.post(f"/api/documents/{document_id}/ingest")

    assert response.status_code == 202
    assert calls
    with SessionLocal() as session:
        status = session.execute(select(Document.status).where(Document.id == document_id)).scalar_one()
    assert status == "pending"


# ---------------- 块内容接口 ----------------


def test_chunk_endpoint_returns_full_text(client: TestClient) -> None:
    _make_document("块.pdf", chunks=1)
    with SessionLocal() as session:
        chunk_id = session.execute(select(Chunk.id)).scalar_one()

    data = client.get(f"/api/chunks/{chunk_id}").json()

    assert data["doc"] == "块.pdf"
    assert data["page"] == 1
    assert "按表4-2 选择线径" in data["content"]


def test_chunk_endpoint_404(client: TestClient) -> None:
    response = client.get("/api/chunks/999999")

    assert response.status_code == 404
    assert "块不存在" in response.json()["detail"]


# ---------------- 静态资源 ----------------


def test_static_css_is_served(client: TestClient) -> None:
    response = client.get("/static/style.css")

    assert response.status_code == 200
    assert "--ink" in response.text  # 确认拿到的是样式文件本身
