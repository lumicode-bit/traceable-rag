/* 文档管理页：上传 → 后台解析 → 轮询状态；删除、重新解析。 */
"use strict";

const form = document.getElementById("upload-form");
const fileEl = document.getElementById("file");
const uploadBtn = document.getElementById("upload");
const refreshBtn = document.getElementById("refresh");
const listEl = document.getElementById("list");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const countEl = document.getElementById("count");

const POLL_MS = 2500;
let timer = null;
let pollUntil = 0;

const STATUS_TEXT = {
  pending: "排队中",
  parsing: "解析中",
  indexed: "已入库",
  failed: "失败",
};

const STATUS_CLASS = {
  pending: "badge badge-running",
  parsing: "badge badge-running",
  indexed: "badge badge-indexed",
  failed: "badge badge-failed",
};

form.addEventListener("submit", async function (event) {
  event.preventDefault();
  const file = fileEl.files && fileEl.files[0];
  if (!file) {
    showError(errorEl, "请先选择一个文件。");
    return;
  }
  clearError(errorEl);
  uploadBtn.disabled = true;
  setStatus(statusEl, "正在上传 " + file.name + "…");

  const body = new FormData();
  body.append("file", file);

  try {
    const data = await fetchJSON("/api/documents", { method: "POST", body: body });
    setStatus(statusEl, data.message + "：" + data.filename);
    fileEl.value = "";
    pollUntil = Date.now() + 120000; // 解析可能要几分钟，先撑两分钟轮询
    await load();
    schedulePoll();
  } catch (err) {
    let hint = "可重试一次；连续失败请看后端日志。";
    if (err.status === 413) hint = "文件太大：调大 MAX_UPLOAD_MB，或 nginx 的 client_max_body_size。";
    if (err.status === 415) hint = "只支持 PDF / DOCX / PNG / JPG。";
    if (err.status === 422) hint = "文件是空的，或表单里没有 file 字段。";
    showError(errorEl, "上传失败：" + err.message, hint);
    setStatus(statusEl, "");
  } finally {
    uploadBtn.disabled = false;
  }
});

refreshBtn.addEventListener("click", function () {
  pollUntil = 0;
  load();
});

async function load() {
  try {
    const data = await fetchJSON("/api/documents");
    renderTable(data.documents || []);
    countEl.textContent = "共 " + (data.total || 0) + " 份";
    clearError(errorEl);
  } catch (err) {
    listEl.textContent = "";
    const box = el("div", "empty", "读取文档列表失败：" + err.message);
    listEl.appendChild(box);
    countEl.textContent = "";
    showError(errorEl, "读取文档列表失败：" + err.message,
      "确认后端在跑；数据库文件路径见 .env 的 DATABASE_URL。");
    return;
  }
  schedulePoll();
}

function schedulePoll() {
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
  const running = Array.prototype.some.call(
    listEl.querySelectorAll("[data-status]"),
    function (node) {
      const value = node.getAttribute("data-status");
      return value === "pending" || value === "parsing";
    }
  );
  const waitingUpload = Date.now() < pollUntil;
  if (!running && !waitingUpload) return;
  timer = setTimeout(load, POLL_MS);
}

function renderTable(documents) {
  listEl.textContent = "";

  if (!documents.length) {
    pollUntil = 0;
    listEl.appendChild(
      el("div", "empty", "还没有文档。上传一份手册（PDF / DOCX / 图片），系统会自动解析、切块并建索引。")
    );
    return;
  }

  const table = el("table");
  const thead = el("thead");
  const headRow = el("tr");
  ["文件", "状态", "页数", "块数", "操作"].forEach(function (title) {
    headRow.appendChild(el("th", null, title));
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  documents.forEach(function (doc) {
    const row = el("tr");

    const nameCell = el("td");
    nameCell.appendChild(el("div", "file", doc.filename));
    if (doc.created_at) {
      nameCell.appendChild(el("div", "hint", doc.created_at.replace("T", " ").slice(0, 16)));
    }
    row.appendChild(nameCell);

    // data-label 供窄屏的卡片式布局做字段前缀，桌面宽屏不显示
    const statusCell = el("td");
    statusCell.setAttribute("data-label", "状态");
    const badge = el("span", STATUS_CLASS[doc.status] || "badge", STATUS_TEXT[doc.status] || doc.status);
    badge.setAttribute("data-status", doc.status);
    statusCell.appendChild(badge);
    if (doc.status === "failed" && doc.failed_reason) {
      statusCell.appendChild(el("div", "reason", doc.failed_reason));
    }
    row.appendChild(statusCell);

    const pagesCell = el("td", null, doc.pages === null || doc.pages === undefined ? "—" : String(doc.pages));
    pagesCell.setAttribute("data-label", "页数");
    if (doc.ocr_pages) {
      pagesCell.appendChild(el("div", "hint", "OCR " + doc.ocr_pages + " 页"));
    }
    row.appendChild(pagesCell);

    const chunkCell = el("td", null, String(doc.chunks || 0));
    chunkCell.setAttribute("data-label", "块数");
    if (doc.chunks && doc.chunks_vectorized < doc.chunks) {
      chunkCell.appendChild(el("div", "hint", "已向量化 " + doc.chunks_vectorized));
    }
    row.appendChild(chunkCell);

    const actions = el("div", "actions");
    const reingest = el("button", "btn btn-small", "重新解析");
    reingest.type = "button";
    reingest.addEventListener("click", function () {
      act(reingest, "/api/documents/" + doc.id + "/ingest", "POST", "已排入解析队列：" + doc.filename);
    });
    const remove = el("button", "btn btn-small", "删除");
    remove.type = "button";
    remove.addEventListener("click", function () {
      if (!window.confirm("删除「" + doc.filename + "」？分块与磁盘文件会一起删除，不可撤销。")) return;
      act(remove, "/api/documents/" + doc.id, "DELETE", "已删除：" + doc.filename);
    });
    actions.appendChild(reingest);
    actions.appendChild(remove);

    const actionCell = el("td");
    actionCell.appendChild(actions);
    row.appendChild(actionCell);

    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  listEl.appendChild(table);
}

async function act(button, url, method, okMessage) {
  button.disabled = true;
  clearError(errorEl);
  try {
    await fetchJSON(url, { method: method });
    setStatus(statusEl, okMessage);
    if (method === "POST") pollUntil = Date.now() + 120000;
    await load();
  } catch (err) {
    showError(errorEl, "操作失败：" + err.message,
      err.status === 404 ? "文档可能已被删除，点「刷新列表」看看。" : "");
  } finally {
    button.disabled = false;
  }
}

load();
