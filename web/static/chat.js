/* 问答页：POST /api/chat 拿 SSE 流，边收边显示；引用卡片按需拉完整原文。 */
"use strict";

const form = document.getElementById("qa-form");
const questionEl = document.getElementById("question");
const modeEl = document.getElementById("mode");
const askBtn = document.getElementById("ask");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const emptyEl = document.getElementById("empty");
const resultEl = document.getElementById("result");
const answerEl = document.getElementById("answer");
const citesEl = document.getElementById("cites");
const metaEl = document.getElementById("meta");

let controller = null;

form.addEventListener("submit", function (event) {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) {
    showError(errorEl, "请先输入问题。");
    questionEl.focus();
    return;
  }
  ask(question, modeEl.value);
});

// Ctrl/Cmd + Enter 直接提交，手机上不会误触
questionEl.addEventListener("keydown", function (event) {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    form.requestSubmit();
  }
});

async function ask(question, mode) {
  if (controller) controller.abort();
  controller = new AbortController();
  const signal = controller.signal;
  // 90 秒还没收完就主动断开：否则后端卡住时页面会一直转
  const timer = setTimeout(function () { controller.abort(); }, 90000);

  clearError(errorEl);
  emptyEl.hidden = true;
  resultEl.hidden = false;
  answerEl.textContent = "";
  citesEl.textContent = "";
  metaEl.textContent = "";
  askBtn.disabled = true;
  setStatus(statusEl, "正在检索片段…");

  let gotDelta = false;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, mode: mode, top_k: window.TOP_K }),
      signal: signal,
    });

    if (!response.ok) {
      let detail = "HTTP " + response.status;
      try {
        const data = await response.json();
        if (data && typeof data.detail === "string") detail = data.detail;
      } catch (err) { /* 响应体不是 JSON 就用状态码 */ }
      throw new ApiError(response.status, detail);
    }
    if (!response.body) {
      throw new ApiError(0, "当前浏览器不支持流式读取，请换新版 Chrome / Edge / Safari。");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });

      let index = buffer.indexOf("\n\n");
      while (index !== -1) {
        const frame = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        const kind = handleFrame(frame);
        if (kind === "delta" && !gotDelta) {
          gotDelta = true;
          setStatus(statusEl, "正在生成…");
        }
        index = buffer.indexOf("\n\n");
      }
    }
    setStatus(statusEl, "");
  } catch (err) {
    if (err && err.name === "AbortError") {
      showError(errorEl, "请求超时或已取消（90 秒）。",
        "后端可能卡在模型调用上；稍后重试，或先看后端日志 uvicorn / docker compose logs app。");
    } else if (err instanceof ApiError) {
      // 后端明确返回了状态码与 detail
      showError(errorEl, err.message,
        err.status === 0 ? "确认服务在跑：docker compose ps，或本机 uvicorn app.main:app。" : "");
    } else {
      // fetch 直接 reject（TypeError: Failed to fetch）或流中途断开：
      // 这两种都是「连不上」，必须翻成人话，不能把浏览器原文糊到页面上
      showError(errorEl, "无法连接后端（网络中断或服务未启动）。",
        "检查手机网络；服务端确认：docker compose ps，日志 docker compose logs --tail=50 app。");
    }
    setStatus(statusEl, "");
    // 一个字都没收到时不要留个空框在那，错误提示自己就是结论
    if (!answerEl.textContent.trim()) {
      resultEl.hidden = true;
      emptyEl.hidden = true;
    }
  } finally {
    clearTimeout(timer);
    askBtn.disabled = false;
  }
}

/** 处理一帧 SSE，返回事件名。 */
function handleFrame(frame) {
  let event = "message";
  const dataLines = [];
  frame.split("\n").forEach(function (line) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  });
  if (!dataLines.length) return event;

  let payload;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch (err) {
    return event;
  }

  if (event === "meta") {
    showMeta(payload);
  } else if (event === "delta") {
    answerEl.textContent += payload.text || "";
  } else if (event === "citations") {
    renderCitations(payload.citations || []);
    showCheck(payload.citation_check);
    setStatus(statusEl, "");
    finalize(payload);
  } else if (event === "error") {
    setStatus(statusEl, "");
    showError(errorEl, "后端返回错误：" + (payload.message || payload.code || "未知错误"),
      payload.code === "config_missing"
        ? "后端没有配置大模型：在 .env 填 LLM_* 三项后重启服务。"
        : "错误码 " + payload.code + "，可重试 " + (payload.retryable ? "是" : "否") + "。");
  }
  return event;
}

function showMeta(meta) {
  const parts = [];
  if (meta.retrieval_ms !== undefined) parts.push("检索 " + Math.round(meta.retrieval_ms) + "ms");
  if (meta.passage_count !== undefined) parts.push("片段 " + meta.passage_count + " 条");
  if (meta.mode) parts.push("模式 " + meta.mode);
  if (meta.model) parts.push("模型 " + meta.model);
  metaEl.textContent = parts.join(" · ");

  if (meta.llm_configured === false) {
    showError(errorEl, "后端未配置大模型，检索可以做，生成做不了。",
      "在 .env 填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后重启；或设 MOCK_MODE=1 用桩模型演示。");
  }
}

function finalize(payload) {
  const timings = payload.timings || {};
  if (timings.total_ms) {
    metaEl.textContent += " · 生成 " + Math.round(timings.generation_ms || 0) + "ms";
  }
  if (payload.usage && payload.usage.total_tokens) {
    metaEl.textContent += " · " + payload.usage.total_tokens + " tokens";
  }
  if (payload.answer_status === "refused") {
    const note = el("p", "hint", "这条回答是「资料里没有」，不是故障不存在——按上面的「缺少的信息」补充资料后再问。");
    answerEl.appendChild(note);
  }
  if (!answerEl.textContent.trim()) {
    answerEl.textContent = "（模型没有返回任何正文，可能是输出长度被截断）";
  }
}

function showCheck(check) {
  if (!check || check.unique === undefined) return;
  if (check.unique === 0) return;
  const text = "引用校验：" + check.unique + " 个来源，全部可回溯" +
    (check.invalid && check.invalid.length ? "；其中 " + check.invalid.length + " 个不在本次片段内" : "");
  metaEl.textContent += " · " + text;
}

function renderCitations(list) {
  citesEl.textContent = "";
  if (!list.length) {
    citesEl.appendChild(el("p", "hint", "本次没有检索到任何片段，所以回答只能是「资料中未找到」。"));
    return;
  }
  list.forEach(function (item, index) {
    const card = el("article", "cite");
    const head = el("div", "cite-head",
      "[" + (index + 1) + "] " + item.doc + "　第 " + (item.page === null ? "—" : item.page) + " 页");
    if (item.section) {
      head.appendChild(el("div", "hint", item.section));
    }
    card.appendChild(head);

    const snippet = el("p", "cite-snippet", item.snippet || "（无摘要）");
    card.appendChild(snippet);

    const toggle = el("button", "link", "查看原文");
    toggle.type = "button";
    const full = el("pre", "cite-full");
    full.hidden = true;
    let loaded = false;

    toggle.addEventListener("click", async function () {
      if (loaded) {
        full.hidden = !full.hidden;
        toggle.textContent = full.hidden ? "查看原文" : "收起原文";
        return;
      }
      toggle.disabled = true;
      toggle.textContent = "读取中…";
      try {
        const data = await fetchJSON("/api/chunks/" + item.chunk_id);
        full.textContent = data.content || "（正文为空）";
        full.hidden = false;
        toggle.textContent = "收起原文";
        loaded = true;
      } catch (err) {
        card.appendChild(el("p", "cite-error", "读取原文失败：" + err.message));
        toggle.textContent = "重试";
      } finally {
        toggle.disabled = false;
      }
    });

    card.appendChild(toggle);
    card.appendChild(full);
    citesEl.appendChild(card);
  });
}
