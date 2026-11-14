/* 三个页面共用的小工具：把「取数据 + 出错怎么说人话」放在一处。 */
"use strict";

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** 统一的请求封装：网络层失败与后端错误都转成带 status 的 ApiError。 */
async function fetchJSON(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    // fetch 直接 reject 只发生在网络层：断网、DNS、服务没起、被 CORS 挡掉
    throw new ApiError(0, "无法连接后端（网络中断或服务未启动）");
  }

  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    data = null;
  }

  if (!response.ok) {
    // FastAPI 的 detail 可能是字符串，也可能是 [{loc, msg}] 数组
    let detail = data && (data.detail || data.message);
    if (Array.isArray(detail)) {
      detail = detail.map((item) => item.msg || JSON.stringify(item)).join("；");
    }
    throw new ApiError(response.status, detail || "HTTP " + response.status);
  }
  return data;
}

/** 状态行：传空字符串即隐藏。 */
function setStatus(el, text) {
  if (!el) return;
  el.textContent = text || "";
  el.hidden = !text;
}

/** 错误框：正文 + 可选的一行排查提示。 */
function showError(el, message, hint) {
  if (!el) return;
  el.textContent = "";
  const body = document.createElement("p");
  body.className = "reason";
  body.textContent = message;
  el.appendChild(body);
  if (hint) {
    const extra = document.createElement("p");
    extra.className = "hint";
    extra.textContent = hint;
    el.appendChild(extra);
  }
  el.hidden = false;
}

function clearError(el) {
  if (!el) return;
  el.textContent = "";
  el.hidden = true;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}
