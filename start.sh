#!/usr/bin/env bash
# Linux / macOS 的双击启动脚本；Windows 用同目录的 启动.bat
set -u
cd "$(dirname "$0")"

PY=".venv/bin/python"
URL="http://127.0.0.1:8000/"

if [ ! -x "$PY" ]; then
  echo "[错误] 没有找到 $PY"
  echo "请先执行："
  echo "    python -m venv .venv"
  echo "    .venv/bin/pip install -r requirements.txt"
  exit 1
fi

open_browser() {
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  elif command -v open >/dev/null 2>&1; then open "$URL"
  else echo "请手动打开 $URL"; fi
}

# 已在运行就直接开浏览器
if curl -sf -o /dev/null "$URL"health 2>/dev/null; then
  echo "服务已经在运行，直接打开页面…"
  open_browser
  exit 0
fi

echo "正在启动服务（Ctrl+C 停止）…"
"$PY" -m uvicorn app.main:app --port 8000 &
SERVER_PID=$!

for _ in $(seq 1 30); do
  if curl -sf -o /dev/null "$URL"health 2>/dev/null; then
    open_browser
    echo "已打开 $URL（Ctrl+C 停止服务）"
    wait "$SERVER_PID"
    exit 0
  fi
  sleep 1
done

echo "[超时] 30 秒内服务没有起来，请看上面的报错。"
kill "$SERVER_PID" 2>/dev/null
exit 1
