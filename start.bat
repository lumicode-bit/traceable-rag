@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 工业设备运维知识库 - 启动器

set "PY=.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8000/"

rem ---- 1. 检查虚拟环境（系统 Python 3.13 装不上 RapidOCR，必须用 .venv） ----
if not exist "%PY%" (
  echo.
  echo [错误] 没有找到 .venv\Scripts\python.exe
  echo.
  echo 请先在当前目录执行一次：
  echo     python -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

rem ---- 2. 端口已被占用：说明服务已经在跑，直接开浏览器 ----
netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo 服务已经在运行，直接打开页面…
  start "" "%URL%"
  timeout /t 2 >nul
  exit /b 0
)

rem ---- 3. 新开一个服务窗口（关掉那个窗口就等于停止服务） ----
echo 正在启动服务，首次启动约 3~5 秒…
start "知识库服务（关闭此窗口=停止服务）" cmd /k ""%PY%" -m uvicorn app.main:app --port 8000"

rem ---- 4. 等 /health 返回，最多等 30 秒 ----
for /l %%i in (1,1,30) do (
  curl -sf -o nul "%URL%health" >nul 2>&1
  if not errorlevel 1 goto ready
  timeout /t 1 >nul
)

echo.
echo [超时] 30 秒内服务没有起来，请查看弹出的服务窗口里的报错。
echo        常见原因：.env 配置有误、端口被其它程序占用。
pause
exit /b 1

:ready
start "" "%URL%"
echo.
echo 已在浏览器打开 %URL%
echo Doc: http://127.0.0.1:8000/  ｜ 文档: /documents ｜ 评测: /eval
echo 停止服务：关闭标题为「知识库服务」的那个窗口。
timeout /t 3 >nul
exit /b 0
