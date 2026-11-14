FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# onnxruntime（RapidOCR 后端）在 slim 镜像里依赖 OpenMP 运行时
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先单独复制依赖清单，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 60 --retries 5 -r requirements.txt

# 只复制运行与测试所需的代码；data/ 由 docker-compose 以卷挂载到 /app/data
COPY app ./app
COPY web ./web
COPY tests ./tests
COPY pytest.ini ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
