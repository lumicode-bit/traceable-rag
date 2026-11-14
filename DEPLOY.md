# 部署手册：从一台空 VPS 到手机能访问

按顺序照做即可。全程只需要 **2 个 API key**，不需要数据库服务、不需要 GPU、不需要装 torch。

预计耗时：首次 30~40 分钟（其中镜像构建 5~15 分钟，取决于网速——依赖约 420MB，不含 torch；
OCR 首次识别还会下载约 80MB 的检测/识别模型）。

---

## 0. 开工前你需要有三样东西

| 东西 | 说明 |
| --- | --- |
| 一台 VPS | 1 核 1G 能跑，**建议 2 核 2G**（OCR 是 CPU 密集）。系统 Ubuntu 22.04 / 24.04 |
| 一个域名 | 加一条 A 记录指向 VPS 公网 IP，例如 `kb.example.com → 1.2.3.4` |
| 两个 key | 对话（DeepSeek 等）+ 向量（硅基流动 / 智谱 / 阿里百炼，`BAAI/bge-m3`） |

没有域名也能跑：跳过第 6 步，直接用 `http://服务器IP` 访问。但没有 HTTPS 时，**手机上流式回答可能被运营商劫持/缓存**，演示前建议补上。

下面的命令都在 VPS 上以有 sudo 权限的用户执行。

---

## 1. 装 Docker（约 3 分钟）

```bash
# 官方一键脚本；国内网络慢可以先用阿里云镜像源，见本节末
curl -fsSL https://get.docker.com | sudo sh

# 免 sudo 使用 docker（重新登录后生效）
sudo usermod -aG docker $USER
newgrp docker

docker --version
docker compose version        # 需要 v2（compose 插件），v1 不行
```

1G 内存的机器务必加 swap，否则构建镜像时可能被 OOM 杀掉：

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

---

## 2. 取代码

```bash
cd /opt
sudo git clone <你的仓库地址> kb && sudo chown -R $USER:$USER /opt/kb
cd /opt/kb
ls        # 应能看到 docker-compose.yml、DEPLOY.md、app/、web/
```

没有 git 仓库就把项目打包上传后解压到 `/opt/kb`。

---

## 3. 配 `.env`

```bash
cd /opt/kb
cp .env.example .env
chmod 600 .env                       # 里面是 key，别让同机其它账号读到
nano .env
```

至少要改这四段：

```ini
MOCK_MODE=0

# 对话
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-你的对话key
LLM_MODEL=deepseek-flash
LLM_MAX_TOKENS=4000
LLM_EXTRA_BODY={"thinking": {"type": "disabled"}}

# 向量
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_API_KEY=sk-你的向量key
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024
```

数据库不用配，默认 `sqlite:////app/data/kb.db`（容器内路径，映射到宿主机的 `./data/kb.db`）。

> `EMBED_DIM` 必须和接口实际返回的维度一致（bge-m3 是 1024）。填错时后端会直接报
> 「接口返回 N 维，EMBED_DIM 配置为 M」，这比悄悄算错要安全得多。拿不准就填 `0`，表示以接口为准。

改完确认 `.env` 没被提交：`git status --short` 不应该出现 `.env`。

---

## 4. 启动服务

```bash
cd /opt/kb
docker compose up -d --build          # 首次构建 5~10 分钟

docker compose ps                     # app 与 nginx 都应是 Up
docker compose logs --tail=50 app     # 不应出现 Traceback
```

自检：

```bash
curl -s localhost/health              # 期望 {"db":"ok"}
curl -sI localhost | head -1          # 期望 HTTP/1.1 200 OK

# 从公网验证（把域名换成你的）
curl -s https://kb.example.com/health
```

`docker compose ps` 里 app 的端口应显示 `127.0.0.1:8000->8000`——这是故意的：**8000 只绑本机，公网只能走 nginx**。

---

## 5. 把手册灌进去

两种方式，选一种。

**A. 页面上传（推荐给演示）**

浏览器打开站点 → 「文档」页 → 选 PDF → 上传。状态会自己从「解析中」变成「已入库」。
一份 160 页的手册（含少量扫描页）约 1~2 分钟，解析在后台跑，**不要关页面等结果也可以关掉，回头刷新即可**。

**B. 命令行批量入库（适合一次导入很多）**

```bash
# 先把文件传到服务器
scp ./CP600*.pdf user@1.2.3.4:/opt/kb/data/raw/

# 解析 + 切块
docker compose exec app python -m app.ingest.pipeline data/raw

# 补向量与 BM25 索引（必须做，否则检索里没有这些文档）
docker compose exec app python -m app.retrieval.bench --rebuild
```

看到 `库内累计：N 份文档 / M 块` 就成功了。

> 首次 OCR 会在容器内下载检测/识别模型（约 80MB）。若卡住，检查服务器能否访问外网；
> 扫描件很少时也可以先跳过（只影响扫描页的识别质量）。

---

## 6. 配 HTTPS（有域名就做）

**6.1 先确认 80 端口通**

```bash
curl -I http://kb.example.com/health     # 能从外网访问才继续
```

云厂商的安全组要放行 **80 与 443**。

**6.2 签发证书**

```bash
cd /opt/kb
docker compose --profile certbot run --rm certbot certonly \
  --webroot -w /var/www/certbot \
  -d kb.example.com \
  --email you@example.com --agree-tos --no-eff-email
```

成功后会提示证书路径 `deploy/certbot/conf/live/kb.example.com/`。

**6.3 切到 HTTPS 配置**

```bash
cd /opt/kb
# 把 nginx.https.conf 里的 kb.example.com 换成你的域名（两处）
sed -i 's/kb\.example\.com/你的真实域名/g' deploy/nginx.https.conf

# 切配置：备份 HTTP 版，换成 HTTPS 版
cp deploy/nginx.conf deploy/nginx.http.conf.bak
cp deploy/nginx.https.conf deploy/nginx.conf

docker compose restart nginx
docker compose logs --tail=20 nginx     # 不应出现 "cannot load certificate"
```

**6.4 验证**

```bash
curl -sI https://kb.example.com | head -1          # 200
curl -sI http://kb.example.com | head -1           # 301，跳 HTTPS
```

**6.5 自动续期**（证书 90 天到期，务必配）

```bash
crontab -e
# 加入两行：
# 30 3 * * * cd /opt/kb && docker compose --profile certbot run --rm certbot renew --quiet
# 40 3 * * * cd /opt/kb && docker compose exec -T nginx nginx -s reload
```

---

## 7. 验收（演示前自己走一遍）

| 检查项 | 怎么做 | 期望 |
| --- | --- | --- |
| 手机能提问 | 手机浏览器打开站点，输入「上电无显示怎么排查」 | 文字一个个蹦出来，不是等十秒后整段出现 |
| 引用可点开 | 回答下方点「查看原文」 | 展开该块的完整正文与页码 |
| 模式切换 | 下拉换成「向量（语义）」再问一次 | 结果可能不同，但引用依然全部可回溯 |
| 后端挂掉 | `docker compose stop app`，然后在页面提问 | 页面显示「无法连接后端（网络中断或服务未启动）」，**不是一直转圈** |
| 断网 | 手机开飞行模式后点「提问」 | 同上，一条明确的错误提示 |
| 上传异常 | 上传一个 `.txt` | 提示「不支持的文件类型：.txt（允许 …）」 |
| 文档列表 | 「文档」页看状态 | 状态标签、页数、块数都对；失败时能看到失败原因 |

验收完记得 `docker compose start app`。

---

## 8. 日常运维

```bash
# 更新代码
cd /opt/kb && git pull && docker compose up -d --build

# 看日志
docker compose logs -f --tail=100 app nginx

# 备份（库是单文件，拷目录即可）
sudo tar czf /root/kb-backup-$(date +%F).tar.gz -C /opt/kb data results

# 重启单个服务
docker compose restart app
```

**磁盘**：`data/` 里是原始 PDF + 一个 SQLite 文件；库里存了块正文与向量，长期占用约「语料大小的 1.5 倍」。低于 2GB 剩余空间时先清理再入库。

---

## 9. 常见故障对照

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 上传大文件报 `413` | nginx 的 `client_max_body_size` 比 `MAX_UPLOAD_MB` 小 | 改 `deploy/nginx.conf` 里的 `client_max_body_size 50m;` 并 `docker compose restart nginx` |
| 回答不是流式，等很久才整段出现 | nginx 没关缓冲 | 确认 `location /api/chat` 里有 `proxy_buffering off;`，且没被别的 `location` 抢先匹配 |
| 页面 502 | app 没起来或崩了 | `docker compose logs app` 看 Traceback；常见是 `.env` 写错、`EMBED_DIM` 与接口不符 |
| 页面显示「后端未配置大模型」 | `LLM_*` 没填或填错 | 改 `.env` 后 `docker compose up -d`（改 env 必须重建容器，`restart` 不生效） |
| 上传后一直「解析中」 | 解析在后台跑，长文档 + OCR 需要时间 | 看 `docker compose logs app`；超过 10 分钟无进展，看是否 OOM（`dmesg | tail`） |
| 搜不到刚传的文档 | 只入库没建索引 | `docker compose exec app python -m app.retrieval.bench --rebuild` |
| 文档页显示「失败」 | 见状态下方的失败原因 | 常见：扫描件缺 OCR 依赖、PDF 加密、文件损坏 |
| 改了 `.env` 没效果 | 环境变量在容器启动时读取 | `docker compose up -d --force-recreate app` |

如果 `app` 反复重启，先看这一条：

```bash
docker compose logs --tail=100 app | grep -i -E "error|traceback" -A5
```

---

## 10. 安全说明（为什么没有登录）

按项目约定，**不做登录注册**。这意味着任何拿到域名的人都能提问和看文档列表。为此做了三件补偿：

1. **8000 只绑本机**（`127.0.0.1:8000`），公网只能经 nginx 进来；
2. **key 只在后端**，前端拿不到任何凭据（页面里搜不到 `sk-`）；
3. **只开放 80/443**，其余端口由云安全组关闭。

另外提醒两条：

- `.env` 权限 `600`，且已在 `.gitignore` 里。**每年都有人把 key 提交上去被人跑光额度**。
- 如果演示前担心被扫，可以在 nginx 里加一段 `allow 你的IP; deny all;`，演示结束再去掉。

---

## 11. 回滚

```bash
cd /opt/kb

# 回到 HTTP（证书出问题时的应急手段）
cp deploy/nginx.http.conf.bak deploy/nginx.conf && docker compose restart nginx

# 完全停掉
docker compose down            # 数据在 ./data，不会丢
```

`docker compose down -v` 会删除卷，本项目数据都在宿主机的 `./data` 目录里，因此不加 `-v` 也不会丢库。
