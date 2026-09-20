# 与本地开发环境保持一致（.venv/pyvenv.cfg: 3.12.0）
FROM python:3.12-slim

# PYTHONUNBUFFERED：日志与 SSE 数据块不被缓冲，实时输出
# PYTHONDONTWRITEBYTECODE：不在镜像里生成 __pycache__
# HOME：指定模型缓存位置，便于把缓存挂成数据卷
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app

WORKDIR /app

# torch（sentence-transformers 的依赖，用于嵌入与 CrossEncoder 重排序）
# 依赖 OpenMP 运行库，slim 基础镜像里没有
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依赖层：只要 requirements.txt 没变，重建镜像时会复用这一层缓存。
# 顺序很关键——若先 COPY 全部代码再装依赖，改一行代码就要重装所有依赖。
COPY requirements.txt .

# 预装 CPU 版 torch：PyPI 上 Linux 的 torch 默认捆绑 CUDA 依赖（约 2.5GB），
# 而本项目只在 CPU 上做嵌入与重排序。装 CPU 版可把镜像从数 GB 降到几百 MB，
# 也让构建在较慢的网络下更容易成功。
# 若该源在你的网络下不可达，删掉这一行即可（代价是镜像体积变大）。
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# 代码层：改代码只会让这一层失效，不会触发依赖重装。
# .dockerignore 已排除 .venv / app.sqlite / chroma_db / .env 等无关或敏感内容。
COPY . .

# 建立非 root 运行用户，并提前建好数据卷挂载点。
# 挂载点必须先于数据卷存在且属主正确，命名卷首次创建时才会继承这个属主，
# 否则非 root 用户会因为没有写权限而启动失败。
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /state /home/app/.cache \
    && chown -R app:app /state /home/app /app \
    && chmod +x /app/docker/entrypoint.sh

USER app

EXPOSE 8000

# 用入口脚本统一处理数据卷软链，再启动服务（见 docker/entrypoint.sh）
ENTRYPOINT ["/app/docker/entrypoint.sh"]