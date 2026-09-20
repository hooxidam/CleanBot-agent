#!/bin/sh
# 容器入口脚本：统一处理"运行时状态"的持久化，然后启动服务。
#
# 背景：代码里的路径是硬编码在项目根目录下的（utils/path_tool.get_abs_path）：
#   /app/app.sqlite    —— LangGraph checkpoint + 会话元数据 + 用户档案
#   /app/chroma_db/    —— Chroma 向量库
#   /app/md5.text      —— 知识库增量索引的 MD5 记录（丢了会重复入库）
#
# 而 Docker 的数据卷只能挂在已存在的目录上；若把命名卷直接挂到 /app/app.sqlite
# 这种"文件路径"上，Docker 会把它建成目录，SQLite 将无法打开。
# 所以这里把这三项软链到数据卷 /state，既持久化，又不需要改动任何业务代码。
set -e

mkdir -p /state/chroma_db
[ -e /state/app.sqlite ] || : > /state/app.sqlite
[ -e /state/md5.text ] || : > /state/md5.text

ln -sfn /state/app.sqlite /app/app.sqlite
ln -sfn /state/chroma_db /app/chroma_db
ln -sfn /state/md5.text /app/md5.text

# 传了命令就执行该命令，不启动服务。
# 用于一次性任务，例如首次在容器内构建知识库：
#   docker compose run --rm backend python -m rag.vector_store
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

# --host 0.0.0.0 是必须的：容器内的 127.0.0.1 只有容器自己可访问，
# 端口映射到宿主机后必须监听全部网卡，否则浏览器连不上。
# 不加 --reload：容器里没有热重载的必要，也避免文件监视占用资源。
exec uvicorn api.server:app --host 0.0.0.0 --port "${PORT:-8000}"