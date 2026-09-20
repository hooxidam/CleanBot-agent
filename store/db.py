"""
应用自身的持久化存储（SQLite）。

存在这里的是**应用运行时产生的数据**，与 data/external/ 下模拟厂家侧的只读数据区分开：
  - checkpoints / writes ：LangGraph 存的会话消息历史
  - sessions             ：会话元数据（归属用户、标题、时间）
  - user_profiles        ：服务过程中收集到的用户档案（用户主动告知的信息）

真实部署时，user_profiles 应写回厂家的用户资料服务，本文件只是它的本地替身。
"""
import sqlite3

from utils.path_tool import get_abs_path

APP_DB_PATH = get_abs_path("app.sqlite")


def connect() -> sqlite3.Connection:
    """建立数据库连接。

    check_same_thread=False：FastAPI 的同步接口跑在线程池里，
    同一个连接会被不同线程复用，不关掉这个检查会报跨线程错误。
    """
    conn = sqlite3.connect(APP_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
