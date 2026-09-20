"""
会话元数据存储。

职责划分要分清：
  - LangGraph 的 checkpointer 负责存「某个 thread_id 下的消息历史」；
  - 但它不知道"这个会话属于谁""会话叫什么名字""什么时候建的"。
本模块就补这一层：一张 sessions 表，把 会话 ↔ 用户 关联起来，供前端渲染会话列表。

表和 checkpointer 共用同一个 sqlite 文件，便于一起备份/清理。
"""
import sqlite3
import uuid
from datetime import datetime

from utils.logger_handler import logger

DEFAULT_TITLE = "新对话"
TITLE_MAX_LEN = 20


class SessionStore:
    def __init__(self, db_path: str):
        #check_same_thread=False：FastAPI 同步接口跑在线程池，连接会跨线程复用
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            #按用户查会话列表、按更新时间倒序 —— 加索引避免全表扫
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON sessions(user_id, updated_at DESC)
            """)

    def create_session(self, user_id: str, title: str = DEFAULT_TITLE) -> dict:
        """新建一场对话，session_id 同时也是 checkpointer 的 thread_id"""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now),
            )
        logger.info(f"[session_store]用户 {user_id} 新建会话 {session_id}")
        return {"session_id": session_id, "user_id": user_id, "title": title,
                "created_at": now, "updated_at": now}

    def list_sessions(self, user_id: str) -> list[dict]:
        """列出某个用户的所有会话，最近使用的排最前"""
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def touch(self, session_id: str, first_query: str | None = None):
        """对话发生后更新时间戳；若还是默认标题，用首条提问自动命名(类似 ChatGPT)"""
        now = datetime.now().isoformat(timespec="seconds")
        session = self.get_session(session_id)
        if not session:
            return

        title = session["title"]
        if title == DEFAULT_TITLE and first_query:
            title = first_query.strip()[:TITLE_MAX_LEN]

        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET updated_at = ?, title = ? WHERE session_id = ?",
                (now, title, session_id),
            )

    def rename(self, session_id: str, title: str):
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (title.strip()[:TITLE_MAX_LEN] or DEFAULT_TITLE, session_id),
            )

    def delete(self, session_id: str):
        """只删会话元数据；对应的消息历史由调用方通过 checkpointer.delete_thread 删除"""
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        logger.info(f"[session_store]删除会话 {session_id}")
