"""
FastAPI 后端服务：把命令行里的 ReactAgent 包装成一个「网络接口」，
并用 SSE(Server-Sent Events) 把大模型的回复「逐字流式」推给前端。

启动方式(在项目根目录执行)：
    uvicorn api.server:app --reload --port 8000

接口一览：
    GET    /                              前端聊天页面
    GET    /health                        健康检查(不调用大模型，不花钱)
    GET    /users                         可选择的用户列表(来自 records.csv)
    GET    /sessions?user_id=             某用户的会话列表
    POST   /sessions                      新建会话
    DELETE /sessions/{session_id}         删除会话(元数据 + 消息历史)
    GET    /sessions/{session_id}/messages 会话的历史消息(切换会话时回显)
    GET    /chat/stream?query=&session_id=&user_id=  SSE 流式问答
"""
import json
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.react_agent import ReactAgent
from agent.guard import AgentGuardError
from store.db import APP_DB_PATH
from store.session_store import SessionStore
from store.user_store import list_users, user_exists
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

app = FastAPI(title="扫地机器人智能客服 Agent API")

# 允许跨域：前端网页(比如 http://localhost:5173)才能调用本接口。
# 生产环境应把 allow_origins 收窄到你自己的前端域名，这里为方便开发先放开。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent 构建较重(要连向量库、初始化模型)，所以只在服务启动时构建一次，全局复用。
# 前端仍由当前 FastAPI 服务提供；这里只挂载拆分后的 CSS/JavaScript 静态资源，
# 不改变任何业务接口或 SSE 数据结构。
app.mount("/assets", StaticFiles(directory=get_abs_path("web")), name="web-assets")

agent = ReactAgent()
# 会话元数据(归属/标题/时间)，与 checkpointer 共用同一个 sqlite 文件
session_store = SessionStore(APP_DB_PATH)


class CreateSessionBody(BaseModel):
    user_id: str


def _sse_pack(data: dict) -> str:
    """把一个字典打包成一条 SSE 消息。
    SSE 规定每条消息形如：  data: <内容>\\n\\n
    这里用 JSON 承载内容，ensure_ascii=False 让中文保持可读、且 JSON 会把换行转义成 \\n，
    不会破坏 SSE 的分帧。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    """根路径：返回前端聊天页面 web/index.html。
    这样启动服务后直接访问 http://127.0.0.1:8000/ 就是聊天界面。"""
    return FileResponse(get_abs_path("web/index.html"))


@app.get("/health")
def health():
    """健康检查：确认服务活着。不触发任何大模型调用。"""
    return {"status": "ok"}


@app.get("/users")
def get_users():
    """可选择的用户列表。本项目不做注册/密码，直接用 records.csv 里已有的用户身份。"""
    return {"users": list_users()}


@app.get("/sessions")
def get_sessions(user_id: str):
    """某个用户的会话列表，最近使用的排最前"""
    return {"sessions": session_store.list_sessions(user_id)}


@app.post("/sessions")
def create_session(body: CreateSessionBody):
    """新建一场对话"""
    if not user_exists(body.user_id):
        raise HTTPException(status_code=404, detail=f"用户 {body.user_id} 不存在")
    return session_store.create_session(body.user_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """删除会话：既要删会话元数据，也要删对应的消息历史，避免留下孤儿数据"""
    if not session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    session_store.delete(session_id)
    agent.delete_history(session_id)
    return {"deleted": session_id}


@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """会话的历史消息，前端切换会话时用它回显之前聊过的内容"""
    if not session_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": agent.get_history(session_id)}


@app.get("/chat/stream")
def chat_stream(query: str, session_id: str, user_id: str):
    """SSE 流式问答接口。
    前端用 EventSource 连上后，服务器会把模型生成的文本一小段一小段地推回来。
    :param query: 用户的问题(URL 查询参数)
    :param session_id: 会话 ID。同一个 session_id 的多次提问属于同一场对话，
                       Agent 会记得之前聊过的内容(多轮记忆)
    :param user_id: 当前用户，注入 Agent 运行时上下文，供各查询工具确定"查谁的数据"。
                    工具签名里不暴露 user_id，模型无权指定，因此无法越权读取他人数据
    """

    def event_generator():
        request_id = uuid4().hex[:12]
        logger.info(f"[api]用户 {user_id} 在会话 {session_id} 提问：{query}")
        try:
            # Agent 产出的事件已经是结构化的(token / tool_start / tool_end)，
            # 这里原样转发即可，前端据此渲染"吐字"和"工具调用步骤"
            for event in agent.stream_events(
                query, session_id, user_id, request_id=request_id
            ):
                yield _sse_pack(event)
            # 对话成功后更新会话时间戳；首次提问时顺便用问题内容自动命名会话
            session_store.touch(session_id, first_query=query)
            # 结束标记：前端收到 done 就知道本次回答结束，可以关闭连接
            yield _sse_pack({"type": "done"})
            logger.info("[api]本次回答结束")
        except AgentGuardError as e:
            # 后端记录完整堆栈；前端只接收稳定错误码和友好提示。
            logger.error(
                "[AgentGuard] request stopped request_id=%s session_id=%s "
                "user_id=%s tool_call_count=%s exception_type=%s",
                e.request_id, e.session_id, e.user_id, e.tool_call_count,
                type(e).__name__, exc_info=True,
            )
            yield _sse_pack(e.to_sse_event())
        except Exception as e:
            logger.error(f"[api]流式生成出错：{e}", exc_info=True)
            yield _sse_pack({
                "type": "error",
                "code": "AGENT_EXECUTION_ERROR",
                "message": "本次回答暂时无法完成，请稍后重试。",
            })

    # media_type 必须是 text/event-stream，浏览器/EventSource 才会按 SSE 处理
    return StreamingResponse(event_generator(), media_type="text/event-stream")
