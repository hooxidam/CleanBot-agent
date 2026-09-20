from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt, load_summary_prompt
from agent.tools.agent_tools import (rag_summarize, get_device_status, get_error_history,
                                     get_device_info, get_my_usage_record, update_my_profile,
                                     fill_context_for_report)
from store.db import connect
from agent.tools.middleware import monitor_tool,log_before_model,dynamic_system_prompt
from agent.guard import (
    AgentGuardState,
    AgentRecursionLimitError,
    LANGGRAPH_RECURSION_LIMIT,
)
from utils.logger_handler import logger
from uuid import uuid4

#===== 上下文压缩阈值 =====
#按 token 而不是消息条数来衡量，因为真正的约束是 context window 和 token 成本，
#而不是"聊了几句"。ReAct 里用户问一句会产生约 4 条消息(提问 + 多轮"思考+调工具" + 工具返回 + 回答)，
#一条塞满检索资料的工具返回，可能顶得上十条闲聊 —— 用条数当尺子会过早触发压缩。
#12000 token 大约相当于十几轮正常问答，日常客服对话基本不会碰到；
#只有真正聊得很长时才压缩，避免撑爆上下文。
SUMMARIZE_TRIGGER_TOKENS = 12000
#压缩时保留最近这些条原文，其余压成摘要。保留足够多才接得住"它刚说的那个呢"这类追问。
SUMMARIZE_KEEP_MESSAGES = 20


class ReactAgent:
    def __init__(self):
        #会话历史与档案共用同一个应用数据库(见 store/db.py)
        #checkpointer(检查点存储)：每轮对话后自动把 messages 状态按 thread_id 存盘，
        #下次带同一个 thread_id 请求时自动加载回来 —— 这就是"多轮记忆"的实现基础。
        self.checkpointer = SqliteSaver(connect())
        self.checkpointer.setup()   #首次运行时建表

        self.agent = create_agent(
            model= chat_model,
            system_prompt= load_system_prompt(),
            tools=[rag_summarize, get_device_status, get_error_history,
                   get_device_info, get_my_usage_record, update_my_profile,
                   fill_context_for_report],
            middleware= [
                monitor_tool,
                log_before_model,
                dynamic_system_prompt,
                #上下文压缩：只在对话真的很长(见上方阈值)时，才把早期历史压成一段中文摘要，
                #保留最近若干条原文。目的是防止撑爆 context window 和 token 成本失控，
                #而不是频繁裁剪正常长度的对话。
                SummarizationMiddleware(
                    model=chat_model,
                    trigger=("tokens", SUMMARIZE_TRIGGER_TOKENS),
                    keep=("messages", SUMMARIZE_KEEP_MESSAGES),
                    summary_prompt=load_summary_prompt(),
                ),
            ],
            checkpointer= self.checkpointer,
        )

    @staticmethod
    def _config(session_id: str) -> dict:
        """把 session_id 包成 LangGraph 的 config。
        thread_id 相同 = 同一场对话，checkpointer 据此加载/保存历史。"""
        return {
            "configurable": {"thread_id": session_id},
            # 图执行层最终保险。一次工具调用通常跨越 model/tools 多个节点，
            # 因此该值必须明显高于业务层的 MAX_TOOL_CALLS。
            "recursion_limit": LANGGRAPH_RECURSION_LIMIT,
        }

    def get_history(self, session_id: str) -> list[dict]:
        """读取某场对话的历史消息，供前端切换会话时回显。
        数据直接来自 checkpointer 存的状态，不用我们自己再存一份消息。
        :return: [{"role": "user"/"bot", "content": "..."}, ...]
        """
        state = self.agent.get_state(self._config(session_id))
        history = []
        for msg in state.values.get("messages", []):
            #只回显"人看的"内容：用户提问和助手的文字回复。
            #工具调用请求(content 为空)、工具返回结果都跳过。
            msg_type = getattr(msg, "type", None)
            content = getattr(msg, "content", None)
            if not content or not isinstance(content, str):
                continue
            if msg_type == "human":
                role = "user"
            elif msg_type == "ai":
                role = "bot"
            else:
                continue

            #合并连续的同角色消息：ReAct 每次调工具前的"思考旁白"都是独立的 AI 消息，
            #而流式直播时它们是被拼进同一个气泡的。这里也合并，
            #保证"刷新/切回会话后看到的"和"当时直播看到的"长得一样。
            if history and history[-1]["role"] == role:
                history[-1]["content"] += content
            else:
                history.append({"role": role, "content": content})
        return history

    def delete_history(self, session_id: str):
        """删除某场对话的全部消息历史(会话元数据由 SessionStore 负责删)"""
        self.checkpointer.delete_thread(session_id)

    def stream_events(
        self,
        query: str,
        session_id: str,
        user_id: str | None = None,
        request_id: str | None = None,
    ):
        """带双层运行限制地输出本轮 SSE 事件。"""
        request_id = request_id or uuid4().hex[:12]
        # 计数器只属于这一次用户请求，不进入 AgentState/SqliteSaver。
        guard = AgentGuardState(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
        )
        try:
            yield from self._stream_events_impl(
                query=query,
                session_id=session_id,
                user_id=user_id,
                request_id=request_id,
                guard=guard,
            )
            if guard.stop_error is not None:
                raise guard.stop_error
        except GraphRecursionError as exc:
            logger.error(
                "[AgentGuard] LangGraph recursion limit reached "
                "request_id=%s session_id=%s user_id=%s tool_call_count=%s "
                "exception_type=%s",
                request_id, session_id, user_id, guard.tool_call_count,
                type(exc).__name__, exc_info=True,
            )
            raise AgentRecursionLimitError(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                tool_call_count=guard.tool_call_count,
            ) from exc
        finally:
            logger.info(
                "[AgentGuard] request finished request_id=%s session_id=%s "
                "tool_call_count=%s/%s",
                request_id, session_id, guard.tool_call_count,
                guard.max_tool_calls,
            )

    def _stream_events_impl(
        self,
        query: str,
        session_id: str,
        user_id: str | None,
        request_id: str,
        guard: AgentGuardState,
    ):
        """
        流式产出结构化事件，供 FastAPI 的 SSE 接口直接转发给前端。

        用多模式流 stream_mode=["updates", "messages"] 一次拿到两类信息：
          - "updates"：每个节点跑完的产出 —— 从中提取「要调用哪个工具/参数」和「工具已返回」，
                       用于前端可视化 Agent 的 ReAct 过程；
          - "messages"：模型生成过程中的 token 块 —— 用于逐字吐字。
        两者顺序天然正确：模型决定调工具 → 工具返回 → 模型生成最终回答。

        :param query: 用户提问
        :param session_id: 会话 ID。相同 = 同一场对话，Agent 能记住之前聊过什么
        :param user_id: 当前登录用户，注入运行时上下文，供各查询工具确定"查谁的数据"
        :return: 生成器，逐个 yield 事件字典：
                 {"type": "tool_start", "name": 工具名, "args": {...}}
                 {"type": "tool_end",   "name": 工具名}
                 {"type": "token",      "content": "文本片段"}
        """
        input_dict = {"messages": [{"role": "user", "content": query}]}
        #context 是运行时上下文：report 标记供动态提示词中间件用，
        #user_id 供各查询工具读取 —— 工具签名里不暴露 user_id，模型无法指定查谁，杜绝越权
        context = {
            "report": False,
            "user_id": user_id,
            "request_id": request_id,
            "agent_guard": guard,
        }

        for mode, payload in self.agent.stream(
            input_dict, stream_mode=["updates", "messages"], context=context,
            config=self._config(session_id),
        ):
            if mode == "messages":
                msg_chunk, metadata = payload
                # 只放行 model 节点产出的 token：
                # rag_summarize 工具内部自己也会调大模型做总结，那些 token 属于 tools 节点。
                # 不过滤的话，工具的内部总结会混进最终回答里，用户会看到重复/割裂的内容。
                if metadata.get("langgraph_node") != "model":
                    continue
                # 只取"给用户看的回复文本"：AI 消息且内容非空
                # (工具返回的 ToolMessage、请求调用工具时的空内容块都跳过)
                # AIMessageChunk 是 AIMessage 的子类，用 AIMessage 判断可同时兼容
                # "流式逐 token 块" 和 "非流式整段消息" 两种情况
                if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                    yield {"type": "token", "content": msg_chunk.content}

            elif mode == "updates":
                for _node, update in (payload or {}).items():
                    if not isinstance(update, dict):
                        continue
                    for msg in update.get("messages") or []:
                        #模型决定要调工具：此时还没执行，正好用来提示"正在检索…"
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {"type": "tool_start",
                                       "name": tc["name"],
                                       "args": tc.get("args") or {}}
                        #工具执行完毕
                        elif isinstance(msg, ToolMessage):
                            yield {"type": "tool_end", "name": msg.name}

if __name__ == '__main__':
    #命令行快速试跑。注意必须给 user_id：Agent 的回答依赖当前用户的档案与设备数据，
    #不传的话拿不到档案，回答会退化成通用的说明书内容。
    agent = ReactAgent()
    for event in agent.stream_events(
            query="我的机器人不动了，怎么回事？",
            session_id="cli-demo",
            user_id="1004",          # 85㎡ | 2猫 | 混合地面，设备当前正报 E5 主刷缠绕
    ):
        if event["type"] == "tool_start":
            print(f"\n[调用工具] {event['name']} {event['args']}\n", flush=True)
        elif event["type"] == "token":
            print(event["content"], end="", flush=True)
    print()
