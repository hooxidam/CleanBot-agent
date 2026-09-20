from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest, before_model
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.graph import END
from utils.logger_handler import logger
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from typing import Callable

from store.profile_store import FIELD_LABELS, get_profile
from store.user_store import get_latest_snapshot
from utils.prompt_loader import load_report_prompt, load_system_prompt
from agent.guard import AgentGuardError, AgentGuardState


@wrap_tool_call
def monitor_tool(
        #请求的数据封装
        request: ToolCallRequest,
        #执行的函数本身
        handler: Callable[[ToolCallRequest],ToolMessage | Command]) -> ToolMessage | Command:
    # 在业务工具 handler 之前执行硬限制；被拦截的调用不会真正执行。
    guard = request.runtime.context.get("agent_guard")
    if isinstance(guard, AgentGuardState):
        try:
            guard.before_tool(
                request.tool_call["name"],
                request.tool_call.get("args") or {},
            )
        except AgentGuardError as exc:
            # ToolNode 默认会把普通异常转成 ToolMessage 并让模型继续循环。
            # 使用官方 Command 直接结束图；外层随后将 stop_error 转为友好 SSE。
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=exc.user_message,
                            name=request.tool_call["name"],
                            tool_call_id=request.tool_call["id"],
                            status="error",
                        )
                    ]
                },
                goto=END,
            )
    #工具执行的监控
    logger.info(f"[tool_monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool_monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool_monitor]工具执行结果：{result}")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"[tool_monitor]工具执行异常：{str(e)}")
        raise


@before_model   #注册为"模型调用前"中间件钩子，create_agent 才能识别它
def log_before_model(
        state: AgentState, #整个agent智能体中的状态记录
        runtime: Runtime, #记录了整个执行过程中的上下文信息

):              #在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content}")

    return None


def _flatten(text: str) -> str:
    """把多项内容压成一行，便于放进提示词。
    注意：records.csv 里的分隔符是**字面的两个字符 \\ 和 n**，不是真正的换行符，
    所以这里要替换字符串 "\\n" 而不是换行符本身。"""
    return text.replace("\\n", "；").replace("\n", "；")


def _build_user_profile_block(user_id: str | None) -> str:
    """把当前用户的档案拼成一段提示词。

    为什么注入而不是让模型调工具去查：
    家居环境(养宠/地面/面积)决定了几乎每一个回答该怎么给。若靠模型自觉去查，
    "主刷多久换"这种看起来很通用的问题它压根想不到要先了解用户，于是又退回成
    背说明书的 FAQ 机器人。注入是塞到它眼前，想忽略都难。

    ⚠️ 关键：**未知的字段也要显式写出来**。
    如果只写已知项，模型会默认"没写的就是没有"——比如 1001 的养宠状态其实是
    "不知道"，但模型会当成"没养宠"，然后自信地给出不适用的建议。
    把"未知"摆明，模型才会在需要时主动去问。
    """
    if not user_id:
        return ""

    profile = get_profile(user_id)
    snapshot = get_latest_snapshot(user_id)
    if not profile and not snapshot:
        return ""

    lines = ["\n\n### 当前用户档案（系统已提供，无需调用工具查询以下信息）",
             f"- 用户ID：{user_id}"]

    unknown = []
    if profile:
        for field, label in FIELD_LABELS.items():
            info = profile["fields"][field]
            if info["source"] == "unknown":
                unknown.append((field, label))
            else:
                #标出来源：设备测得的可信，用户自述的也可信，但性质不同
                origin = "设备自动测得" if info["source"] == "device" else "用户此前告知"
                lines.append(f"- {label}：{info['value']}（{origin}）")

    if snapshot:
        if snapshot["latest_month"]:
            lines.append(f"- 设备数据截至：{snapshot['latest_month']}")
        if snapshot["consumables"]:
            lines.append(f"- 最新耗材状态：{_flatten(snapshot['consumables'])}")
        if snapshot["efficiency"]:
            lines.append(f"- 最新清洁表现：{_flatten(snapshot['efficiency'])}")

    if unknown:
        lines.append("- **以下信息尚不掌握**：" + "、".join(label for _, label in unknown))

    lines.append(
        "\n【回答要求】"
        "\n1. 问题与用户的家居环境、耗材或使用情况相关时，必须结合上述档案给出针对"
        "这位用户的具体建议并说明依据（如“您家养狗且铺短毛地毯，毛发易缠绕，因此建议…”），"
        "不要给放之四海而皆准的通用答案；"
        "\n2. 若档案显示耗材临近更换或故障反复，主动提醒；"
        "\n3. **当某项未掌握的信息会明显改变你的建议时**，先给出当前能给的答案，"
        "再顺带问用户一句以便后续更准确（例如“主刷一般 3-6 个月更换；请问您家里养宠物吗？"
        "如果有，建议缩短到 1-3 个月”）。不要一次追问多项，也不要为了填档案而问"
        "——只在这项信息确实影响建议时才问；"
        "\n4. 用户告知了这类信息后，**调用 update_my_profile 记录下来**，之后就不必再问。"
    )
    return "\n".join(lines)


@dynamic_prompt   #每一次在生成提示词之前，调用此函数
def dynamic_system_prompt(request: ModelRequest):
    """动态组装系统提示词 = 基础提示词(按场景切换) + 当前用户档案"""
    is_report = request.runtime.context.get("report", False)
    #报告场景用报告写手提示词，其余用通用客服提示词
    base = load_report_prompt() if is_report else load_system_prompt()
    #无论哪种场景，都带上当前用户的档案
    return base + _build_user_profile_block(request.runtime.context.get("user_id"))

