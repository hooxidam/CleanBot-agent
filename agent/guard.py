"""ReAct Agent 的单轮运行时硬限制。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from utils.config_handler import agent_conf
from utils.logger_handler import logger


MAX_TOOL_CALLS = int(agent_conf.get("max_tool_calls", 8))
LANGGRAPH_RECURSION_LIMIT = int(agent_conf.get("recursion_limit", 24))
MAX_CONSECUTIVE_DUPLICATE_TOOL_CALLS = int(
    agent_conf.get("max_consecutive_duplicate_tool_calls", 3)
)

if MAX_TOOL_CALLS < 1:
    raise ValueError("max_tool_calls 必须大于 0")
if LANGGRAPH_RECURSION_LIMIT <= MAX_TOOL_CALLS:
    raise ValueError("recursion_limit 必须明显高于 max_tool_calls")
if MAX_CONSECUTIVE_DUPLICATE_TOOL_CALLS < 1:
    raise ValueError("max_consecutive_duplicate_tool_calls 必须大于 0")


AGENT_STEP_LIMIT_MESSAGE = (
    "当前问题需要的诊断步骤较多，本次自动诊断已达到最大执行次数。"
    "根据目前已获取的信息，我暂时无法继续自动查询。"
    "你可以补充更具体的故障现象后重新提问。"
)
AGENT_REPEATED_TOOL_MESSAGE = (
    "本次诊断连续进行了相同查询，为避免重复执行，系统已停止本轮自动诊断。"
    "你可以补充更具体的故障现象后重新提问。"
)


def _normalized_tool_signature(tool_name: str, arguments: Any) -> str:
    """生成稳定签名，仅用于内存中的连续重复判断，不写入日志。"""
    try:
        normalized_arguments = json.dumps(
            arguments or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        normalized_arguments = repr(arguments)
    return f"{tool_name}:{normalized_arguments}"


class AgentGuardError(RuntimeError):
    """可以安全转换成 SSE 友好事件的 Agent 运行限制异常。"""

    code = "AGENT_STEP_LIMIT"
    user_message = AGENT_STEP_LIMIT_MESSAGE

    def __init__(self, *, request_id: str, session_id: str,
                 user_id: str | None, tool_call_count: int) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        self.tool_call_count = tool_call_count
        super().__init__(self.user_message)

    def to_sse_event(self) -> dict[str, str]:
        return {"type": "error", "code": self.code, "message": self.user_message}


class AgentToolCallLimitError(AgentGuardError):
    """本轮执行满业务工具次数后，模型仍尝试继续调用工具。"""


class AgentRepeatedToolCallError(AgentGuardError):
    """模型连续重复完全相同的工具调用。"""

    code = "AGENT_REPEATED_TOOL_CALL"
    user_message = AGENT_REPEATED_TOOL_MESSAGE


class AgentRecursionLimitError(AgentGuardError):
    """LangGraph 图执行步数触发最终保险。"""


@dataclass
class AgentGuardState:
    """单条用户消息对应的线程安全工具调用计数器。"""

    request_id: str
    session_id: str
    user_id: str | None
    max_tool_calls: int = MAX_TOOL_CALLS
    max_consecutive_duplicate_calls: int = MAX_CONSECUTIVE_DUPLICATE_TOOL_CALLS
    tool_call_count: int = 0
    stop_error: AgentGuardError | None = None
    _last_signature: str | None = None
    _consecutive_duplicate_calls: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def before_tool(self, tool_name: str, arguments: Any) -> int:
        """在业务工具执行前登记调用；被拦截的调用不会进入 handler。"""
        signature = _normalized_tool_signature(tool_name, arguments)

        with self._lock:
            if self.tool_call_count >= self.max_tool_calls:
                logger.warning(
                    "[AgentGuard] max tool calls reached: %s/%s "
                    "request_id=%s session_id=%s user_id=%s",
                    self.tool_call_count, self.max_tool_calls, self.request_id,
                    self.session_id, self.user_id,
                )
                self.stop_error = AgentToolCallLimitError(
                    request_id=self.request_id, session_id=self.session_id,
                    user_id=self.user_id, tool_call_count=self.tool_call_count,
                )
                raise self.stop_error

            consecutive_calls = (
                self._consecutive_duplicate_calls + 1
                if signature == self._last_signature else 1
            )
            if consecutive_calls > self.max_consecutive_duplicate_calls:
                logger.warning(
                    "[AgentGuard] repeated tool call blocked: tool=%s repeats=%s "
                    "request_id=%s session_id=%s user_id=%s",
                    tool_name, consecutive_calls, self.request_id,
                    self.session_id, self.user_id,
                )
                self.stop_error = AgentRepeatedToolCallError(
                    request_id=self.request_id, session_id=self.session_id,
                    user_id=self.user_id, tool_call_count=self.tool_call_count,
                )
                raise self.stop_error

            self._last_signature = signature
            self._consecutive_duplicate_calls = consecutive_calls
            self.tool_call_count += 1
            current_count = self.tool_call_count

        logger.info(
            "[AgentGuard] tool_call=%s/%s tool=%s request_id=%s session_id=%s",
            current_count, self.max_tool_calls, tool_name,
            self.request_id, self.session_id,
        )
        return current_count
