import pytest
from types import SimpleNamespace

from langgraph.graph import END
from langgraph.types import Command

from agent.guard import (
    AgentGuardState,
    AgentRecursionLimitError,
    AgentRepeatedToolCallError,
    AgentToolCallLimitError,
    LANGGRAPH_RECURSION_LIMIT,
    MAX_TOOL_CALLS,
)


def make_guard(**overrides) -> AgentGuardState:
    values = {
        "request_id": "req-test",
        "session_id": "session-test",
        "user_id": "1004",
    }
    values.update(overrides)
    return AgentGuardState(**values)


def test_normal_chat_starts_with_zero_tool_calls():
    guard = make_guard()

    assert guard.tool_call_count == 0


def test_at_most_eight_tools_are_allowed():
    guard = make_guard(max_tool_calls=8, max_consecutive_duplicate_calls=8)

    for index in range(8):
        assert guard.before_tool("get_device_status", {"attempt": index}) == index + 1

    with pytest.raises(AgentToolCallLimitError) as captured:
        guard.before_tool("get_error_history", {})

    assert guard.tool_call_count == 8
    assert captured.value.tool_call_count == 8
    assert captured.value.code == "AGENT_STEP_LIMIT"


def test_multiple_tool_calls_are_counted_individually():
    guard = make_guard()

    guard.before_tool("get_device_status", {})
    guard.before_tool("get_error_history", {"days": 7})
    guard.before_tool("rag_summarize", {"query": "E5"})

    assert guard.tool_call_count == 3


def test_same_tool_and_normalized_arguments_are_stopped_early():
    guard = make_guard(max_consecutive_duplicate_calls=3)

    guard.before_tool("get_device_status", {"a": 1, "b": 2})
    guard.before_tool("get_device_status", {"b": 2, "a": 1})
    guard.before_tool("get_device_status", {"a": 1, "b": 2})

    with pytest.raises(AgentRepeatedToolCallError):
        guard.before_tool("get_device_status", {"a": 1, "b": 2})

    # 第 4 次被拦截，没有进入真实工具执行，所以已执行次数仍为 3。
    assert guard.tool_call_count == 3


def test_different_tool_call_resets_consecutive_duplicate_count():
    guard = make_guard(max_consecutive_duplicate_calls=2)

    guard.before_tool("get_device_status", {})
    guard.before_tool("get_device_status", {})
    guard.before_tool("get_error_history", {})
    guard.before_tool("get_device_status", {})

    assert guard.tool_call_count == 4


def test_each_user_turn_uses_a_fresh_counter():
    first_turn = make_guard(request_id="req-1")
    for index in range(6):
        first_turn.before_tool("rag_summarize", {"query": str(index)})

    second_turn = make_guard(request_id="req-2")

    assert first_turn.tool_call_count == 6
    assert second_turn.tool_call_count == 0
    assert second_turn.before_tool("get_device_status", {}) == 1


def test_recursion_limit_is_higher_than_business_tool_limit():
    assert MAX_TOOL_CALLS == 8
    assert LANGGRAPH_RECURSION_LIMIT > MAX_TOOL_CALLS


def test_recursion_error_has_safe_sse_payload():
    error = AgentRecursionLimitError(
        request_id="req-test",
        session_id="session-test",
        user_id="1004",
        tool_call_count=4,
    )

    event = error.to_sse_event()

    assert event["type"] == "error"
    assert event["code"] == "AGENT_STEP_LIMIT"
    assert "GraphRecursionError" not in event["message"]
    assert "recursion_limit" not in event["message"]


def test_tool_middleware_stops_graph_before_blocked_tool_executes():
    # 延迟导入，避免本模块其他纯计数测试依赖工具实现细节。
    from agent.tools.middleware import monitor_tool

    guard = make_guard(max_tool_calls=1)
    context = {"agent_guard": guard, "report": False}
    calls = []

    def handler(request):
        calls.append(request.tool_call["name"])
        return "ok"

    first_request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={"id": "call-1", "name": "get_device_status", "args": {}},
    )
    second_request = SimpleNamespace(
        runtime=SimpleNamespace(context=context),
        tool_call={"id": "call-2", "name": "get_error_history", "args": {}},
    )

    assert monitor_tool.wrap_tool_call(first_request, handler) == "ok"
    blocked = monitor_tool.wrap_tool_call(second_request, handler)

    assert calls == ["get_device_status"]
    assert isinstance(blocked, Command)
    assert blocked.goto == END
    assert guard.stop_error is not None
