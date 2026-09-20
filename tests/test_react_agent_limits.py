import pytest
from langgraph.errors import GraphRecursionError

from agent.guard import AgentRecursionLimitError
from agent.guard import LANGGRAPH_RECURSION_LIMIT
from agent.react_agent import ReactAgent


def test_runtime_config_keeps_thread_id_and_adds_recursion_limit():
    config = ReactAgent._config("session-1")

    assert config["configurable"]["thread_id"] == "session-1"
    assert config["recursion_limit"] == LANGGRAPH_RECURSION_LIMIT


def test_graph_recursion_error_is_converted_to_guard_error(monkeypatch):
    agent = ReactAgent.__new__(ReactAgent)

    def fake_stream(*, query, session_id, user_id, request_id, guard):
        raise GraphRecursionError("mock recursion limit")
        yield  # pragma: no cover - 让该函数保持生成器形态

    monkeypatch.setattr(agent, "_stream_events_impl", fake_stream)

    with pytest.raises(AgentRecursionLimitError) as captured:
        list(agent.stream_events("测试", "session-1", "1004", request_id="req-1"))

    assert captured.value.request_id == "req-1"
    assert captured.value.tool_call_count == 0
    assert captured.value.to_sse_event()["code"] == "AGENT_STEP_LIMIT"


def test_new_stream_request_gets_new_turn_counter(monkeypatch):
    agent = ReactAgent.__new__(ReactAgent)
    observed_counts = []

    def fake_stream(*, query, session_id, user_id, request_id, guard):
        observed_counts.append(guard.tool_call_count)
        guard.before_tool("get_device_status", {"query": query})
        yield {"type": "token", "content": "ok"}

    monkeypatch.setattr(agent, "_stream_events_impl", fake_stream)

    list(agent.stream_events("第一轮", "same-session", "1004", request_id="req-1"))
    list(agent.stream_events("第二轮", "same-session", "1004", request_id="req-2"))

    assert observed_counts == [0, 0]
