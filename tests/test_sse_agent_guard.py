import json

from fastapi.testclient import TestClient

from agent.guard import AgentToolCallLimitError
from api import server


def _events(response_text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response_text.splitlines()
        if line.startswith("data: ")
    ]


def test_sse_normal_stream_still_sends_done(monkeypatch):
    class FakeAgent:
        def stream_events(self, query, session_id, user_id, request_id=None):
            yield {"type": "token", "content": "你好"}

    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server.session_store, "touch", lambda *args, **kwargs: None)

    response = TestClient(server.app).get(
        "/chat/stream",
        params={"query": "你好", "session_id": "sse-normal", "user_id": "1004"},
    )

    assert response.status_code == 200
    assert _events(response.text) == [
        {"type": "token", "content": "你好"},
        {"type": "done"},
    ]


def test_sse_guard_error_is_friendly_and_stream_ends(monkeypatch):
    class LimitedAgent:
        def stream_events(self, query, session_id, user_id, request_id=None):
            raise AgentToolCallLimitError(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                tool_call_count=8,
            )
            yield  # pragma: no cover

    monkeypatch.setattr(server, "agent", LimitedAgent())

    response = TestClient(server.app).get(
        "/chat/stream",
        params={"query": "循环测试", "session_id": "sse-limit", "user_id": "1004"},
    )
    events = _events(response.text)

    assert response.status_code == 200
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "AGENT_STEP_LIMIT"
    assert "GraphRecursionError" not in events[0]["message"]
    assert all(event["type"] != "done" for event in events)

