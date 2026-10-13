"""Unit tests for Orchestrator state between runs (issue #15)."""

from typing import TYPE_CHECKING, cast

import pytest

from agent.memory.session_store import SessionStore
from agent.orchestrator import Orchestrator

if TYPE_CHECKING:
    import redis


class FakeRedis:
    """In-memory stand-in for the Redis client used by SessionStore."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return the stored value, or None if missing."""
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        """Store a value. The TTL is ignored."""
        self.store[key] = value

    def delete(self, key: str) -> None:
        """Remove a key if present."""
        self.store.pop(key, None)


class CountingTool:
    """Tool that returns a new call number on every real call."""

    def __init__(self, name: str) -> None:
        """Initialize with a name and zero calls."""
        self.name = name
        self.calls = 0

    def execute(self, tool_input: dict) -> dict:
        """Count the call and return the call number."""
        self.calls += 1
        return {"tool": self.name, "call_number": self.calls}


def _make_orchestrator() -> tuple[Orchestrator, dict[str, CountingTool], FakeRedis]:
    """Build an Orchestrator with counting tools and a fake Redis client."""
    tools = {n: CountingTool(n) for n in ["readme_scorer", "skill_extractor", "market_analyzer"]}
    fake_redis = FakeRedis()
    orch = Orchestrator(tools=tools, session_store=SessionStore(cast("redis.Redis", fake_redis)))
    return orch, tools, fake_redis


@pytest.mark.unit
def test_second_run_with_same_input_calls_tool_again() -> None:
    """A second run must not reuse the first run's cached tool result."""
    orch, tools, _ = _make_orchestrator()
    profile = {"readme_content": "# My project"}

    first = orch.run("profile-1", profile)
    second = orch.run("profile-1", profile)

    assert tools["readme_scorer"].calls == 2
    assert first["tool_results"]["readme_scorer"]["call_number"] == 1
    assert second["tool_results"]["readme_scorer"]["call_number"] == 2


@pytest.mark.unit
def test_saved_session_drops_tools_not_used_in_latest_run() -> None:
    """Session state saved by an earlier run must not survive into the next one."""
    orch, _, fake_redis = _make_orchestrator()

    orch.run("profile-2", {"readme_content": "# A", "resume_text": "Python"})
    orch.run("profile-2", {"readme_content": "# B"})

    saved = SessionStore(cast("redis.Redis", fake_redis)).get("profile-2")
    assert saved is not None
    assert sorted(saved) == ["market_analyzer", "readme_scorer"]
