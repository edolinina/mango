import asyncio
import os
from types import SimpleNamespace

import pytest

from utils import helpers


class _FakeMCPClient:
    def __init__(self, tools):
        self._tools = tools
        self.calls = 0

    async def get_tools(self):
        self.calls += 1
        return self._tools


def test_get_mcp_endpoint_uses_cache() -> None:
    helpers._MCP_TOOLS_CACHE.clear()
    client = _FakeMCPClient([SimpleNamespace(name="list_messages"), SimpleNamespace(name="send_feedback")])

    first = asyncio.run(helpers.get_mcp_endpoint(client, "list_messages"))
    second = asyncio.run(helpers.get_mcp_endpoint(client, "send_feedback"))

    assert first.name == "list_messages"
    assert second.name == "send_feedback"
    assert client.calls == 1


def test_load_model_openai_without_api_key_raises() -> None:
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            helpers.load_model(provider="openai", model="gpt-4o-mini")
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old
