import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from services.agent_api import create_agent_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeAgent:
    name = "BusinessAgent"
    print_name = "BusinessAgent X"

    async def process_agent_directive(self):
        return []


class _BlockingFakeAgent(_FakeAgent):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def process_agent_directive(self):
        self.started.set()
        await self.release.wait()
        self.finished.set()
        return []


@pytest.mark.anyio
async def test_process_directive_accepts_when_idle() -> None:
    app = create_agent_app(_FakeAgent())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/process-directive")

    assert response.status_code == 200
    assert response.json() == {"agent": "BusinessAgent X", "status": "accepted"}


@pytest.mark.anyio
async def test_process_directive_returns_already_running_when_task_active() -> None:
    agent = _BlockingFakeAgent()
    app = create_agent_app(agent)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/process-directive")
        await agent.started.wait()
        second = await client.post("/process-directive")
        agent.release.set()
        await agent.finished.wait()

    assert first.status_code == 200
    assert first.json()["status"] == "accepted"
    assert second.status_code == 200
    assert second.json() == {"agent": "BusinessAgent X", "status": "already_running"}
