import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from services.ce_api import create_ce_app


class _FakeCE:
    def __init__(self):
        self.workflow_error = None
        self.agents = [SimpleNamespace(print_name="BusinessAgent X")]
        self.directives = None
        self.generate_directives = AsyncMock()
        self.collect_agent_feedback = AsyncMock(return_value={})

    def clear_workflow_error(self):
        self.workflow_error = None

    def set_workflow_error(self, message: str):
        self.workflow_error = message

    def update_directives(self, directives):
        self.directives = directives


def test_run_returns_pending_when_autonomous_mode_false() -> None:
    ce = _FakeCE()
    ce.generate_directives.return_value = SimpleNamespace(
        directives=[SimpleNamespace(agent="BusinessAgent", task="Increase margin", capability="profitability")]
    )

    os.environ["AUTONOMOUS_MODE"] = "false"
    app = create_ce_app(ce)

    with TestClient(app) as client:
        response = client.post("/run", json={"task": "Improve profit"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert "BusinessAgent" in body["directives"]


def test_approve_returns_running() -> None:
    ce = _FakeCE()
    app = create_ce_app(ce)

    with TestClient(app) as client:
        response = client.post("/approve")

    assert response.status_code == 200
    assert response.json() == {"status": "running", "agents": ["BusinessAgent X"]}


def test_reject_returns_rejected() -> None:
    ce = _FakeCE()
    app = create_ce_app(ce)

    with TestClient(app) as client:
        response = client.post("/reject")

    assert response.status_code == 200
    assert response.json() == {"status": "rejected"}


def test_results_returns_no_results_when_feedback_empty() -> None:
    ce = _FakeCE()
    ce.collect_agent_feedback.return_value = {}
    app = create_ce_app(ce)

    with TestClient(app) as client:
        response = client.get("/results")

    assert response.status_code == 200
    assert response.json() == {"status": "no_results", "agents": []}
