import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.central_executive import CEOutput, CentralExecutive
from schemas import Directive


class _ModelWithStructuredOutput:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _):
        return self

    async def ainvoke(self, _prompt):
        return self._result


def test_generate_directives_sets_task_id() -> None:
    expected = CEOutput(
        directives=[Directive(agent="BusinessAgent", task="Improve margin", capability="profitability")],
        task_id="task-123",
    )
    model = _ModelWithStructuredOutput(expected)
    ce = CentralExecutive(model=model, mcp_client=object(), config={"instructions": "Task: {task}; Agents: {agents}"}, agents=[])

    result = asyncio.run(ce.generate_directives("Increase profit"))

    assert result.task_id == "task-123"
    assert ce.task_id == "task-123"


def test_send_directives_calls_mcp_for_each_directive() -> None:
    model = _ModelWithStructuredOutput(None)
    ce = CentralExecutive(model=model, mcp_client=object(), config={"instructions": "x"}, agents=[])

    intent = CEOutput(
        directives=[
            Directive(agent="BusinessAgent", task="Task A", capability="capA"),
            Directive(agent="HRAgent", task="Task B", capability="capB"),
        ],
        task_id="task-999",
    )

    endpoint = SimpleNamespace(ainvoke=AsyncMock(return_value="ok"))
    with patch("agents.central_executive.get_mcp_endpoint", AsyncMock(return_value=endpoint)):
        result = asyncio.run(ce.send_directives(intent))

    assert result == ["ok", "ok"]
    assert endpoint.ainvoke.await_count == 2


def test_collect_agent_feedback_filters_by_task_id() -> None:
    model = _ModelWithStructuredOutput(None)
    ce = CentralExecutive(model=model, mcp_client=object(), config={"instructions": "x"}, agents=[])
    ce.task_id = "task-42"

    payload = {
        "message_type": "agent_feedback",
        "sender": "BusinessAgent",
        "payload": {"task_id": "task-42", "results": "ok"},
    }
    other = {
        "message_type": "agent_feedback",
        "sender": "BusinessAgent",
        "payload": {"task_id": "other", "results": "skip"},
    }
    endpoint = SimpleNamespace(
        ainvoke=AsyncMock(return_value=[{"text": "[" + __import__("json").dumps(payload) + "," + __import__("json").dumps(other) + "]"}])
    )

    with patch("agents.central_executive.get_mcp_endpoint", AsyncMock(return_value=endpoint)):
        result = asyncio.run(ce.collect_agent_feedback())

    assert "BusinessAgent" in result
    assert result["BusinessAgent"][0]["task_id"] == "task-42"
