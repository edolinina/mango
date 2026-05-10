import asyncio

from services import mcp_server


class _Ctx:
    def __init__(self):
        self.messages = []

    async def info(self, message: str):
        self.messages.append(message)


def test_send_directive_appends_message() -> None:
    mcp_server.MESSAGES.clear()
    ctx = _Ctx()

    envelope = {
        "message_type": "directive",
        "sender": "CentralExecutive",
        "target": "BusinessAgent",
        "payload": {"agent": "BusinessAgent", "task": "Increase profit", "capability": "profitability"},
    }

    result = asyncio.run(mcp_server.send_directive(envelope, ctx))

    assert result.startswith("Directive")
    assert len(mcp_server.MESSAGES) == 1


def test_remove_directive_returns_not_found_for_unknown_id() -> None:
    mcp_server.MESSAGES.clear()
    ctx = _Ctx()

    result = asyncio.run(mcp_server.remove_directive("missing-id", ctx))

    assert result == "Message not found"


def test_list_messages_returns_current_envelopes() -> None:
    mcp_server.MESSAGES.clear()
    ctx = _Ctx()
    mcp_server.MESSAGES.append({"message_id": "m1"})

    result = asyncio.run(mcp_server.list_messages(ctx))

    assert result == [{"message_id": "m1"}]
