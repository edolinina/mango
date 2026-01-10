import os

from fastmcp import FastMCP
from mcp_server.protocol import MCPEnvelope, Directive, AgentOutput

mcp = FastMCP("MANGO-MCP")

MCP_PORT = os.getenv("MCP_PORT", 8000)
MESSAGES: list[MCPEnvelope] = [] # in-memory messages and directives store

def _find_message_by_id(id: str) -> MCPEnvelope | None:
    for msg in MESSAGES:
        if msg["message_id"] == id:
            return msg
    return None

@mcp.tool()
def send_directive(envelope: dict) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = Directive(**msg.payload)
    return f"Directive '{payload.task}' accepted"

@mcp.tool()
def remove_directive(id: str) -> str:
    msg = _find_message_by_id(id)
    if not msg:
        return "Message not found"

    MESSAGES.remove(msg)
    return f"Message {id} removed"

@mcp.tool()
def send_feedback(envelope: dict) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = AgentOutput(**msg.payload)
    return f"AgentOutput {payload.agent} accepted"

@mcp.tool()
def list_messages() -> list:
    """Return a copy of stored MCP envelopes."""
    return list(MESSAGES)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)
