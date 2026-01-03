from fastmcp import FastMCP
from mcp_server.protocol import MCPEnvelope, Directive, AgentOutput

mcp = FastMCP("MANGO-MCP")

# in-memory messages and directives store
MESSAGES: list[MCPEnvelope] = []

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
   # payload = Directive(**msg.payload)
   # DIRECTIVES[payload.id] = payload.model_dump()
   # return f"Directive '{payload.task}' accepted"

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
    mcp.run(transport="http", host="127.0.0.1", port=8000)
