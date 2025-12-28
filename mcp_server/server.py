from fastmcp import FastMCP
from mcp_server.protocol import MCPEnvelope, Directive, AgentOutput

mcp = FastMCP("MANGO-MCP")

# in-memory message store for list_messages
MESSAGES = []

@mcp.tool()
def send_directive(envelope: dict) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = Directive(**msg.payload)
    return f"Directive '{payload.objective}' accepted"

@mcp.tool()
def send_feedback(envelope: dict) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = AgentOutput(**msg.payload)
    return f"AgentOutput '{payload.agent} {payload.status}' accepted"

@mcp.tool()
def list_messages() -> list:
    """Return a copy of stored MCP envelopes."""
    return list(MESSAGES)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
