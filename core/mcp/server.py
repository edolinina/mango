from fastmcp import FastMCP
from core.mcp.protocol import MCPEnvelope, Directive

mcp = FastMCP("MANGO-MCP")

@mcp.tool()
def send_directive(envelope: dict) -> str:
    msg = MCPEnvelope(**envelope)
    payload = Directive(**msg.payload)
    return f"Directive '{payload.objective}' accepted"

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
