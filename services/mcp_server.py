import os

from fastmcp import FastMCP, Context
from schemas import MCPEnvelope, Directive, AgentOutput
from agents.dataset_analysis_tool import run_dataset_analysis

mcp = FastMCP("MANGO-MCP")

MCP_PORT = int(os.getenv("MCP_PORT", 8000))
MESSAGES: list[MCPEnvelope] = [] # in-memory messages and directives store

def _find_message_by_id(id: str) -> dict | None:
    for msg in MESSAGES:
        if msg["message_id"] == id:
            return msg
    return None

@mcp.tool()
async def send_directive(envelope: dict, ctx: Context) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = Directive(**msg.payload)
    await ctx.info(f"Directive accepted for {msg.target}: {payload.task}")
    return f"Directive '{payload.task}' accepted"

@mcp.tool()
async def remove_directive(id: str, ctx: Context) -> str:
    msg = _find_message_by_id(id)
    if not msg:
        await ctx.info(f"Directive not found: {id}")
        return "Message not found"

    MESSAGES.remove(msg)
    await ctx.info(f"Directive removed: {id}")
    return f"Message {id} removed"

@mcp.tool()
async def send_feedback(envelope: dict, ctx: Context) -> str:
    msg = MCPEnvelope(**envelope)
    MESSAGES.append(msg.model_dump())

    payload = AgentOutput(**msg.payload)
    await ctx.info(f"Feedback accepted from {payload.agent} ({payload.capability})")
    return f"AgentOutput {payload.agent} accepted"

@mcp.tool()
async def list_messages(ctx: Context) -> list:
    """Return a copy of stored MCP envelopes."""
    await ctx.info(f"Listing {len(MESSAGES)} message envelope(s)")
    return list(MESSAGES)

mcp.tool()(run_dataset_analysis)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)
