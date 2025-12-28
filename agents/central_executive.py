import logging

from uuid import uuid4
from langchain.agents import create_agent

from mcp_server.protocol import CEOutput, MCPEnvelope

logger = logging.getLogger("mango")


class CentralExecutive:
    def __init__(self, model, mcp_client):
        self.client = mcp_client
        self.model = model.with_structured_output(CEOutput)

    async def generate_directives(self, intent: str) -> CEOutput:
        """Use LLM ONLY to generate structured directives"""
        return await self.model.ainvoke(intent)

    async def send_directives(self, intent: CEOutput):
        """Send directives to MCP"""
        tools = await self.client.get_tools()
        send_tool = next(t for t in tools if t.name == "send_directive")

        results = []
        for d in intent.directives:
            directive = d.model_dump()
            logger.info(f"Sending directive {directive} to MCP")
            envelope = MCPEnvelope(
                message_type="directive",
                sender="CentralExecutive",
                target=d.agent,
                message_id=str(uuid4()),
                payload=directive
            )
            res = await send_tool.ainvoke({"envelope": envelope.model_dump()})
            results.append(res)

        return results
