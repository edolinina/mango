import asyncio
import logging

from uuid import uuid4
from langchain.agents import create_agent

from utils.llm import load_model
from core.mcp.protocol import AgentOutput, MCPEnvelope


class BaseAgent:
    def __init__(self, agent_name, mcp_client):
        self.logger = logging.getLogger("mango")
        self.client = mcp_client
        self.name = agent_name

        llm = load_model()
        self.model = llm.with_structured_output(AgentOutput)

    async def get_directives(self):
        """Get agent directive from MCP"""
        pass

    async def send_response(self, response: AgentOutput):
        """Send directives to MCP"""
        tools = await self.client.get_tools()
        send_tool = next(t for t in tools if t.name == "send_response")

        response = response.model_dump()
        self.logger.info(f"Sending agent response {response} to MCP")
        envelope = MCPEnvelope(
            message_type="agent_response",
            sender=self.name,
            target="CentralExecutive",
            message_id=str(uuid4()),
            payload=response
        )
        res = await send_tool.ainvoke({"envelope": envelope.model_dump()})
        results.append(res)

        return results