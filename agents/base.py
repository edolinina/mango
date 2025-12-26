import logging
import json

from uuid import uuid4
from langchain.agents import create_agent

from utils.llm import load_model
from core.mcp.protocol import AgentOutput, MCPEnvelope


class BaseAgent:
    def __init__(self, agent_name, mcp_client, manager_name="CentralExecutive"):
        self.logger = logging.getLogger("mango")
        self.client = mcp_client
        self.name = agent_name
        self.manager = manager_name

        llm = load_model()
        self.model = llm.with_structured_output(AgentOutput)

    async def handle_directive(self, directive=None):
        """Fetch directive for this agent from the MCP `list_messages` tool
        if `directive` is not provided, then call `process_directive` and
        send feedback to the Central Executive.
        """
        if not directive:
            tools = await self.client.get_tools()
            list_tool = next((t for t in tools if getattr(t, "name", "") == "list_messages"), None)
            if not list_tool:
                self.logger.info("list_messages tool not available on MCP client")
                return None
            
            messages = await list_tool.ainvoke({})
            for msg in messages:
                txt = msg['text']
                parsed = json.loads(txt)

                match = None
                for entry in parsed:
                    if entry.get('message_type') == 'directive' and entry.get('target') == self.name:
                        match = entry.get('payload')

                if match:
                    directive = match
                    break

        if not directive:
            self.logger.info(f"No directive found for agent {self.name}")
            return None

        result = await self.process_directive(directive)
        print(result)

        # send feedback back to MCP
        tools = await self.client.get_tools()
        send_tool = next((t for t in tools if getattr(t, "name", "") == "send_feedback"), None)
        if send_tool:
            await send_tool.ainvoke({
                "envelope": {
                    "message_type": "agent_feedback",
                    "sender": self.name,
                    "target": self.manager,
                    "message_id": str(uuid4()),
                    "payload": result.model_dump(),
                }
            })

        return result

    async def process_directive(self, directive):
        """Subclasses should implement this to perform work on `directive`.

        Returns an `AgentOutput` (or dict compatible with it).
        """
        raise NotImplementedError
