import json
import logging

from pydantic import BaseModel
from typing import List
from langchain.agents import create_agent

from mcp_server.protocol import MCPEnvelope, Directive
from utils.helpers import get_mcp_endpoint

logger = logging.getLogger("mango")


class CEOutput(BaseModel):
    directives: List[Directive]


class CentralExecutive:
    def __init__(self, model, mcp_client, config, agents):
        self.client = mcp_client
        self.model = model.with_structured_output(CEOutput)
        self.instructions = config["instructions"]
        self.avatar = config.get("avatar", "")
        self.name = f"CentralExecutive {self.avatar}"
        self.agents = agents

    async def generate_directives(self, task: str) -> CEOutput:
        """Use LLM ONLY to generate structured directives"""
        logger.info(f"CE task: {task}.")
        directive = self.instructions.format(
            task=task,
            agents={
                a.name: {
                    "capabilities": [cap["name"] for cap in a.config["capabilities"]],
                    "role": a.config["role"], 
                    "data": a.data_headers
                }
                for a in self.agents
            }
        )
        return await self.model.ainvoke(directive)

    async def send_directives(self, intent: CEOutput):
        """Send directives to MCP"""
        mcp_endpoint = await get_mcp_endpoint(self.client, "send_directive")
        results = []

        logger.info("Sending directives to agents:")
        for d in intent.directives:
            directive = d.model_dump()
            logger.info(
                f"Directive for {d.agent}: {d.task} Capability to use: {d.capability}"
            )
            envelope = MCPEnvelope(
                message_type="directive",
                sender="CentralExecutive",
                target=d.agent,
                payload=directive
            )
            res = await mcp_endpoint.ainvoke({"envelope": envelope.model_dump()})
            results.append(res)

        return results
    
    async def collect_agent_feedback(self):
        mcp_endpoint = await get_mcp_endpoint(self.client, "list_messages")
        if not mcp_endpoint:
            return {}

        messages = await mcp_endpoint.ainvoke({})

        results = {}
        for msg in messages:
            parsed = json.loads(msg["text"])
            for entry in parsed:
                if entry.get("message_type") == "agent_feedback":
                    agent = entry["sender"]
                    results.setdefault(agent, []).append(entry["payload"])

        return results
