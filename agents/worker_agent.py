import os
import logging
import json
import asyncio

from schemas import AgentOutput
from agents.worker_network import run_reflection_agent
from utils.helpers import get_mcp_endpoint

logger = logging.getLogger("mango")


class WorkerAgent:
    def __init__(self, config, model, mcp_client, knowledge_retriever, manager_name="CentralExecutive"):
        self.client = mcp_client
        self.manager = manager_name
        self.model = model
        self.config = config
        self.enabled = config.get("enabled", True)
        self.avatar = config.get("avatar", "")
        self.name = config["name"]
        self.print_name = f"{config['name']} {self.avatar}"
        self.host = config.get("host", "localhost")
        port_env = config.get("port-env")
        self.port = os.getenv(port_env, "8000")
        self.data_source = self.config.get("data_source")
        self.knowledge_retriever = knowledge_retriever

    def get_context(self, task):
        query = f"Provide context for decision making on this task: {task}"
        docs = self.knowledge_retriever.invoke(query)
        return "\n\n".join(
            f"### {d.metadata.get('source', '')}\n{d.page_content}"
            for d in docs
        )

    async def process_agent_directive(self):
        logger.info(f"Agent {self.name}: process_agent_directive started")
        if not self.enabled:
            logger.info(f"Agent {self.name} is disabled in config; skipping directive processing")
            return None

        mcp_endpoint = await get_mcp_endpoint(self.client, "list_messages")
        if not mcp_endpoint:
            logger.info("list_messages tool not available on MCP client")
            return None

        directives = {}
        messages = await mcp_endpoint.ainvoke({})
        logger.info(f"Agent {self.name}: fetched {len(messages)} message envelope(s) from MCP")
        for msg in messages:
            parsed = json.loads(msg['text'])
            for entry in parsed:
                if entry.get('message_type') == 'directive' and entry.get('target') == self.name:
                    directives[entry.get('message_id')] = entry.get('payload')

        if not directives:
            logger.info(f"No directives found for agent {self.name}")
            return None

        input_states = []
        for _id, directive in directives.items():
            capability = directive.get("capability", "")
            if capability not in [cap["name"] for cap in self.config["capabilities"]]:
                logger.info(f"Capability {capability} not found in agent {self.name}")
                continue

            for cap in self.config["capabilities"]:
                if cap["name"] != capability:
                    continue

                task = directive.get("task")

                logger.info(f"Agent {self.print_name} found directive for {capability}: {task}")
                data_path = None
                if isinstance(self.data_source, dict):
                    data_path = self.data_source.get("target_file")
                elif isinstance(self.data_source, str):
                    data_path = self.data_source

                validator_features = []
                validator_target = ""
                validator_name = cap.get("validator", "")
                for v in self.config.get("validators", []):
                    if v.get("name") == validator_name:
                        validator_features = v.get("features", [])
                        validator_target = v.get("target", "")
                        break

                input_states.append({
                    "agent_name": self.name,
                    "validator_features": validator_features,
                    "validator_target": validator_target,
                    "prompt": self.config["instructions"],
                    "task": task,
                    "context": self.get_context(task),
                    "model": self.model,
                    "data_path": data_path,
                    "cap": cap,
                    "directive": directive,
                    "_id": _id,
                })

        if not input_states:
            logger.info(f"Agent {self.name}: no runnable input states built from directives")
            return None

        logger.info(f"Agent {self.name}: running reasoning for {len(input_states)} directive state(s)")
        result_states = await run_reflection_agent(input_states)
        logger.info(f"Agent {self.name}: reasoning finished with {len(result_states)} result state(s)")

        response = []
        for result_state in result_states:
            cap = result_state.get("cap")
            directive = result_state.get("directive")
            _id = result_state.get("_id")
            results = json.dumps(result_state.get("results"))
            validation = result_state.get("validation", {})

            result = AgentOutput(
                agent=self.name,
                capability=f"{cap['name']} {cap['avatar']}",
                validation=json.dumps(validation or {}),
                results=results,
                task_id=directive.get("task_id", ""),
            )

            response.append(result)

            mcp_endpoint = await get_mcp_endpoint(self.client, "send_feedback")
            if mcp_endpoint:
                await mcp_endpoint.ainvoke({
                    "envelope": {
                        "message_type": "agent_feedback",
                        "sender": self.name,
                        "target": self.manager,
                        "payload": result.model_dump(),
                    }
                })
                logger.info(f"Agent {self.name}: feedback sent for capability {result.capability}")

            mcp_endpoint = await get_mcp_endpoint(self.client, "remove_directive")
            await mcp_endpoint.ainvoke({"id": _id})
            logger.info(f"Agent {self.name}: directive {_id} removed from MCP queue")

        logger.info(f"Agent {self.name}: process_agent_directive completed")
        return response
