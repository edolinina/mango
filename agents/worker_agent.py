import os
import logging
import json
import asyncio

from mcp_server.protocol import AgentOutput
from agents.worker_network import run_agent_network
from utils.helpers import get_mcp_endpoint

logger = logging.getLogger("mango")
TRAINED_MODELS_PATH = os.getenv("TRAINED_MODELS_PATH", "models")

class WorkerAgent:
    def __init__(self, config, model, mcp_client, knowledge_retriever, manager_name="CentralExecutive"):
        self.client = mcp_client
        self.manager = manager_name
        self.model = model
        self.config = config
        self.avatar = config.get("avatar", "")
        self.name = config["name"]
        self.print_name = f"{config["name"]} {self.avatar}"
        self.host = config.get("host", "localhost")
        port_env = config.get("port-env")
        self.port = os.getenv(port_env, "8000")
        self.data_source = self.config.get("data_source")
        with open(self.data_source, 'r') as f:
            self.data_headers = next(f).strip()

        self.knowledge_retriever = knowledge_retriever

    def get_context(self, task):
        query = f"Provide context for decision making on this task: {task}"
        docs = self.knowledge_retriever.invoke(query)

        context = "\n\n".join(
            f"### {d.metadata.get('source', '')}\n{d.page_content}"
            for d in docs
        )
        return context

    async def process_agent_directive(self):
        mcp_endpoint = await get_mcp_endpoint(self.client, "list_messages")
        if not mcp_endpoint:
            logger.info("list_messages tool not available on MCP client")
            return None
        
        directives = {}
        messages = await mcp_endpoint.ainvoke({})
        for msg in messages:
            parsed = json.loads(msg['text'])
            for entry in parsed:
                if entry.get('message_type') == 'directive' and entry.get('target') == self.name:
                    directives[entry.get('message_id')] = entry.get('payload')

        if not directives:
            logger.info(f"No directives found for agent {self.name}")
            return None

        # Build input states for all directives
        input_states = []  # list of (message_id, cap, directive, input_state)
        for _id, directive in directives.items():
            capability = directive.get("capability", "")
            if capability not in [cap["name"] for cap in self.config["capabilities"]]:
                logger.info(f"Capability {capability} not found in agent {self.name}")
                continue

            for cap in self.config["capabilities"]:
                if cap["name"] != capability:
                    continue

                task = directive.get("task")
                validator = cap.get("validator")
                constraints = "\n".join([f"{v['target']} {v['pass_condition']}"
                    for v in self.config.get("validators", [])])

                logger.info(f"Agent {self.print_name} found directive for {capability}: {task}")
                context = self.get_context(task)
                input_state = {
                    "prompt": self.config["instructions"],
                    "task": task,
                    "context": context,
                    "model": self.model,
                    "data_source": self.data_source,
                    "constraints": constraints,
                    # carry metadata for result aggregation
                    "cap": cap,
                    "directive": directive,
                    "_id": _id,
                }

                if validator:
                    validator_config = [v for v in self.config.get("validators", []) if v["name"] == validator]
                    if not validator_config:
                        logger.info(f"Validator {validator} not found for agent {self.name}")
                        continue
                    validator_config = validator_config[0]
                    input_state["validator"] = validator_config
                    input_state["validator"]["model_path"] = os.path.join(
                        TRAINED_MODELS_PATH, self.name, f"{validator_config['name']}.pkl"
                    )

                input_states.append(input_state)

        if not input_states:
            return None

        # Run ONE network covering all directives in parallel
        result_states = await run_agent_network(input_states)

        # Aggregate results and send feedback
        response = []
        for result_state in result_states:
            cap = result_state.get("cap")
            directive = result_state.get("directive")
            _id = result_state.get("_id")
            results = json.dumps(result_state.get("results"))
            validation = result_state.get("validation", {})

            validation_results = ""
            if validation:
                passed = validation.get("passed", 0)
                failed = validation.get("failed", 0)
                pass_rate = passed / (passed + failed)
                if pass_rate == 1.0:
                    validation_results = "✅ success"
                elif pass_rate >= 0.7:
                    validation_results = f"🟡 partial pass ({passed} passed, {failed} failed)"
                else:
                    validation_results = f"❌ fail ({passed} passed, {failed} failed)"

            result = AgentOutput(
                agent=self.name,
                capability=f"{cap['name']} {cap['avatar']}",
                validation=validation_results,
                results=results,
                task_id=directive.get("task_id", "")
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

            mcp_endpoint = await get_mcp_endpoint(self.client, "remove_directive")
            await mcp_endpoint.ainvoke({"id": _id})

        return response
