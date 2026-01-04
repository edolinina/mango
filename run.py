import json
import asyncio
import traceback

from langchain_core.messages import AIMessage

from agents.central_executive import CentralExecutive
from agents.worker_agent import WorkerAgent
from utils.helpers import *
from utils.knowledge import get_knowledge_retriever

TASK = "Reduce operational costs by 10% without impacting delivery timelines"

logger = setup_logger()


class MangoOperator:
    def __init__(self):
        self.mcp_client = get_mcp_client()
        config = load_config("agents.yaml")
        self.config = config["agents"]
        self.llm_prompt = config["common"]["reasoning"]["prompt"]
        self.model = load_model()
        self.knowledge_retriever = get_knowledge_retriever()
        
        self.agents = [WorkerAgent(a, self.model, self.mcp_client, self.knowledge_retriever) \
            for a in self.config if a["name"] != "CentralExecutive"]
        
        ce_config = [a for a in self.config if a["name"] == "CentralExecutive"][0]
        self.ce = CentralExecutive(self.model, self.mcp_client, ce_config, self.agents)
        
        agents_list = ", ".join([a.name for a in self.agents])
        logger.info(f"Initialized 🥭 MangoOperator with agents: {agents_list}")

    async def operational_loop(self, task):
        operation_result = []
        ce_output = await self.ce.generate_directives(task)
        logger.info(f"Directive for {self.ce.name}: {task}")
        result = await self.ce.send_directives(ce_output)

        logger.info(f"Waiting for agents to perform directives...")
        # Agents will read MCP server messages and pick directives
        agents_tasks = []
        for agent in self.agents:
            agent.config["instructions"] = self.llm_prompt
            agents_tasks.append(agent.process_agent_directive())
        
        all_agent_reports = await asyncio.gather(
            *agents_tasks,
            return_exceptions=True,
        )

        for agent, agent_reports in zip(self.agents, all_agent_reports):
            if isinstance(agent_reports, Exception):
                logger.error(
                    f"Agent {agent.name} failed with exception:\n"
                    f"{''.join(traceback.format_exception(agent_reports))}"
                )
                continue

            if not agent_reports:
                logger.info(f"Agent {agent.name} returned no reports.")
                continue

            logger.info(f"Operational results for agent {agent.name}:")
            for report in agent_reports:
                results = json.loads(report.results)
                logger.info(
                    f"{bold_str('Capability applied')}: {report.capability}\n"
                    f"{bold_str('Recommendation')}: {results['recommendation']}\n"
                    f"{bold_str('Explanation')}: {results['explanation']}\n"
                    f"{bold_str('Validation')}: {report.validation}\n"
                )
                logger.info("==" * 50)

            operation_result.append(agent_reports)

        return operation_result

if __name__ == "__main__":
    mango = MangoOperator()
    operation_result = asyncio.run(mango.operational_loop("Reduce operational costs by 10% without impacting delivery timelines"))
