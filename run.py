import asyncio

from langchain_core.messages import AIMessage

from agents.central_executive import CentralExecutive
from agents.worker_agent import WorkerAgent
from utils.helpers import load_config, get_mcp_client, setup_logger, load_model

TASK = "Reduce operational costs by 10% without impacting delivery timelines"

logger = setup_logger()


class MangoOperator:
    def __init__(self):
        self.mcp_client = get_mcp_client()
        self.config = load_config("agents.yaml")["agents"]
        self.model = load_model()

        self.ce = CentralExecutive(self.model, self.mcp_client)
        self.agents = [WorkerAgent(a, self.config[a], self.model, self.mcp_client) for a in self.config.keys() \
            if a != "CentralExecutive"]
        logger.info(f"Initialized MangoOperator with agents: {[a.name for a in self.agents]}")

    async def operational_loop(self, task):
        operation_result = []  
        ce_directive = self.config["CentralExecutive"]["instructions"].format(task=task, 
            agents=[a.name for a in self.agents])
        
        logger.info(f"Sending directive to CE: {ce_directive}")
        ce_output = await self.ce.generate_directives(ce_directive)
        result = await self.ce.send_directives(ce_output)

        logger.info(f"Waiting for agents to perform directives: {ce_output}")
        # Agents will read MCP server messages and pick directives
        for agent in self.agents:
            agent_response = await agent.process_agent_directive()
            logger.info(f"Operational results for agent {agent.name}: {agent_response}")
            operation_result.append(agent_response)

        return operation_result

if __name__ == "__main__":
    mango = MangoOperator()
    operation_result = asyncio.run(mango.operational_loop("Reduce operational costs by 10% without impacting delivery timelines"))
