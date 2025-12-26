import asyncio

from langchain_core.messages import AIMessage

from core.central_executive import CentralExecutive
from agents.llm_agent import LLMAgent
from utils.helpers import load_config, get_mcp_client
from utils.logging import setup_logger


TASK = "Reduce operational costs by 10% without impacting delivery timelines"
AGENTS = ["FinanceAgent", "OpsAgent", "InfraAgent"]

logger = setup_logger()


class MangoOperator:
    def __init__(self):
        self.mcp_client = get_mcp_client()
        self.ce = CentralExecutive(self.mcp_client)

        self.agents = [LLMAgent(a, self.mcp_client) for a in AGENTS]
        self.config = load_config("agents.yaml")

    async def operational_loop(self, task):
        operation_result = []  
        ce_directive = self.config["central-executive"]["instructions"].format(task=task, 
            agents=[a.name for a in self.agents])
        
        logger.info(f"Sending directive to CE: {ce_directive}")
        ce_output = await self.ce.generate_directives(ce_directive)
        result = await self.ce.send_directives(ce_output)

        logger.info(f"Sending directive to Agents: {ce_output}")
        # Agents will read MCP server messages themselves and pick directives
        for agent in self.agents:
            agent_response = await agent.handle_directive()
            logger.info(f"Operational results for agent {agent.name}: {agent_response}")
            operation_result.append(agent_response)

        return operation_result

if __name__ == "__main__":
    mango = MangoOperator()
    operation_result = asyncio.run(mango.operational_loop("Reduce operational costs by 10% without impacting delivery timelines"))
