import asyncio

from langchain_core.messages import AIMessage
from core.central_executive import CentralExecutive
from utils.helpers import load_config, get_mcp_client
from utils.logging import setup_logger


TASK = "Reduce operational costs by 10% without impacting delivery timelines"
AGENTS = ["FinanceAgent", "OpsAgent", "InfraAgent"]

logger = setup_logger()


class MangoOperator:
    def __init__(self):
        self.mcp_client = get_mcp_client()
        self.ce = CentralExecutive(self.mcp_client)
        self.agents = AGENTS
        self.config = load_config("agents.yaml")

    async def decision_making_loop(self, task):        
        ce_directive = self.config["central-executive"]["instructions"].format(task=task, 
            agents=self.agents)
        
        logger.info(f"Sending directive to CE: {ce_directive}")
        ce_output = await self.ce.generate_directives(ce_directive)
        result = await self.ce.send_directives(ce_output)

        return result

if __name__ == "__main__":
    mango = MangoOperator()
    
    decision = asyncio.run(mango.decision_making_loop("Reduce operational costs by 10% without impacting delivery timelines"))
    print(decision)
