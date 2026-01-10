import json
import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_agent_app(agent):
    app = FastAPI()

    @app.post("/process-directive")
    async def process_directive():
        reports = await agent.process_agent_directive()

        for report in reports:
            results = json.loads(report.results)
            logger.info(f"Agent {agent.name} results: {results}")

        return {
            "agent": agent.print_name,
            "reports": [r.model_dump() for r in reports],
        }

    return app
