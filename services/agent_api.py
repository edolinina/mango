import json
import asyncio
import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_agent_app(agent):
    app = FastAPI()
    active_run = {"task": None}

    async def _run_agent_directives():
        try:
            reports = await agent.process_agent_directive()
            if not reports:
                logger.info("Agent %s returned no reports.", agent.name)
                return

            for report in reports:
                results = json.loads(report.results)
                logger.info("Agent %s results: %s", agent.name, results)
        except Exception:
            logger.exception("Agent %s directive processing failed", agent.name)
        finally:
            active_run["task"] = None

    @app.post("/process-directive")
    async def process_directive():
        current = active_run["task"]
        if current and not current.done():
            return {
                "agent": agent.print_name,
                "status": "already_running",
            }

        active_run["task"] = asyncio.create_task(_run_agent_directives())
        return {
            "agent": agent.print_name,
            "status": "accepted",
        }

    return app
