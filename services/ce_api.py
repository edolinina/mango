import os
import asyncio
import httpx
import logging
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
static_files = StaticFiles(directory="static")


# ---------- async workflow ----------
async def run_workflow_async(ce, directive_id):
    try:
        logger.info("Starting workflow")

        # send directives to MCP
        await ce.send_directives(ce.directives)
        logger.info("Directives sent")

        # give agents MORE time to start
        await asyncio.sleep(10)  # Increased from 5 to 10 seconds

        # trigger agents with retry logic
        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = []
            for agent in ce.agents:
                url = f"http://{agent.host}:{agent.port}/process-directive"
                logger.info(f"Triggering agent {agent.name} at {url}")
                tasks.append(trigger_agent_with_retry(client, url, agent.name, max_retries=5))

            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All agents triggered")

    except Exception:
        logger.exception("Workflow execution failed")


async def trigger_agent_with_retry(client, url, agent_name, max_retries=5):
    """Trigger agent with retry on connection errors."""
    for attempt in range(max_retries):
        try:
            await client.post(url, json={"directive_id": None})
            logger.info(f"Successfully connected to {agent_name}")
            return
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Connection to {agent_name} failed, retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Failed to connect to {agent_name} after {max_retries} attempts")
                raise


# ---------- background wrapper ----------
def run_workflow_bg(ce):
    asyncio.run(run_workflow_async(ce, directive_id=None))


# ---------- FastAPI app ----------
def create_ce_app(ce):
    app = FastAPI()

    class TaskRequest(BaseModel):
        task: str

    # ---------- UI ----------
    @app.get("/")
    async def ui(request: Request):
        app.mount("/static", static_files, name="static")
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )

    # ---------- Orchestrator entry ----------
    @app.post("/run")
    async def run(req: TaskRequest, background: BackgroundTasks):
        logger.info(f"Generating directives for CE task: {req.task}")

        directives = await ce.generate_directives(req.task)
        ce.update_directives(directives)

        if os.getenv("AUTONOMOUS_MODE", "false").lower() != "true":
            directives_text = "\n".join(
                f"- {d.agent}: {d.task} (Capability: {d.capability})"
                for d in directives.directives
            )
            return {
                "status": "pending",
                "directives": directives_text,
            }

        background.add_task(run_workflow_bg, ce)

        return {
            "status": "running",
            "agents": [agent.print_name for agent in ce.agents],
        }
    
    # ---------- Approve directives ----------
    @app.post("/approve")
    async def approve(background: BackgroundTasks):
        logger.info(f"Running approved directives for CE task")
        background.add_task(run_workflow_bg, ce)
        return {
            "status": "running",
            "agents": [agent.print_name for agent in ce.agents],
        }
    
    # ---------- Reject directives ----------
    @app.post("/reject")
    async def reject():
        logger.info(f"Cancelling directives for CE task")
        ce.update_directives(None)
        return {
            "status": "rejected",
        }


    # ---------- Results polling ----------
    @app.get("/results")
    async def results():
        return await ce.collect_agent_feedback()

    return app
