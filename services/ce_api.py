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
async def run_workflow_async(ce, directives):
    try:
        logger.info("Starting workflow")

        # send directives to MCP
        await ce.send_directives(directives)
        logger.info("Directives sent")

        # give agents time to start
        await asyncio.sleep(3)

        # trigger agents
        async with httpx.AsyncClient() as client:
            tasks = []
            for agent in ce.agents:
                url = f"http://{agent.host}:{agent.port}/process-directive"
                logger.info(f"Triggering agent {agent.name} at {url}")
                tasks.append(client.post(url))

            await asyncio.gather(*tasks)

        logger.info("All agents triggered")

    except httpx.ReadTimeout:
        logger.warning("A request to an agent timed out")

    except Exception:
        logger.exception("Workflow execution failed")


# ---------- background wrapper ----------
def run_workflow_bg(ce, directives):
    asyncio.run(run_workflow_async(ce, directives))


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

        # MUST use sync wrapper for BackgroundTasks
        background.add_task(run_workflow_bg, ce, directives)

        return {
            "status": "running",
            "agents": [agent.print_name for agent in ce.agents],
        }

    # ---------- Results polling ----------
    @app.get("/results")
    async def results():
        return await ce.collect_agent_feedback()

    return app
