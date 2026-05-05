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


def _format_exception(exc: Exception) -> str:
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}: {repr(exc)}"


async def run_workflow_async(ce):
    try:
        logger.info("Starting workflow")
        ce.clear_workflow_error()

        try:
            await ce.send_directives(ce.directives)
            logger.info("Directives sent")
        except Exception as e:
            logger.error(f"Failed to send directives: {e}")
            ce.set_workflow_error(f"Failed to send directives: {e}")
            return

        await asyncio.sleep(10)

        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = []
            for agent in ce.agents:
                url = f"http://{agent.host}:{agent.port}/process-directive"
                logger.info(f"Triggering agent {agent.name} at {url}")
                tasks.append(trigger_agent_with_retry(client, url, agent.name, max_retries=5))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [_format_exception(result) for result in results if isinstance(result, Exception)]
            if failures:
                ce.set_workflow_error("; ".join(failures))
                logger.error(f"Agent trigger failures: {ce.workflow_error}")
                return

        logger.info("All agents triggered")

    except Exception as e:
        ce.set_workflow_error(f"Workflow execution failed: {e}")
        logger.exception("Workflow execution failed")


async def trigger_agent_with_retry(client, url, agent_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = await client.post(url, json={"directive_id": None})
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"{agent_name} returned HTTP {response.status_code}: {response.text}")
            logger.info(f"Successfully connected to {agent_name}")
            return
        except (httpx.ConnectError, RuntimeError) as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Connection to {agent_name} failed ({exc}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"Failed to connect to {agent_name} after {max_retries} attempts: {exc}")
                raise


def run_workflow_bg(ce):
    asyncio.run(run_workflow_async(ce))


def create_ce_app(ce):
    app = FastAPI()
    app.mount("/static", static_files, name="static")

    class TaskRequest(BaseModel):
        task: str

    @app.get("/")
    async def ui(request: Request):
        return templates.TemplateResponse(request, "index.html")

    @app.post("/run")
    async def run(req: TaskRequest, background: BackgroundTasks):
        logger.info(f"Generating directives for CE task: {req.task}")
        ce.clear_workflow_error()

        try:
            directives = await ce.generate_directives(req.task)
        except httpx.ConnectError as e:
            logger.error(f"LLM connection failed: {e}")
            return {
                "status": "error",
                "error": "LLM service unavailable. Check OLLAMA_URL or ensure Ollama is running.",
            }
        except Exception as e:
            logger.error(f"Failed to generate directives: {e}")
            return {
                "status": "error",
                "error": f"Failed to generate directives: {str(e)}",
            }

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

    @app.post("/approve")
    async def approve(background: BackgroundTasks):
        logger.info(f"Running approved directives for CE task")
        ce.clear_workflow_error()
        background.add_task(run_workflow_bg, ce)
        return {
            "status": "running",
            "agents": [agent.print_name for agent in ce.agents],
        }
    
    @app.post("/reject")
    async def reject():
        logger.info(f"Cancelling directives for CE task")
        ce.update_directives(None)
        ce.clear_workflow_error()
        return {
            "status": "rejected",
        }


    @app.get("/results")
    async def results():
        try:
            if ce.workflow_error:
                return {"status": "error", "error": ce.workflow_error}
            feedback = await ce.collect_agent_feedback()
            if not feedback:
                return {"status": "no_results", "agents": []}
            return feedback
        except Exception as e:
            logger.error(f"Error fetching results: {e}")
            return {"status": "error", "error": "Failed to fetch results"}

    return app
