from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")

def create_orchestrator_app(ce):
    app = FastAPI()

    class TaskRequest(BaseModel):
        task: str

    # ---------- UI ----------
    @app.get("/")
    async def ui(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )

    # ---------- Orchestrator entry ----------
    @app.post("/run")
    async def run(req: TaskRequest):
        directives = await ce.generate_directives(req.task)
        results = await ce.send_directives(directives)
        return {
            "status": "completed",
            "directives": len(directives.directives),
            "results": results,
        }

    # ---------- CE-compatible API ----------
    @app.post("/generate-directives")
    async def generate_directives(req: TaskRequest):
        directives = await ce.generate_directives(req.task)
        await ce.send_directives(directives)
        return {
            "status": "sent",
            "count": len(directives.directives),
        }

    return app
