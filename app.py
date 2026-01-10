from contextlib import asynccontextmanager
import os
from fastapi import FastAPI

from bootstrap import init_system
from services.ce_api import create_ce_app
from services.agent_api import create_agent_app

SERVICE_MODE = os.getenv("SERVICE_MODE", "ce")
AGENT_NAME = os.getenv("AGENT_NAME")

system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    system = init_system()

    if SERVICE_MODE == "ce":
        ce_app = create_ce_app(system["ce"])
        app.mount("/", ce_app)

    elif SERVICE_MODE == "agent":
        if not AGENT_NAME:
            raise RuntimeError("AGENT_NAME must be set")

        agent = system["agents_by_name"].get(AGENT_NAME)
        if not agent:
            raise RuntimeError(f"Unknown agent: {AGENT_NAME}")

        agent_app = create_agent_app(agent)
        app.mount("/", agent_app)

    yield  # Startup done, app runs

app = FastAPI(lifespan=lifespan)