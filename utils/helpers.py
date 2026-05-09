import os
import yaml
import logging
import httpx
import time
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = "config"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")
MCP_TOOLS_CACHE_TTL_SECONDS = float(os.getenv("MCP_TOOLS_CACHE_TTL_SECONDS", "300"))

# Cache MCP tools per client instance to avoid repeated list/get_tools round-trips.
_MCP_TOOLS_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}

class color:
    BOLD = '\033[1m'
    END = '\033[0m'

def bold_str(text: str):
    return f"{color.BOLD}{text}{color.END}"

def load_config(config_file) -> dict:
    data = {}
    config_path = f"{CONFIG_PATH}/{config_file}"
    try:
        with open(config_path) as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"Unexpected error loading spec from '{config_path}'")
        raise

def is_agent_enabled(agent_config: dict) -> bool:
    return agent_config.get("enabled", True)

def load_model(provider=LLM_PROVIDER, model=LLM_MODEL):
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "ollama":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=model,
            temperature=temperature,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set"
            )

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

def get_mcp_client():
    mcp_host = os.getenv("MCP_HOST", "localhost")
    mcp_port = os.getenv("MCP_PORT", 8000)
    return MultiServerMCPClient(
        {
            "mango": {
                "transport": "http",
                "url": f"http://{mcp_host}:{mcp_port}/mcp",
            }
        }
    )

async def get_mcp_endpoint(mcp_client, endpoint):
    # Cached endpoint lookup with TTL to reduce MCP get_tools() round-trips.
    try:
        client_key = id(mcp_client)
        now = time.monotonic()
        cached = _MCP_TOOLS_CACHE.get(client_key)

        if cached:
            cached_at, tools_by_name = cached
            if (now - cached_at) < MCP_TOOLS_CACHE_TTL_SECONDS:
                return tools_by_name.get(endpoint)

        tools = await mcp_client.get_tools()
        tools_by_name = {t.name: t for t in tools}
        _MCP_TOOLS_CACHE[client_key] = (now, tools_by_name)
        return tools_by_name.get(endpoint)
    except httpx.ConnectError as e:
        logger = logging.getLogger("mango")
        logger.error(f"Failed to connect to MCP service: {e}. Check MCP_HOST and MCP_PORT configuration.")
        return None
    except Exception as e:
        logger = logging.getLogger("mango")
        logger.error(f"Error retrieving MCP endpoint '{endpoint}': {e}")
        return None

def setup_logger(name: str = "mango", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
