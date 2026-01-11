import os
import yaml
import logging
import httpx

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = "config"


class color:
    BOLD = '\033[1m'
    END = '\033[0m'

def bold_str(text: str):
    return f"{color.BOLD}{text}{color.END}"

def load_config(config_file) -> dict:
    """
    Load YAML configuration file
    """
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

def load_model():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "ollama":
        return ChatOllama(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
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
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
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
    tools = await mcp_client.get_tools()
    return next((t for t in tools if t.name == endpoint), None)

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
