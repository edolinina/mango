import yaml
import logging

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = "config"

# Ollama model configuration
OLLAMA_BASE_URL = "http://192.168.1.120:11434"
OLLAMA_MODEL_NAME = "gpt-oss:20b" # llm model
OLLAMA_MODEL_TEMPERATURE = 0.2 # for more deterministic responses
OLLAMA_MODEL_SEED = 42 # for repetitive responses
OLLAMA_MODEL_GPUS = 1
OLLAMA_MODEL_THREADS = 20

def load_model() -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL_NAME,
        temperature=OLLAMA_MODEL_TEMPERATURE
    )

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

def get_mcp_client():
    mcp_config = load_config("base.yaml")["mcp"]
    return MultiServerMCPClient(
        {
            "mango": {
                "transport": mcp_config["schema"],
                "url": f"{mcp_config["schema"]}://{mcp_config["host"]}:{mcp_config["port"]}/mcp",
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