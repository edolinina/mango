import yaml
import logging

from langchain_ollama import ChatOllama
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

def load_model() -> ChatOllama:
    config = load_config("base.yaml")
    ollama_config = config["ollama"]
    return ChatOllama(
        base_url=ollama_config["base_url"],
        model=ollama_config["llm-model"]["name"],
        temperature=ollama_config["llm-model"]["temperature"]
    )

def get_mcp_client():
    mcp_config = load_config("base.yaml")["mcp"]
    return MultiServerMCPClient(
        {
            "mango": {
                "transport": mcp_config["schema"],
                "url": f"{mcp_config['schema']}://{mcp_config['host']}:{mcp_config['port']}/mcp",
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
