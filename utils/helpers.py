import yaml
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = "config"


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