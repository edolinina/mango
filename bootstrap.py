from agents.central_executive import CentralExecutive
from agents.worker_agent import WorkerAgent
from utils.helpers import load_config, load_model, get_mcp_client
from utils.knowledge import get_knowledge_retriever

def init_system():
    mcp_client = get_mcp_client()
    config = load_config("agents.yaml")

    model = load_model()
    knowledge_retriever = get_knowledge_retriever()

    agents_cfg = config["agents"]
    common_prompt = config["common"]["reasoning"]["prompt"]

    # build worker agents
    agents = []
    for a in agents_cfg:
        if a["name"] == "CentralExecutive":
            continue
        agent = WorkerAgent(
            a,
            model,
            mcp_client,
            knowledge_retriever,
        )
        agent.config["instructions"] = common_prompt
        agents.append(agent)

    # build CE
    ce_cfg = next(a for a in agents_cfg if a["name"] == "CentralExecutive")
    ce = CentralExecutive(
        model,
        mcp_client,
        ce_cfg,
        agents,
    )

    return {
        "ce": ce,
        "agents": agents,
        "agents_by_name": {a.name: a for a in agents},
    }
