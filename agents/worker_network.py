from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from typing import TypedDict, List, Any
from typing_extensions import Annotated


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

class AgentState(TypedDict):
    task: str
    model: Any
    tools: List[dict]
    results: Annotated[dict, merge_dicts]


# --- Nodes implementations ---
def predictor_node(state):
    return {"results": {"predictor": {"predictions": ["ok"]}}}

def llm_node(state):
    return {"results": {"llm": {"analysis": "insights"}}}

def classifier_node(state):
    return {"results": {"classifier": {"classes": ["A", "B"]}}}

def rl_node(state):
    return {"results": {"learner": {"policy": "updated"}}}

def aggregate_node(state: AgentState) -> AgentState:
    return state

def start_node(state: AgentState) -> AgentState:
    return state

# --- Graph ---
def build_agent_network(tools) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("start", start_node)

    # Tools nodes
    if "predictor" in tools:
        graph.add_node("predictor", predictor_node)
    if "llm" in tools:
        graph.add_node("llm", llm_node)
    if "classifier" in tools:
        graph.add_node("classifier", classifier_node)
    if "learner" in tools:
        graph.add_node("learner", rl_node)

    graph.add_node("aggregate", aggregate_node)

    # Entry → all tools (fan-out)
    for node in tools:
        graph.add_edge("start", node)
        graph.add_edge(node, "aggregate")

    graph.set_entry_point("start")
    graph.set_finish_point("aggregate")

    return graph


async def run_agent_network(state: AgentState):
    """Runs the agent network asynchronously and returns the final state."""
    tools = [t["type"] for t in state.get("tools", [])]
    graph = build_agent_network(tools)
    app = graph.compile()

    results = await app.ainvoke(state)
    return results