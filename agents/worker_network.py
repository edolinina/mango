from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from typing import TypedDict, List, Literal
from typing_extensions import Annotated
from pydantic import BaseModel

SUBTASKS = ["predict", "analyze", "classify", "learn"]


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict):
    task: str
    plan: list
    model: any
    tools: list
    results: Annotated[dict, merge_dicts]


class Plan(BaseModel):
    steps: List[str]


# --- Node implementations ---
async def llm_planner(state: AgentState) -> AgentState:
    """LLM planner node that produces a Plan object."""
    model = state["model"].with_structured_output(Plan)
    output_schema = Plan.model_json_schema()

    answer = await model.ainvoke(f"Based on these available tools: {state['tools']}, \
        decide about the steps required to accomplish the task: {state['task']}. \
        The steps should be chosen from {SUBTASKS}. \
        The output should be a JSON object matching this schema: {output_schema}")
    state["plan"] = [s for s in answer.steps if s in SUBTASKS]
    return state

def predictor_node(state):
    return {"results": {"predict": {"predictions": ["ok"]}}}

def llm_node(state):
    return {"results": {"analyze": {"analysis": "insights"}}}

def classifier_node(state):
    return {"results": {"classify": {"classes": ["A", "B"]}}}

def rl_node(state):
    return {"results": {"learn": {"policy": "updated"}}}

def aggregate_node(state: AgentState) -> AgentState:
    return state


# --- Graph ---
def route_steps(state: AgentState):
    return state["plan"]

def build_agent_network() -> StateGraph:
    """Builds and returns the agent's state graph."""
    graph = StateGraph(AgentState)

    graph.add_node("plan", llm_planner)
    graph.add_node("predict", predictor_node)
    graph.add_node("analyze", llm_node)
    graph.add_node("classify", classifier_node)
    graph.add_node("learn", rl_node)
    graph.add_node("aggregate", aggregate_node)

    # Fan-out
    graph.add_conditional_edges("plan", route_steps)

    # Fan-in (ALL go to aggregate)
    for step in SUBTASKS:
        graph.add_edge(step, "aggregate")

    graph.set_entry_point("plan")
    graph.set_finish_point("aggregate")

    return graph

async def run_agent_network(state: AgentState):
    """Runs the agent network asynchronously and returns the final state."""
    graph = build_agent_network()

    app = graph.compile()

    results = await app.ainvoke(state)
    return results