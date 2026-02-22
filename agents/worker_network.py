import re
import os
import joblib
import logging
import pandas as pd
import asyncio
from typing import TypedDict, List, Annotated
import operator

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger("mango")

class Recommendation(TypedDict):
    recommendation: str
    explanation: str
    validation_samples: List[dict]

class AgentState(TypedDict):
    directives: List[dict]
    results: Annotated[List[dict], operator.add]


# --- Nodes implementations ---
def load_validator(state):
    data_source = state["data_source"]
    validator = state["validator"]
    target = validator["target"]
    features = validator["features"]
    pass_condition = validator["pass_condition"]

    df = pd.read_csv(data_source)
    df.columns = df.columns.str.strip()

    missing = set(features + [target]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema mismatch in {data_source}. "
            f"Missing: {missing}, Available: {df.columns.tolist()}"
        )

    mean_y = df[target].mean()
    min_y = df[target].min()
    max_y = df[target].max()

    pass_condition = pass_condition.replace("MEAN", str(mean_y))
    pass_condition = pass_condition.replace("MIN", str(min_y))
    pass_condition = pass_condition.replace("MAX", str(max_y))

    model_path = validator.get("model_path")
    if not os.path.exists(model_path):
        raise ValueError(f"Pretrained model not found: {model_path}")

    state["validator"]["model"] = joblib.load(model_path)
    state["validator"]["pass_condition"] = pass_condition
    return state


async def llm_node(state):
    model = state["model"].with_structured_output(Recommendation)

    with open(state["data_source"], "r") as f:
        data = f.read()

    validation_features = state.get("validator", {}).get("features", [])

    answer = await model.ainvoke(
        f"TASK: {state['task']}\n"
        f"{state['prompt']}\n"
        f"CONSTRAINTS: {state['constraints']}\n"
        f"ALLOWED_FEATURES: {validation_features}\n"
        f"DATA: {data}\n"
        f"CONTEXT: {state['context']}"
    )

    state["results"] = answer
    return state


def validation_node(state):
    if "validator" not in state:
        return state

    validator = state["validator"]
    model = validator["model"]
    pass_condition = validator["pass_condition"]

    samples = state["results"].get("validation_samples", [])
    df_val = pd.DataFrame(samples)

    features = validator["features"]
    expected = set(features)

    if set(df_val.columns) != expected:
        raise ValueError(
            f"Invalid validation_samples schema.\n"
            f"Expected: {expected}\n"
            f"Got: {df_val.columns.tolist()}"
        )

    X_val = df_val[features]
    preds = model.predict(X_val).tolist()

    validation_results = [
        eval(f"{p} {pass_condition}") for p in preds
    ]

    passed = sum(validation_results)
    failed = len(validation_results) - passed

    state["validation"] = {
        "passed": passed,
        "failed": failed
    }

    return state


async def run_directive_branch(directive_state: dict) -> dict:
    """Run llm (+ optional validator) for a single directive and return result."""
    if "validator" in directive_state:
        directive_state = load_validator(directive_state)
    directive_state = await llm_node(directive_state)
    if "validator" in directive_state:
        directive_state = validation_node(directive_state)

    return {
        "cap": directive_state.get("cap"),
        "directive": directive_state.get("directive"),
        "_id": directive_state.get("_id"),
        "results": directive_state.get("results"),
        "validation": directive_state.get("validation"),
    }


def build_agent_network(directives: list) -> StateGraph:
    """Build a LangGraph for visualization showing all directive branches."""
    graph = StateGraph(AgentState)
    
    def aggregate_node(state: AgentState) -> AgentState:
        return state
    
    graph.add_node("aggregate", aggregate_node)

    for i, directive in enumerate(directives):
        has_validator = "validator" in directive
        agent_name = directive.get("_agent", f"agent_{i}")
        cap_name = directive.get("cap", {}).get("name", f"directive_{i}")
        branch = f"{agent_name}_{cap_name}"

        if has_validator:
            graph.add_node(f"{branch}_load_validator", lambda s: s)
            graph.add_node(f"{branch}_llm", lambda s: s)
            graph.add_node(f"{branch}_validation", lambda s: s)
            graph.add_edge(START, f"{branch}_load_validator")
            graph.add_edge(f"{branch}_load_validator", f"{branch}_llm")
            graph.add_edge(f"{branch}_llm", f"{branch}_validation")
            graph.add_edge(f"{branch}_validation", "aggregate")
        else:
            graph.add_node(f"{branch}_llm", lambda s: s)
            graph.add_edge(START, f"{branch}_llm")
            graph.add_edge(f"{branch}_llm", "aggregate")

    graph.add_edge("aggregate", END)
    return graph


async def run_agent_network(directives: list):
    """Runs all directive branches in parallel using asyncio.gather."""
    results = await asyncio.gather(
        *[run_directive_branch(d) for d in directives]
    )
    return list(results)