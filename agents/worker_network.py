import re
import os
import joblib
import logging
import pandas as pd

from typing import TypedDict, List, Any, Optional

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

logger = logging.getLogger("mango")

class AgentState(TypedDict):
    prompt: str
    task: str
    constraints: str
    validator: dict
    model: Any
    context: str
    data_source: str
    results: dict
    validation: dict

class Recommendation(TypedDict):
    recommendation: str
    explanation: str
    validation_samples: List[dict]


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


def aggregate_node(state: AgentState) -> AgentState:
    return state


# --- Graph ---
def build_agent_network(state) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("load_validator", load_validator)
    graph.add_node("llm", llm_node)
    graph.add_node("validate", validation_node)
    graph.add_node("aggregate", aggregate_node)

    if "validator" in state:
        graph.set_entry_point("load_validator")
        graph.add_edge("load_validator", "llm")
        graph.add_edge("llm", "validate")
        graph.add_edge("validate", "aggregate")
    else:
        graph.set_entry_point("llm")
        graph.add_edge("llm", "aggregate")

    graph.set_finish_point("aggregate")
    return graph


async def run_agent_network(state: AgentState):
    """Runs the agent network asynchronously and returns the final state."""
    graph = build_agent_network(state)
    app = graph.compile()

    results = await app.ainvoke(state)
    return results