import re
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
def validator_training_node(state):
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

    X = df[features]
    y = df[target]
    mean_y = df[target].mean()
    min_y = df[target].min()
    max_y = df[target].max()

    pass_condition = pass_condition.replace("MEAN", str(mean_y))
    pass_condition = pass_condition.replace("MIN", str(min_y))
    pass_condition = pass_condition.replace("MAX", str(max_y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # classification vs regression based on target type
    if y.dtype.kind in {"i", "b"} and y.nunique() <= 10:
        engine = LogisticRegression(max_iter=100)
    else:
        engine = LinearRegression()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("engine", engine)
    ])

    model.fit(X_train, y_train)
    state["validator"]["model"] = model
    state["validator"]["pass_condition"] = pass_condition
    return state


async def llm_node(state):
    model = state["model"].with_structured_output(Recommendation) 
    data_source = state["data_source"]
    if data_source:
        with open(data_source, 'r') as f:
            data = f.read()

    validation_features = state.get("validator", {}).get("features", [])
    answer = await model.ainvoke(
        f"TASK: {state["task"]}\n"
        f"{state["prompt"]}\n"
        f"CONSTRAINTS: {state['constraints']}\n"
        f"ALLOWED_FEATURES: {validation_features}\n"
        f"DATA: {data}\n"
        f"CONTEXT: {state["context"]}"
    )

    results = {"results": answer}
    validation_results = []
    if "validator" in state:
        model = state["validator"]["model"]
        pass_condition = state["validator"]["pass_condition"]
        if re.match(r"^[<>=!]=?\s*\d+(\.\d+)?$", pass_condition) is None:
            raise ValueError(f"Invalid pass_condition format: {pass_condition}")

        samples = answer.get("validation_samples", [])
        df_val = pd.DataFrame(samples)

        expected = set(validation_features)
        if set(df_val.columns) != expected:
            raise ValueError(
                f"Invalid validation_samples schema.\n"
                f"Expected: {expected}\n"
                f"Got: {df_val.columns.tolist()}\n"
                f"Raw: {samples}"
            )

        X_val = df_val[validation_features]
        preds = model.predict(X_val).tolist()
        for pred in preds:
            if eval(f"{pred} {pass_condition}"):
                validation_results.append(True)
            else:
                validation_results.append(False)

        passed = sum(validation_results)
        failed = len(validation_results) - passed
    
        results["validation"] = {
            "passed": passed,
            "failed": failed
        }

    return results

def reinforcement_learning_node(state):
    return {"results": {"learner": {"policy": "updated"}}}

def aggregate_node(state: AgentState) -> AgentState:
    return state


# --- Graph ---
def build_agent_network(state) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("aggregate", aggregate_node)

    if "validator" in state:
        graph.add_node("validator", validator_training_node)
        graph.set_entry_point("validator")
        graph.add_edge("validator", "llm")
    else:
        graph.set_entry_point("llm")

    graph.set_finish_point("aggregate")
    return graph


async def run_agent_network(state: AgentState):
    """Runs the agent network asynchronously and returns the final state."""
    graph = build_agent_network(state)
    app = graph.compile()

    results = await app.ainvoke(state)
    return results