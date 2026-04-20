import json
import os
import joblib
import logging
import pandas as pd
import asyncio
from typing import TypedDict, List, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from utils.helpers import load_model, LLM_PROVIDER, LLM_MODEL, EVALUATION_MODE

logger = logging.getLogger("mango")

class Recommendation(TypedDict):
    recommendation: str
    explanation: str
    validation_samples: List[dict]

class AgentState(TypedDict, total=False):
    # LangGraph control fields
    directives: List[dict]
    results: dict
    # Directive fields
    prompt: str
    task: str
    context: str
    model: object
    data_summary: dict
    constraints: str
    cap: dict
    directive: dict
    _id: str
    validator: dict
    judge_config: dict
    # Output fields
    validation: dict
    llm_judge: dict


class JudgeOutput(TypedDict):
    strategic_alignment: int
    logical_coherence: int
    constraint_awareness: int
    clarity_actionability: int
    overall_score: float
    explanation: str


# --- Nodes implementations ---
def load_validator(state):
    data_summary = state["data_summary"]
    validator = dict(state["validator"])  # shallow copy to avoid mutating input
    target = validator["target"]
    features = validator["features"]
    pass_condition = validator["pass_condition"]

    if not data_summary:
        raise ValueError("No data summary available for validation")
    
    available_columns = list(data_summary.keys())
    missing = set(features + [target]) - set(available_columns)
    if missing:
        raise ValueError(
            f"Schema mismatch in data summary. "
            f"Missing: {missing}, Available: {available_columns}"
        )

    # Get statistics from summary for target variable
    target_stats = data_summary.get(target, {})
    mean_y = target_stats.get('mean', 0)
    min_y = target_stats.get('min', 0)
    max_y = target_stats.get('max', 0)
    
    # Replace placeholders in pass condition
    for percentile in ['PERCENTILE_95', 'PERCENTILE_90', 'PERCENTILE_80', 'PERCENTILE_75', 'PERCENTILE_70', 'PERCENTILE_50', 'PERCENTILE_25']:
        percentile_key = percentile.lower()
        if percentile_key in target_stats:
            pass_condition = pass_condition.replace(percentile, str(target_stats[percentile_key]))
    
    pass_condition = pass_condition.replace("MEAN", str(mean_y))
    pass_condition = pass_condition.replace("MIN", str(min_y))
    pass_condition = pass_condition.replace("MAX", str(max_y))

    model_path = validator.get("model_path")
    if not os.path.exists(model_path):
        raise ValueError(f"Pretrained model not found: {model_path}")
    
    validator["model"] = joblib.load(model_path)

    scaler_path = model_path.replace(".pkl", "_scaler.pkl")
    encoders_path = model_path.replace(".pkl", "_encoders.pkl")

    if os.path.exists(scaler_path):
        validator["scaler"] = joblib.load(scaler_path)

    if os.path.exists(encoders_path):
        validator["encoders"] = joblib.load(encoders_path)

    validator["pass_condition"] = pass_condition

    return {"validator": validator}  # return only the updated key


async def llm_node(state):
    model = state["model"].with_structured_output(Recommendation)

    data_summary = state["data_summary"]
    if isinstance(data_summary, str):
        # If it's a file path, read it
        with open(data_summary, "r") as f:
            data = f.read()
    else:
        # If it's already the data object, convert to JSON string
        data = json.dumps(data_summary, indent=2)

    validation_features = state.get("validator", {}).get("features", [])
    prompt = (
        f"TASK: {state['task']}\n"
        f"{state['prompt']}\n"
        f"CONSTRAINTS: {state['constraints']}\n"
        f"ALLOWED_FEATURES: {validation_features}\n"
        f"DATA: {data}\n"
        f"CONTEXT: {state['context']}"
    )
    
    answer = await model.ainvoke(prompt)
    return {"results": answer}  # return only updated key


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

    missing_features = expected - set(df_val.columns)
    if missing_features:
        raise ValueError(
            f"Invalid validation_samples schema.\n"
            f"Missing required features: {missing_features}\n"
            f"Expected at least: {expected}\n"
            f"Got: {df_val.columns.tolist()}"
        )

    X_val = df_val[features].copy()
    encoders = validator.get("encoders", {})
    scaler = validator.get("scaler")

    # Apply label encoders
    for col, le in encoders.items():
        if col in X_val.columns:
            # Map unseen labels to the closest known class or 0
            def safe_encode(val, le=le):
                try:
                    return le.transform([str(val)])[0]
                except ValueError:
                    # Try numeric fallback: if value is already a valid int index, use it clipped
                    try:
                        idx = int(float(str(val)))
                        return max(0, min(idx, len(le.classes_) - 1))
                    except (ValueError, TypeError):
                        return 0

            X_val[col] = X_val[col].apply(safe_encode)

    # Ensure numeric
    for col in X_val.columns:
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")

    X_val = X_val.fillna(0)

    # Apply scaler
    if scaler:
        X_val = scaler.transform(X_val)

    preds = model.predict(X_val).tolist()
    logger.info(f"Validation predictions: {preds}")

    validation_results = [
        eval(f"{p} {pass_condition}") for p in preds
    ]

    passed = sum(validation_results)
    failed = len(validation_results) - passed
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    if pass_rate == 100:
        status = "success"
    elif pass_rate >= 70:
        status = "partial_pass"
    else:
        status = "fail"

    state["validation"] = {
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": round(pass_rate, 2),
        "status": status
    }

    return {"validation": {
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": round(pass_rate, 2),
        "status": status
    }}


async def judge_node_wrapper(state):
    """Adapter so judge_node can be used as a graph node (reads judge_config from state)."""
    judge_config = state.get("judge_config", {})
    return await judge_node(state, judge_config)


async def judge_node(state, judge_config):
    """Evaluate agent output using LLM judge."""
    judge_provider = judge_config.get("provider", LLM_PROVIDER)
    judge_model_name =  judge_config.get("model", LLM_MODEL)
    judge_model = load_model(judge_provider, judge_model_name)
    judge_model = judge_model.with_structured_output(JudgeOutput)

    answer = state["results"]
    prompt_template = judge_config.get("prompt", "")
    
    data_summary = state.get("data_summary")
    if isinstance(data_summary, str) and os.path.exists(data_summary):
        with open(data_summary, 'r') as f:
            summary_stats = json.load(f)
    else:
        summary_stats = data_summary or "No summary available"
    
    judge_prompt = prompt_template.format(
        task=state.get("task", ""),
        constraints=state.get("constraints", ""),
        data_summary=summary_stats,
        recommendation=answer.get("recommendation", ""),
        explanation=answer.get("explanation", "")
    )

    judgment = await judge_model.ainvoke(judge_prompt)

    # Compute aggregated reasoning score
    scores = [
        judgment["strategic_alignment"],
        judgment["logical_coherence"],
        judgment["constraint_awareness"],
        judgment["clarity_actionability"],
    ]

    judgment["overall_score"] = sum(scores) / len(scores)
    return {"llm_judge": judgment}  # return only updated key


async def run_directive_branch(directive_state: dict) -> dict:
    """Compile and run a per-directive graph, return result."""
    compiled = build_directive_graph(directive_state)
    final_state = await compiled.ainvoke(directive_state)

    return {
        "cap": final_state.get("cap"),
        "directive": final_state.get("directive"),
        "_id": final_state.get("_id"),
        "results": final_state.get("results"),
        "validation": final_state.get("validation"),
        "llm_judge": final_state.get("llm_judge"),
    }


def build_directive_graph(directive: dict):
    """
    Build and compile a LangGraph for a single directive.
    One directive = one graph run.
    """
    graph = StateGraph(AgentState)

    has_validator = "validator" in directive
    has_judge = EVALUATION_MODE and "judge_config" in directive

    if has_validator:
        graph.add_node("load_validator", load_validator)

    graph.add_node("llm", llm_node)

    if has_validator:
        graph.add_node("validation", validation_node)

    if has_judge:
        graph.add_node("judge", judge_node_wrapper)

    if has_validator:
        graph.add_edge(START, "load_validator")
        graph.add_edge("load_validator", "llm")
        graph.add_edge("llm", "validation")

        if has_judge:
            graph.add_edge("validation", "judge")
            graph.add_edge("judge", END)
        else:
            graph.add_edge("validation", END)
    else:
        graph.add_edge(START, "llm")

        if has_judge:
            graph.add_edge("llm", "judge")
            graph.add_edge("judge", END)
        else:
            graph.add_edge("llm", END)

    return graph.compile()


async def run_agent_network(directives: list):
    """Runs all directive branches in parallel using asyncio.gather."""
    results = await asyncio.gather(
        *[run_directive_branch(d) for d in directives]
    )
    return list(results)
