import operator
import logging
import os

import joblib
import pandas as pd
import yaml

CONFIG_PATH = "config/agents.yaml"
MODELS_DIR = "models"

logger = logging.getLogger("mango")

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _load_agent_config(agent_name: str) -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f) or {}

    for agent in config.get("agents", []):
        if agent.get("name") == agent_name:
            return agent

    raise ValueError(f"Agent '{agent_name}' not found in {CONFIG_PATH}")


def _resolve_validator(agent_cfg: dict, capability_name: str) -> dict:
    capabilities = agent_cfg.get("capabilities", [])
    capability = next((c for c in capabilities if c.get("name") == capability_name), None)
    if not capability:
        raise ValueError(
            f"Capability '{capability_name}' not found for agent '{agent_cfg.get('name', '')}'"
        )

    validator_name = capability.get("validator")
    if not validator_name:
        raise ValueError(
            f"Capability '{capability_name}' has no validator mapping in agents.yaml"
        )

    validators = agent_cfg.get("validators", [])
    validator_cfg = next((v for v in validators if v.get("name") == validator_name), None)
    if not validator_cfg:
        raise ValueError(
            f"Validator '{validator_name}' not found for agent '{agent_cfg.get('name', '')}'"
        )

    return validator_cfg


def _load_validation_df(validation_samples: list[dict]) -> pd.DataFrame:
    if not validation_samples:
        raise ValueError("No validation_samples provided by LLM — cannot run ML validation")
    return pd.DataFrame(validation_samples)


def _prepare_feature_frame(df_val: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    expected = set(features)
    missing_features = sorted(expected - set(df_val.columns))
    if missing_features:
        logger.info("ml_validation: filling missing features with 0: %s", missing_features)
        for col in missing_features:
            df_val[col] = 0
    return df_val[features].copy(), missing_features


def _safe_encode_series(series: pd.Series, label_encoder) -> pd.Series:
    def safe_encode(val):
        try:
            return label_encoder.transform([str(val)])[0]
        except ValueError:
            try:
                idx = int(float(str(val)))
                return max(0, min(idx, len(label_encoder.classes_) - 1))
            except (ValueError, TypeError):
                return 0

    return series.apply(safe_encode)


def _parse_pass_condition(pass_condition: str, preds: list[float]) -> tuple:
    if not pass_condition:
        raise ValueError("Validator pass_condition is missing")

    condition = pass_condition.strip()
    op_symbol = next((op for op in [">=", "<=", "==", "!=", ">", "<"] if condition.startswith(op)), None)
    if not op_symbol:
        raise ValueError(f"Unsupported pass_condition format: '{pass_condition}'")

    rhs_text = condition[len(op_symbol):].strip()
    if rhs_text.upper() == "MEAN":
        rhs_value = float(sum(preds) / len(preds)) if preds else 0.0
    else:
        rhs_value = float(rhs_text)

    return OPS[op_symbol], rhs_value, op_symbol


def run_ml_validation(
    agent_name: str,
    capability_name: str,
    validation_samples: list[dict] | None = None,
) -> dict:
    """Run trained ML validator and return pass metrics."""
    agent_cfg = _load_agent_config(agent_name)
    validator_cfg = _resolve_validator(agent_cfg, capability_name)
    validator_name = validator_cfg["name"]

    model_dir = os.path.join(MODELS_DIR, agent_name)
    model_path = os.path.join(model_dir, f"{validator_name}.pkl")
    scaler_path = os.path.join(model_dir, f"{validator_name}_scaler.pkl")
    encoders_path = os.path.join(model_dir, f"{validator_name}_encoders.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Validator model not found: {model_path}. Run train_validators first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    encoders = joblib.load(encoders_path) if os.path.exists(encoders_path) else {}

    df_val = _load_validation_df(validation_samples or [])
    features = validator_cfg.get("features", [])
    x_val, missing_features = _prepare_feature_frame(df_val, features)

    for col, le in (encoders or {}).items():
        if col in x_val.columns:
            x_val[col] = _safe_encode_series(x_val[col], le)

    for col in x_val.columns:
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")

    x_val = x_val.fillna(0)

    if scaler is not None:
        scaled = scaler.transform(x_val)
        # Keep feature names to avoid sklearn warnings when estimator was fit with named columns.
        x_val = pd.DataFrame(scaled, columns=features, index=x_val.index)

    preds = model.predict(x_val).tolist()

    comparator, rhs_value, op_symbol = _parse_pass_condition(validator_cfg.get("pass_condition", ""), preds)
    validation_results = [bool(comparator(p, rhs_value)) for p in preds]

    passed = sum(validation_results)
    failed = len(validation_results) - passed
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0.0

    if pass_rate == 100:
        status = "success"
    elif pass_rate >= 70:
        status = "partial_pass"
    else:
        status = "fail"

    return {
        "validator": validator_name,
        "pass_condition": validator_cfg.get("pass_condition", ""),
        "resolved_threshold": {"operator": op_symbol, "value": rhs_value},
        "missing_features_filled": missing_features,
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": round(pass_rate, 2),
        "status": status,
    }
