import os
import yaml
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


CONFIG_PATH = "config/agents.yaml"
MODELS_DIR = "models"


def detect_model_type(y):
    if y.dtype.kind in {"i", "b"} and y.nunique() <= 10:
        return "classification"
    return "regression"


def train_validator(agent_name, data_source, validator):
    print(f"\nTraining {agent_name} → {validator['name']}")

    df = pd.read_csv(data_source)
    df.columns = df.columns.str.strip()

    target = validator["target"]
    features = validator["features"]

    missing = set(features + [target]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for {agent_name}/{validator['name']}: {missing}"
        )

    X = df[features]
    y = df[target]

    model_engine = None
    model_type = detect_model_type(y)
    if model_type == "classification":
        model_engine = RandomForestClassifier
    else:
        model_engine = RandomForestRegressor

    model = model_engine(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Create folder per agent
    agent_model_dir = os.path.join(MODELS_DIR, agent_name)
    os.makedirs(agent_model_dir, exist_ok=True)

    model_path = os.path.join(
        agent_model_dir,
        f"{validator['name']}.pkl"
    )

    joblib.dump(model, model_path)

    print(f"Saved model → {model_path}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    agents = config.get("agents", [])

    for agent in agents:
        agent_name = agent["name"]
        data_source = agent.get("data_source")

        if not data_source:
            print(f"No data source for {agent_name}, skipping.")
            continue
    
        validators = agent.get("validators", [])
        if not validators:
            print(f"No validators for {agent_name}")
            continue

        for validator in validators:
            train_validator(agent_name, data_source, validator)

    print("\nAll validators trained successfully.")


if __name__ == "__main__":
    main()