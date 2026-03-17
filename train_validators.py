import os
import json
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

CONFIG_PATH = "config/agents.yaml"
MODELS_DIR = "models"
SUMMARY_DIR = "models/summary"
MODEL_EVAL_PASSING_THRESHOLD = 0.75
MODEL_RESULTS = []

# Shared RandomForest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "random_state": 42,
}
RF_MIN_SAMPLES_LEAF_CLASSIFIER = 20
RF_MIN_SAMPLES_LEAF_REGRESSOR = 10

AUGMENT_MIN_ROWS = 500      # augment if training set has fewer rows than this
AUGMENT_MULTIPLIER = 5      # how many synthetic copies to add
AUGMENT_NOISE_STD = 0.05    # std of Gaussian noise (relative to feature std=1 since scaled)

def plot_model_scores():
    if not MODEL_RESULTS:
        return

    df = pd.DataFrame(MODEL_RESULTS)

    labels = df["validator"]
    train_scores = df["train_score"]
    test_scores = df["test_score"]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_scores, width, label='Train Score', color='skyblue')
    plt.bar(x + width/2, test_scores, width, label='Test Score', color='coral')

    plt.axhline(
        y=MODEL_EVAL_PASSING_THRESHOLD,
        color="red",
        linestyle="--",
        label="Pass Threshold"
    )

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Validator Model Evaluation Scores")
    plt.legend()

    plot_path = "validators_training_scores.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    print(f"Saved evaluation plot → {plot_path}")


def find_file_recursively(root_dir: str, filename: str) -> str | None:
    """Find a file by name anywhere under root_dir."""
    root = Path(root_dir)
    matches = list(root.rglob(filename))
    if matches:
        return str(matches[0])
    return None


def list_dataset_files(root_dir: str) -> list[str]:
    """Return relative file paths under a dataset directory for debugging."""
    root = Path(root_dir)
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                files.append(str(p.relative_to(root)))
            except Exception:
                files.append(str(p))
    return sorted(files)


def download_dataset(data_source_config):
    """
    Download dataset from Kaggle.
    Supports automatic Excel -> CSV conversion when sheet_name is provided.
    """

    target_path = data_source_config["target_file"]
    data_file = data_source_config["data_file"]
    kaggle_id = data_source_config["download_from"]
    sheet_name = data_source_config.get("sheet_name")

    downloaded_path = kagglehub.dataset_download(kaggle_id)

    source_file = os.path.join(downloaded_path, data_file)

    if not os.path.exists(source_file):
        source_file = find_file_recursively(downloaded_path, data_file)

    if not source_file or not os.path.exists(source_file):
        available_files = list_dataset_files(downloaded_path)
        raise FileNotFoundError(
            f"Configured data_file '{data_file}' not found in dataset '{kaggle_id}'.\n"
            f"Downloaded to: {downloaded_path}\n"
            f"Available files:\n- " + "\n- ".join(available_files[:100])
        )

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Load dataframe
    if source_file.endswith((".xlsx", ".xls")):

        if not sheet_name:
            raise ValueError(
                f"Excel file detected but no sheet_name provided in config: {data_file}"
            )

        df = pd.read_excel(source_file, sheet_name=sheet_name)

    else:

        df = pd.read_csv(source_file, dtype=str)

    # Detect duplicated header rows and keep last dataset
    header_col = df.columns[0]

    duplicate_header_rows = df[df[header_col] == header_col].index

    if len(duplicate_header_rows) > 0:
        split_index = duplicate_header_rows[-1] + 1
        df = df.iloc[split_index:].reset_index(drop=True)

    # restore correct column names
    df.columns = df.columns.str.strip()

    df.to_csv(target_path, index=False)

    print(f"Saved cleaned dataset -> {target_path}")

    return target_path


def preprocess_data(task, X, y):
    """
    Clean dataset, encode categorical features, and prepare target
    for regression or classification.
    """
    df = pd.concat([X, y], axis=1)

    print(f"Rows before cleaning: {df.shape[0]}")

    # Remove invalid values
    df = df.replace([np.inf, -np.inf], pd.NA).dropna().reset_index(drop=True)

    print(f"Rows after cleaning: {df.shape[0]}")

    y_clean = df[y.name].copy()
    X_clean = df.drop(columns=[y.name]).copy()

    encoders = {}
    target_encoder = None

    # Target processing based on task type
    if task == "classification":
        # categorical target like PerformanceRating
        if not pd.api.types.is_numeric_dtype(y_clean):
            target_encoder = LabelEncoder()
            y_clean = pd.Series(
                target_encoder.fit_transform(y_clean.astype(str)),
                name=y.name
            )
            print("Encoded categorical target with LabelEncoder")

    else:
        # regression target
        y_clean = pd.to_numeric(y_clean, errors="coerce")

    # Encode categorical features and ensure all are numeric
    for col in X_clean.columns:
        if not pd.api.types.is_numeric_dtype(X_clean[col]):
            le = LabelEncoder()
            X_clean[col] = le.fit_transform(X_clean[col].astype(str))
            encoders[col] = le

        X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce")

    # Final cleanup after encoding
    combined = pd.concat([X_clean, y_clean], axis=1).dropna().reset_index(drop=True)

    y_clean = combined[y.name]
    X_clean = combined.drop(columns=[y.name])

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=X_clean.columns,
        index=X_clean.index
    )

    print(f"Processed shape: X={X_scaled.shape} y={y_clean.shape}")

    return X_scaled, y_clean, scaler, encoders, target_encoder


def generate_summary_stats(df, agent_name, capability_name, validator):
    """Generate simple summary stats for LLM prompt grounding."""
    cols = validator["features"] + [validator["target"]]
    df_subset = df[cols].copy()

    for col in df_subset.columns:
        if not pd.api.types.is_numeric_dtype(df_subset[col]):
            df_subset[col] = LabelEncoder().fit_transform(df_subset[col].astype(str))
        df_subset[col] = pd.to_numeric(df_subset[col], errors="coerce")

    summary = {}
    for col in df_subset.columns:
        c = df_subset[col].dropna()
        if len(c) == 0:
            continue

        summary[col] = {
            "mean": float(c.mean()),
            "min": float(c.min()),
            "max": float(c.max()),
            "p70": float(c.quantile(0.70)),
            "p80": float(c.quantile(0.80)),
            "p90": float(c.quantile(0.90)),
        }

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(SUMMARY_DIR, f"{agent_name}_{capability_name}_stats.json")

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def create_model(task: str):
    """Use only the simplest models."""
    if task == "classification":
        return RandomForestClassifier(
            **RF_PARAMS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF_CLASSIFIER,
            class_weight="balanced",
        )
    return RandomForestRegressor(
        **RF_PARAMS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF_REGRESSOR,
    )


def augment_data(X: pd.DataFrame, y: pd.Series, task: str,
                 multiplier: int = AUGMENT_MULTIPLIER,
                 noise_std: float = AUGMENT_NOISE_STD):
    """
    Augment a small dataset by adding Gaussian noise copies.
    For classification, preserves class balance by augmenting per class.
    """
    rng = np.random.default_rng(42)
    X_parts = [X]
    y_parts = [y]

    if task == "classification":
        classes = y.unique()
        for cls in classes:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
            for _ in range(multiplier):
                noise = rng.normal(0, noise_std, size=X_cls.shape)
                X_parts.append(pd.DataFrame(X_cls.values + noise, columns=X.columns))
                y_parts.append(y_cls.reset_index(drop=True))
    else:
        for _ in range(multiplier):
            noise = rng.normal(0, noise_std, size=X.shape)
            X_parts.append(pd.DataFrame(X.values + noise, columns=X.columns))
            y_parts.append(y.reset_index(drop=True))

    X_aug = pd.concat(X_parts, ignore_index=True)
    y_aug = pd.concat(y_parts, ignore_index=True)

    print(f"Augmented dataset: {len(X)} → {len(X_aug)} rows")
    return X_aug, y_aug


def train_validator(agent_name, data_source, validator, capability_name):
    print(f"\nTraining {agent_name} → {validator['name']}")

    df = pd.read_csv(data_source)
    df.columns = df.columns.str.strip()

    target = validator["target"]
    features = validator["features"]
    task = validator.get("task", "regression")

    generate_summary_stats(df, agent_name, capability_name, validator)
    print(f"Saved summary stats → {SUMMARY_DIR}/{agent_name}_{capability_name}_stats.json")

    X = df[features].copy()
    y = df[target].copy()

    print(f"Original shape: X={X.shape} y={y.shape}")
    print(f"Target dtype: {y.dtype}")
    print(f"Sample target values: {y.head().tolist()}")

    X_processed, y_processed, scaler, encoders, target_encoder = \
        preprocess_data(task, X, y)

    print(f"Processed shape: X={X_processed.shape} y={y_processed.shape}")

    model = create_model(task)
    print(f"Using model: {model.__class__.__name__}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=0.1,
        random_state=42,
        stratify=y_processed if task == "classification" else None
    )

    # Augment only the training split if it's small
    augment = validator.get("augment", True)
    augment_min_rows = validator.get("augment_min_rows", AUGMENT_MIN_ROWS)
    if augment and len(X_train) < augment_min_rows:
        print(f"Small training set ({len(X_train)} rows < {augment_min_rows}), applying augmentation...")
        X_train, y_train = augment_data(X_train, y_train, task)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    train_preds = model.predict(X_train)

    if task == "classification":
        score = f1_score(y_test, preds, average='weighted')
        train_score = f1_score(y_train, train_preds, average='weighted')
        metric_name = "F1 Score"
    else:
        preds = np.maximum(preds, 0)
        score = r2_score(y_test, preds)
        train_score = r2_score(y_train, train_preds)
        metric_name = "R²"
    
    passed = score >= MODEL_EVAL_PASSING_THRESHOLD
    print(f"Test {metric_name}: {score:.3f}")
    print(f"Train {metric_name}: {train_score:.3f}")
    print("Validator result:", "PASS" if passed else "FAIL")

    sample_pred = model.predict(X_processed[:5])
    sample_true = y_processed.iloc[:5].tolist() if hasattr(y_processed, "iloc") else y_processed[:5].tolist()

    print("Sample predictions:", sample_pred)
    print("Actual values:", sample_true)

    agent_dir = os.path.join(MODELS_DIR, agent_name)
    os.makedirs(agent_dir, exist_ok=True)

    model_path = os.path.join(agent_dir, f"{validator['name']}.pkl")
    scaler_path = os.path.join(agent_dir, f"{validator['name']}_scaler.pkl")
    encoders_path = os.path.join(agent_dir, f"{validator['name']}_encoders.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoders_path)

    if target_encoder:
        target_encoder_path = os.path.join(agent_dir, f"{validator['name']}_target_encoder.pkl")
        joblib.dump(target_encoder, target_encoder_path)
        print(f"Saved target encoder → {target_encoder_path}")

    print(f"Saved model → {model_path}")
    print(f"Saved scaler → {scaler_path}")
    print(f"Saved encoders → {encoders_path}")

    MODEL_RESULTS.append({
        "agent": agent_name,
        "validator": validator["name"],
        "metric": metric_name,
        "test_score": score,
        "train_score": train_score
    })

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    for agent in config.get("agents", []):
        agent_name = agent["name"]
        data_source_config = agent.get("data_source")

        if not data_source_config:
            continue

        if isinstance(data_source_config, dict):
            data_source = download_dataset(data_source_config)
        else:
            data_source = data_source_config

        for validator in agent.get("validators", []):
            capability_name = next(
                (
                    cap["name"]
                    for cap in agent.get("capabilities", [])
                    if cap.get("validator") == validator["name"]
                ),
                validator["name"],
            )

            train_validator(agent_name, data_source, validator, capability_name)
    
    plot_model_scores()


if __name__ == "__main__":
    main()