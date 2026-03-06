import os
import yaml
import joblib
import pandas as pd
import shutil
import kagglehub
import json

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

CONFIG_PATH = "config/agents.yaml"
MODELS_DIR = "models"
SUMMARY_DIR = "models/summary"


def download_dataset_if_needed(data_source_config):
    """Download dataset from Kaggle if it doesn't exist locally."""
    target_path = data_source_config.get("target_file")
    
    if os.path.exists(target_path):
        print(f"Dataset already exists at {target_path}")
        return target_path
    
    # Download from Kaggle
    kaggle_id = data_source_config['download_from']
    data_file = data_source_config['data_file']
    
    print(f"Downloading dataset: {kaggle_id}")
    downloaded_path = kagglehub.dataset_download(kaggle_id)
    
    # Find the specific data file in downloaded directory
    source_file = os.path.join(downloaded_path, data_file)
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Data file {data_file} not found in {downloaded_path}")
    
    # Copy to target location
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_file, target_path)
    
    print(f"✓ Dataset saved to {target_path}")
    return target_path


def preprocess_data(X, y):
    """Preprocess features: drop nulls, encode categoricals, and scale."""
    df = pd.concat([X, y], axis=1)
    
    # Drop nulls and infinities
    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()
    
    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    
    y_clean = df[y.name]
    X_clean = df.drop(columns=[y.name])
    
    # Encode categorical target
    target_encoder = None
    if y_clean.dtype == 'object' or pd.api.types.is_string_dtype(y_clean):
        target_encoder = LabelEncoder()
        y_clean = pd.Series(target_encoder.fit_transform(y_clean.astype(str)), name=y.name)
    
    # Encode categorical features - handle ALL non-numeric columns
    encoders = {}
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(X_clean[col]) or not pd.api.types.is_numeric_dtype(X_clean[col]):
            encoders[col] = LabelEncoder()
            X_clean[col] = encoders[col].fit_transform(X_clean[col].astype(str))
        
        # Ensure column is numeric after encoding
        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
    
    # Drop any remaining NaN values created by failed numeric conversion
    combined_again = pd.concat([X_clean, y_clean], axis=1).dropna()
    y_clean = combined_again[y.name]
    X_clean = combined_again.drop(columns=[y.name])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled, columns=X_clean.columns)
    
    print(f"Scaled {X_scaled.shape[1]} features using StandardScaler")
    
    return X_scaled, y_clean, scaler, encoders, target_encoder


def detect_model_type(y):
    if y.dtype.kind in {"i", "b"} and y.nunique() <= 10:
        return "classification"
    return "regression"


def generate_summary_stats(df_full, agent_name, capability_name, validator):
    """Generate summary statistics for validation-relevant columns only."""
    target = validator["target"]
    features = validator["features"]
    relevant_columns = features + [target]
    
    # Only use columns relevant to this validator
    df_relevant = df_full[relevant_columns].copy()
    
    # Encode categorical columns for stats generation
    df_numeric = df_relevant.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object' or pd.api.types.is_string_dtype(df_numeric[col]):
            le = LabelEncoder()
            # Convert to string first to handle mixed types
            df_numeric[col] = df_numeric[col].astype(str)
            df_numeric[col] = le.fit_transform(df_numeric[col])
        
        # Ensure all columns are numeric
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Replace infinities with NaN before stats calculation
    df_numeric = df_numeric.replace([float('inf'), float('-inf')], pd.NA)
    
    summary = {}
    
    for col in df_numeric.columns:
        # Drop NaN for statistics calculation
        col_clean = df_numeric[col].dropna()
        
        if len(col_clean) > 0 and pd.api.types.is_numeric_dtype(col_clean):
            summary[col] = {
                'mean': float(col_clean.mean()),
                'min': float(col_clean.min()),
                'max': float(col_clean.max()),
                'percentile_70': float(col_clean.quantile(0.70)),
                'percentile_80': float(col_clean.quantile(0.80)),
                'percentile_90': float(col_clean.quantile(0.90))
            }
        else:
            summary[col] = {
                'error': 'No valid numeric data'
            }
    
    # Save summary statistics
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, f"{agent_name}_{capability_name}_stats.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary stats → {summary_path}")
    
    return summary


def train_validator(agent_name, data_source, validator, capability_name):
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

    # Generate summary statistics BEFORE encoding (to preserve original data types)
    generate_summary_stats(df, agent_name, capability_name, validator)

    # Extract features and target for model training
    X = df[features].copy()
    y = df[target].copy()

    # Preprocess: encode categoricals, drop nulls and scale
    X_processed, y_processed, scaler, encoders, target_encoder = preprocess_data(X, y)

    model_engine = None
    model_type = detect_model_type(y_processed)
    if model_type == "classification":
        model_engine = LogisticRegression
    else:
        model_engine = LinearRegression

    model = model_engine()
    model.fit(X_processed, y_processed)

    # Create folder per agent
    agent_model_dir = os.path.join(MODELS_DIR, agent_name)
    os.makedirs(agent_model_dir, exist_ok=True)

    model_path = os.path.join(
        agent_model_dir,
        f"{validator['name']}.pkl"
    )
    
    scaler_path = os.path.join(
        agent_model_dir,
        f"{validator['name']}_scaler.pkl"
    )
    
    encoders_path = os.path.join(
        agent_model_dir,
        f"{validator['name']}_encoders.pkl"
    )
    
    target_encoder_path = os.path.join(
        agent_model_dir,
        f"{validator['name']}_target_encoder.pkl"
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoders, encoders_path)
    if target_encoder:
        joblib.dump(target_encoder, target_encoder_path)

    print(f"Saved model → {model_path}")
    print(f"Saved scaler → {scaler_path}")
    print(f"Saved encoders → {encoders_path}")
    if target_encoder:
        print(f"Saved target encoder → {target_encoder_path}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    agents = config.get("agents", [])

    for agent in agents:
        agent_name = agent["name"]
        data_source_config = agent.get("data_source")

        if not data_source_config:
            print(f"No data source for {agent_name}, skipping.")
            continue
        
        # Download dataset if needed
        if isinstance(data_source_config, dict):
            data_source = download_dataset_if_needed(data_source_config)
        else:
            # Legacy: direct path
            data_source = data_source_config
    
        validators = agent.get("validators", [])
        if not validators:
            print(f"No validators for {agent_name}")
            continue

        for validator in validators:
            # Find the capability that uses this validator
            capability_name = None
            for capability in agent.get("capabilities", []):
                if capability.get("validator") == validator["name"]:
                    capability_name = capability["name"]
                    break
            
            if not capability_name:
                print(f"No capability found for validator {validator['name']}")
                continue
                
            train_validator(agent_name, data_source, validator, capability_name)

    print("\nAll validators trained successfully.")


if __name__ == "__main__":
    main()