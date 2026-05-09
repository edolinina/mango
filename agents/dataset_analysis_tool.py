import os
import logging

import pandas as pd
from fastmcp import Context

logger = logging.getLogger("mango")


def _load_dataframe(data_path: str) -> pd.DataFrame:
    if data_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(data_path)
    return pd.read_csv(data_path, low_memory=False)


def _build_structured_summary(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict:
    rows, cols = df.shape
    missing_pct = float(df.isna().mean().mean() * 100) if rows else 0.0

    dataset_overview = {
        "rows": rows,
        "columns": cols,
        "overall_missing_pct": round(missing_pct, 2),
    }

    target_summary: dict = {
        "target": target_col,
        "present": bool(target_col and target_col in df.columns),
    }

    if target_col and target_col in df.columns:
        tgt = df[target_col]
        if pd.api.types.is_numeric_dtype(tgt):
            t = pd.to_numeric(tgt, errors="coerce").dropna()
            if len(t):
                target_summary.update(
                    {
                        "type": "numeric",
                        "mean": round(float(t.mean()), 6),
                        "median": round(float(t.median()), 6),
                        "std": round(float(t.std()), 6),
                    }
                )
        else:
            top = tgt.astype(str).value_counts(dropna=False).head(3).to_dict()
            target_summary.update({"type": "categorical", "top_classes": top})

    feature_stats: dict = {}
    for col in feature_cols[:8]:
        if col not in df.columns:
            continue

        s = df[col]
        miss = float(s.isna().mean() * 100)

        if pd.api.types.is_numeric_dtype(s):
            n = pd.to_numeric(s, errors="coerce").dropna()
            if len(n):
                feature_stats[col] = {
                    "type": "numeric",
                    "mean": round(float(n.mean()), 6),
                    "p50": round(float(n.median()), 6),
                    "min": round(float(n.min()), 6),
                    "max": round(float(n.max()), 6),
                    "missing_pct": round(miss, 2),
                }
            else:
                feature_stats[col] = {
                    "type": "numeric",
                    "missing_pct": round(miss, 2),
                    "note": "empty_after_numeric_coercion",
                }
        else:
            top = s.astype(str).value_counts(dropna=False).head(2).to_dict()
            feature_stats[col] = {
                "type": "categorical",
                "top_values": top,
                "missing_pct": round(miss, 2),
            }

    correlations: list[dict] = []
    if target_col and target_col in df.columns and feature_cols and pd.api.types.is_numeric_dtype(df[target_col]):
        target_num = pd.to_numeric(df[target_col], errors="coerce")
        corr_rows: list[tuple[str, float]] = []

        for col in feature_cols:
            if col not in df.columns or col == target_col:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            x = pd.to_numeric(df[col], errors="coerce")
            valid = x.notna() & target_num.notna()
            if valid.sum() < 3:
                continue

            corr = x[valid].corr(target_num[valid])
            if pd.notna(corr):
                corr_rows.append((col, float(corr)))

        corr_rows.sort(key=lambda item: abs(item[1]), reverse=True)
        correlations = [
            {
                "feature": col,
                "corr": round(corr, 6),
                "abs_corr": round(abs(corr), 6),
                "method": "pearson_numeric",
            }
            for col, corr in corr_rows[:8]
        ]

    return {
        "dataset_overview": dataset_overview,
        "target_summary": target_summary,
        "feature_stats": feature_stats,
        "feature_target_correlations": correlations,
    }


async def run_dataset_analysis(
    ctx: Context,
    data_path: str = "",
    feature_cols: list[str] | None = None,
    target_col: str = "",
    agent_name: str = "",
) -> dict:
    """Analyze a dataset and return structured summary stats and feature-target correlations."""
    await ctx.info(f"run_dataset_analysis started for agent='{agent_name or 'unknown'}'")

    try:
        resolved_path = data_path.strip()
        if not resolved_path:
            raise ValueError("No dataset path provided")
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Dataset not found: {resolved_path}")

        df = _load_dataframe(resolved_path)
        selected_features = list(feature_cols or [])

        if not selected_features:
            selected_features = [c for c in df.columns if c != target_col][:8]

        result = _build_structured_summary(df, selected_features, target_col)

        rows = result.get("dataset_overview", {}).get("rows", 0)
        cols = result.get("dataset_overview", {}).get("columns", 0)
        logger.info(
            "run_dataset_analysis | agent=%s | path=%s | rows=%s | cols=%s",
            agent_name or "unknown",
            resolved_path,
            rows,
            cols,
        )
        await ctx.info(f"run_dataset_analysis completed: rows={rows}, cols={cols}")

        return result
    except Exception as exc:
        logger.exception("run_dataset_analysis failed | agent=%s | path=%s", agent_name or "unknown", data_path)
        await ctx.error(f"run_dataset_analysis failed: {exc}")
        raise
