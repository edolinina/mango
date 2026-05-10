from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from agents.dataset_analysis_tool import run_dataset_analysis


class _Ctx:
    def __init__(self):
        self.info = AsyncMock()
        self.error = AsyncMock()


def test_run_dataset_analysis_raises_when_path_missing() -> None:
    ctx = _Ctx()

    with pytest.raises(ValueError, match="No dataset path provided"):
        __import__("asyncio").run(run_dataset_analysis(ctx, data_path=""))


def test_run_dataset_analysis_returns_summary_from_mock_dataframe() -> None:
    ctx = _Ctx()
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [100, 200, 300, 400],
        }
    )

    with patch("agents.dataset_analysis_tool.os.path.exists", return_value=True), patch(
        "agents.dataset_analysis_tool._load_dataframe", return_value=df
    ):
        result = __import__("asyncio").run(
            run_dataset_analysis(
                ctx,
                data_path="/app/data_sources/business.csv",
                feature_cols=["feature_a", "feature_b"],
                target_col="target",
                agent_name="BusinessAgent",
            )
        )

    assert result["dataset_overview"]["rows"] == 4
    assert result["target_summary"]["type"] == "numeric"
    assert len(result["feature_target_correlations"]) >= 1
