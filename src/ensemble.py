from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .artifacts import write_json_atomic, write_parquet_atomic
from .config import ExperimentConfig
from .evaluation import (
    form_equal_weight_deciles,
    monthly_rank_ic,
    oos_r2,
    performance_stats,
)
from .model_comparison import build_model_comparison_table


ENSEMBLE_ID = "ENSEMBLE_LGBM40_DEEPSET40_DYNAMIC_50_50"
ENSEMBLE_COMPONENTS = ("LGBM_40", "DEEPSET_40_DYNAMIC")


def _prediction_path(config: ExperimentConfig, model_id: str) -> Path:
    return config.run_dir / "predictions" / f"{model_id}.parquet"


def _load_predictions(config: ExperimentConfig, model_id: str) -> pd.DataFrame:
    path = _prediction_path(config, model_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions for {model_id}: {path}")

    data = pd.read_parquet(path)
    data["eom"] = pd.to_datetime(data["eom"])
    data["target_date"] = pd.to_datetime(data["target_date"])
    return data


def build_fixed_fifty_fifty(
    config: ExperimentConfig,
    *,
    model_id: str = ENSEMBLE_ID,
    component_ids: tuple[str, str] = ENSEMBLE_COMPONENTS,
) -> dict:
    """
    Build and evaluate the fixed 50/50 ensemble from saved OOS component predictions.

    No model is fitted and no weight is estimated. The function requires the two
    component prediction files to cover the identical OOS stock-month sample.
    """
    metric_path = config.run_dir / "metrics" / f"{model_id}.json"
    required_outputs = (
        _prediction_path(config, model_id),
        metric_path,
        config.run_dir / "diagnostics" / f"{model_id}_monthly_rank_ic.parquet",
        config.run_dir / "portfolios" / f"{model_id}_long_short.parquet",
    )
    if all(path.exists() and path.stat().st_size > 0 for path in required_outputs):
        saved = json.loads(metric_path.read_text(encoding="utf-8"))
        if (
            saved.get("model_id") == model_id
            and saved.get("component_1") == component_ids[0]
            and saved.get("component_2") == component_ids[1]
        ):
            return saved

    left_id, right_id = component_ids
    left = _load_predictions(config, left_id)
    right = _load_predictions(config, right_id)

    keys = [
        "eom",
        "target_date",
        "id",
        "security_id",
        "country",
        "test_year",
        "refit_id",
    ]

    metadata = [
        "y_true",
        "me",
        "size_grp",
        "target_available",
    ]

    for frame, name in ((left, left_id), (right, right_id)):
        missing = set(keys + metadata + ["y_pred"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate OOS keys.")

    for column in metadata:
        left_values = left[keys + [column]].sort_values(keys).reset_index(drop=True)[column]
        right_values = right[keys + [column]].sort_values(keys).reset_index(drop=True)[column]
        if not left_values.equals(right_values):
            raise RuntimeError(f"Component metadata differs for {column}.")

    right_forecast = right[keys + ["y_pred"]].rename(columns={"y_pred": "right_pred"})

    combined = (
        left[keys + metadata + ["y_pred"]]
        .rename(columns={"y_pred": "left_pred"})
        .merge(right_forecast, on=keys, how="inner", validate="one_to_one")
    )

    if len(combined) != len(left) or len(combined) != len(right):
        raise RuntimeError("The fixed 50/50 components do not cover the same sample.")

    combined["y_pred"] = 0.5 * combined["left_pred"] + 0.5 * combined["right_pred"]
    combined["model_id"] = model_id

    predictions = (
        combined.drop(columns=["left_pred", "right_pred"])
        .sort_values(["eom", "security_id"])
        .reset_index(drop=True)
    )

    monthly_ic, rank_stats = monthly_rank_ic(
        predictions,
        config.portfolio.newey_west_lags,
    )

    long_short = form_equal_weight_deciles(
        predictions,
        config.portfolio.n_groups,
    )

    portfolio_stats = performance_stats(
        long_short["long_short_ret"],
        config.portfolio.newey_west_lags,
    )

    metrics = {
        "model_id": model_id,
        "training_version": "derived_fixed_50_50_from_saved_oos_predictions",
        "component_1": left_id,
        "component_2": right_id,
        "pooled_oos_r2": oos_r2(predictions),
        **rank_stats,
        **portfolio_stats,
    }

    write_parquet_atomic(predictions, _prediction_path(config, model_id))
    write_parquet_atomic(
        monthly_ic,
        config.run_dir / "diagnostics" / f"{model_id}_monthly_rank_ic.parquet",
    )
    write_parquet_atomic(
        long_short,
        config.run_dir / "portfolios" / f"{model_id}_long_short.parquet",
    )
    write_json_atomic(metrics, config.run_dir / "metrics" / f"{model_id}.json")

    # Refresh the comparison table after ensemble construction.
    build_model_comparison_table(config.run_dir)

    return metrics
