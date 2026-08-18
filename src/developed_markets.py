from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import stable_hash, write_json_atomic, write_parquet_atomic
from .config import ExperimentConfig
from .evaluation import (
    form_portfolio_variants,
    monthly_rank_ic,
    oos_r2,
    performance_stats,
    robust_oos_r2,
)


COUNTRY_NAMES = {
    "GBR": "United Kingdom",
    "AUS": "Australia",
    "DEU": "Germany",
    "FRA": "France",
}
EXTERNAL_MODEL_IDS = (
    "LGBM_40",
    "DEEPSET_40_DYNAMIC",
    "HYBRID_LGBM40_DEEPSET40_DYNAMIC",
)
FIFTY_FIFTY_ID = "HYBRID_LGBM40_DEEPSET40_DYNAMIC_50_50"
COMPARATOR_VERSION = "standalone_oos_prediction_average_v1"


def build_fifty_fifty_comparator(config: ExperimentConfig) -> dict:
    """Average aligned standalone OOS predictions without fitting another model."""
    run_dir = config.run_dir
    left_path = run_dir / "predictions" / "LGBM_40.parquet"
    right_path = run_dir / "predictions" / "DEEPSET_40_DYNAMIC.parquet"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError("Standalone component predictions are incomplete.")

    keys = ["eom", "target_date", "id", "security_id", "country", "test_year", "refit_id"]
    metadata = ["y_true", "me", "size_grp", "target_available"]
    left = pd.read_parquet(left_path)
    right = pd.read_parquet(right_path)
    left_signature = left["model_signature"].dropna().astype(str).unique()
    right_signature = right["model_signature"].dropna().astype(str).unique()
    if len(left_signature) != 1 or len(right_signature) != 1:
        raise RuntimeError("Each comparator component must have exactly one signature.")
    right_forecast = right[keys + ["y_pred"]].rename(columns={"y_pred": "right_pred"})
    combined = left[keys + metadata + ["y_pred"]].rename(
        columns={"y_pred": "left_pred"}
    ).merge(right_forecast, on=keys, validate="one_to_one")
    if len(combined) != len(left) or len(combined) != len(right):
        raise RuntimeError("The 50/50 components do not cover identical OOS observations.")

    signature = stable_hash({
        "version": COMPARATOR_VERSION,
        "left_signature": left_signature[0],
        "right_signature": right_signature[0],
    })
    combined["y_pred"] = 0.5 * (combined["left_pred"] + combined["right_pred"])
    combined["model_id"] = FIFTY_FIFTY_ID
    combined["model_signature"] = signature
    predictions = combined[
        keys + metadata + ["y_pred", "model_id", "model_signature"]
    ].sort_values(["eom", "security_id"])

    _, rank_stats = monthly_rank_ic(predictions, config.portfolio.newey_west_lags)
    variants = form_portfolio_variants(predictions)
    tail = variants.query("strategy == 'TAIL_10PCT'")
    performance = performance_stats(
        tail["long_short_ret"], config.portfolio.newey_west_lags
    )
    metrics = {
        "model_id": FIFTY_FIFTY_ID,
        "model_signature": signature,
        "comparator_version": COMPARATOR_VERSION,
        "pooled_oos_r2": oos_r2(predictions),
        "robust_oos_r2": robust_oos_r2(predictions),
        **rank_stats,
        **performance,
    }
    write_parquet_atomic(
        predictions, run_dir / "predictions" / f"{FIFTY_FIFTY_ID}.parquet"
    )
    write_parquet_atomic(
        variants, run_dir / "portfolios" / f"{FIFTY_FIFTY_ID}_variants.parquet"
    )
    write_json_atomic(metrics, run_dir / "metrics" / f"{FIFTY_FIFTY_ID}.json")
    return metrics


def country_comparison(config: ExperimentConfig, include_fifty_fifty: bool = True) -> pd.DataFrame:
    """Return the report-facing model table for one completed country run."""
    comparison_path = config.run_dir / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)
    comparison = pd.read_csv(comparison_path)
    rows = comparison[comparison["model_id"].isin(EXTERNAL_MODEL_IDS)].copy()
    if len(rows) != len(EXTERNAL_MODEL_IDS):
        missing = sorted(set(EXTERNAL_MODEL_IDS) - set(rows["model_id"]))
        raise RuntimeError(f"Country comparison is incomplete: {missing}")
    if include_fifty_fifty:
        rows = pd.concat([rows, pd.DataFrame([build_fifty_fifty_comparator(config)])])
    rows.insert(0, "country_name", COUNTRY_NAMES[config.universe.country])
    rows.insert(0, "country", config.universe.country)
    columns = [
        "country", "country_name", "model_id", "pooled_oos_r2", "robust_oos_r2",
        "mean_monthly_rank_ic", "rank_ic_newey_west_t_stat",
        "annualized_return", "annualized_volatility", "sharpe",
        "newey_west_t_stat", "max_drawdown", "hit_rate", "n_months",
    ]
    return rows[columns].sort_values("model_id").reset_index(drop=True)


def aggregate_country_comparisons(
    configs: dict[str, ExperimentConfig], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine all pre-specified countries and summarize hybrid improvements."""
    results = pd.concat(
        [country_comparison(config) for config in configs.values()],
        ignore_index=True,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "developed_markets_model_comparison.csv", index=False)

    wide = results.pivot(index=["country", "country_name"], columns="model_id")
    summary_rows = []
    chosen = "HYBRID_LGBM40_DEEPSET40_DYNAMIC"
    for country, country_name in wide.index:
        row = {"country": country, "country_name": country_name}
        for benchmark in ("LGBM_40", "DEEPSET_40_DYNAMIC", FIFTY_FIFTY_ID):
            label = benchmark.lower()
            row[f"hybrid_minus_{label}_sharpe"] = (
                wide.loc[(country, country_name), ("sharpe", chosen)]
                - wide.loc[(country, country_name), ("sharpe", benchmark)]
            )
            row[f"hybrid_minus_{label}_annualized_return"] = (
                wide.loc[(country, country_name), ("annualized_return", chosen)]
                - wide.loc[(country, country_name), ("annualized_return", benchmark)]
            )
        summary_rows.append(row)
    improvements = pd.DataFrame(summary_rows)
    improvements.to_csv(output_dir / "developed_markets_hybrid_improvements.csv", index=False)
    return results, improvements
