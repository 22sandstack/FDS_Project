from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json_atomic, write_parquet_atomic
from .evaluation import has_cross_sectional_signal, merge_predictions, performance_stats


ROBUSTNESS_VERSION = "portfolio_robustness_v4_missing_return_stress"
ROBUSTNESS_SPECS: tuple[tuple[str, str], ...] = (
    ("FULL", "EQUAL"),
    ("FULL", "VALUE"),
    ("EX_MICRO", "EQUAL"),
    ("EX_MICRO", "VALUE"),
)
COST_BPS: tuple[int, ...] = (10, 25, 50)


def _prediction_signature(path: Path) -> str:
    signatures = pd.read_parquet(path, columns=["model_signature"])[
        "model_signature"
    ].dropna().unique()
    if len(signatures) != 1:
        raise ValueError(f"{path} must contain exactly one model signature.")
    return str(signatures[0])


def _target_weights(month: pd.DataFrame, weighting: str) -> pd.DataFrame:
    month = month.sort_values(["y_pred", "permno"], kind="mergesort").copy()
    leg_size = max(1, int(math.ceil(0.10 * len(month))))
    short = month.iloc[:leg_size].copy()
    long = month.iloc[-leg_size:].copy()
    if weighting == "EQUAL":
        long["weight"] = 1.0 / len(long)
        short["weight"] = -1.0 / len(short)
    elif weighting == "VALUE":
        long["weight"] = long["me"] / long["me"].sum()
        short["weight"] = -short["me"] / short["me"].sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    return pd.concat([long, short], ignore_index=True)


def _realized_leg_return(
    weights: pd.DataFrame, positive_leg: bool, missing_return_fill: float
) -> tuple[float, float, int, float]:
    leg = weights[weights["weight"] > 0] if positive_leg else weights[weights["weight"] < 0]
    valid = leg[leg["target_available"] & leg["y_true"].notna()].copy()
    absolute_weight = valid["weight"].abs()
    realized = (
        float((absolute_weight / absolute_weight.sum() * valid["y_true"]).sum())
        if absolute_weight.sum() > 0 else np.nan
    )
    stressed_y = leg["y_true"].where(
        leg["target_available"] & leg["y_true"].notna(), missing_return_fill
    )
    stressed = float((leg["weight"].abs() * stressed_y).sum())
    return realized, stressed, int(len(valid)), float(len(valid) / len(leg)) if len(leg) else np.nan


def build_robustness_portfolios(
    evaluated: pd.DataFrame, universe_name: str, weighting: str
) -> pd.DataFrame:
    data = evaluated.copy()
    data["y_pred"] = data["y_pred"].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["eom", "permno", "y_pred"])
    if universe_name == "EX_MICRO":
        data = data[data["size_grp"].isin(["small", "large", "mega"])]
    elif universe_name != "FULL":
        raise ValueError(f"Unknown robustness universe: {universe_name}")
    if weighting == "VALUE":
        data = data[np.isfinite(data["me"]) & (data["me"] > 0)]

    rows, previous = [], pd.Series(dtype=float)
    for eom, month in data.groupby("eom", sort=True):
        if len(month) < 20:
            continue
        if not has_cross_sectional_signal(month):
            current = pd.Series(dtype=float)
            turnover = 0.5 * float(previous.abs().sum())
            row = {
                "eom": eom, "test_year": int(pd.Timestamp(eom).year),
                "universe": universe_name, "weighting": weighting,
                "strategy": "TAIL_10PCT", "gross_long_short_ret": 0.0,
                "missing_return_stress_ret": 0.0,
                "long_ret": np.nan, "short_ret": np.nan,
                "turnover": turnover, "n_eligible": int(len(month)),
                "n_long": 0, "n_short": 0, "n_long_realized": 0,
                "n_short_realized": 0, "long_coverage": np.nan,
                "short_coverage": np.nan, "signal_available": False,
            }
            for cost in COST_BPS:
                row[f"net_return_{cost}bps"] = -cost / 10_000.0 * turnover
            rows.append(row)
            previous = current
            continue
        weights = _target_weights(month, weighting)
        current = weights.set_index("permno")["weight"]
        combined = current.index.union(previous.index)
        turnover = 0.5 * float(
            (current.reindex(combined, fill_value=0.0) - previous.reindex(combined, fill_value=0.0)).abs().sum()
        )
        realized_returns = month.loc[
            month["target_available"] & month["y_true"].notna(), "y_true"
        ]
        if realized_returns.empty:
            adverse_long_return = adverse_short_return = np.nan
        else:
            adverse_long_return, adverse_short_return = realized_returns.quantile([0.01, 0.99])
        long_return, stressed_long, n_long_realized, long_coverage = _realized_leg_return(
            weights, True, adverse_long_return
        )
        short_return, stressed_short, n_short_realized, short_coverage = _realized_leg_return(
            weights, False, adverse_short_return
        )
        gross_return = long_return - short_return
        missing_return_stress = stressed_long - stressed_short
        row = {
            "eom": eom, "test_year": int(pd.Timestamp(eom).year),
            "universe": universe_name, "weighting": weighting,
            "strategy": "TAIL_10PCT", "gross_long_short_ret": gross_return,
            "missing_return_stress_ret": missing_return_stress,
            "long_ret": long_return, "short_ret": short_return,
            "turnover": turnover, "n_eligible": int(len(month)),
            "n_long": int((weights["weight"] > 0).sum()),
            "n_short": int((weights["weight"] < 0).sum()),
            "n_long_realized": n_long_realized, "n_short_realized": n_short_realized,
            "long_coverage": long_coverage, "short_coverage": short_coverage,
            "signal_available": True,
        }
        for cost in COST_BPS:
            row[f"net_return_{cost}bps"] = gross_return - cost / 10_000.0 * turnover
        rows.append(row)
        previous = current
    return pd.DataFrame(rows)


def robustness_summary(monthly: pd.DataFrame) -> list[dict]:
    rows = []
    for (universe_name, weighting), group in monthly.groupby(["universe", "weighting"], sort=True):
        row = {
            "universe": universe_name, "weighting": weighting,
            "mean_monthly_turnover": float(group["turnover"].mean()),
            "annualized_turnover": float(12.0 * group["turnover"].mean()),
            "mean_n_eligible": float(group["n_eligible"].mean()),
            "mean_long_coverage": float(group["long_coverage"].mean()),
            "mean_short_coverage": float(group["short_coverage"].mean()),
        }
        for label, column in [("gross", "gross_long_short_ret")] + [
            (f"net_{cost}bps", f"net_return_{cost}bps") for cost in COST_BPS
        ]:
            for key, value in performance_stats(group[column]).items():
                row[f"{label}_{key}"] = value
        for key, value in performance_stats(group["missing_return_stress_ret"]).items():
            row[f"missing_return_stress_{key}"] = value
        rows.append(row)
    return rows


def run_portfolio_robustness(
    run_dir: Path, model_ids: tuple[str, ...] | list[str] | None = None
) -> pd.DataFrame:
    """Evaluate saved pooled predictions; current cached model results are skipped."""
    run_dir = Path(run_dir)
    prediction_dir = run_dir / "predictions"
    if model_ids is None:
        model_ids = tuple(path.stem for path in sorted(prediction_dir.glob("*.parquet")))
    universe = pd.read_parquet(run_dir / "oos_universe.parquet")
    summaries = []
    for model_id in model_ids:
        prediction_path = prediction_dir / f"{model_id}.parquet"
        if not prediction_path.exists():
            print(f"{model_id}: pooled predictions missing; skipping robustness")
            continue
        signature = _prediction_signature(prediction_path)
        monthly_path = run_dir / "robustness" / f"{model_id}_monthly.parquet"
        summary_path = run_dir / "robustness" / f"{model_id}_summary.json"
        if monthly_path.exists() and summary_path.exists():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("robustness_version") == ROBUSTNESS_VERSION and cached.get("model_signature") == signature:
                print(f"{model_id}: current robustness; skipping")
                summaries.extend(cached["rows"])
                continue
        predictions = pd.read_parquet(prediction_path)
        evaluated = merge_predictions(predictions, universe)
        pieces = [
            build_robustness_portfolios(evaluated, universe_name, weighting)
            for universe_name, weighting in ROBUSTNESS_SPECS
        ]
        monthly = pd.concat(pieces, ignore_index=True)
        rows = robustness_summary(monthly)
        for row in rows:
            row.update({"model_id": model_id, "model_signature": signature})
        write_parquet_atomic(monthly, monthly_path)
        write_json_atomic(
            {"model_id": model_id, "model_signature": signature,
             "robustness_version": ROBUSTNESS_VERSION, "rows": rows},
            summary_path,
        )
        summaries.extend(rows)
        print(f"{model_id}: saved portfolio robustness")
    result = pd.DataFrame(summaries)
    if not result.empty:
        result = result.sort_values(["model_id", "universe", "weighting"])
        result.to_csv(run_dir / "portfolio_robustness_comparison.csv", index=False)
    return result
