from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_parquet_atomic
from .evaluation import clustered_mean_tstat, newey_west_tstat


DEFAULT_MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("NN4_20", "NN2_20"),
    ("NN2_40", "NN2_20"),
    ("NN2_40", "MLP_40"),
    ("NN4_20", "NN3_20"),
    ("LGBM_40", "LGBM_20"),
    ("LGBM_40", "LGBM_40_LAG1"),
    ("LGBM_20_LAG1", "LGBM_20_LAG2"),
    ("LGBM_40", "LGBM_40_LAG2"),
    ("LGBM_40_LAG1", "LGBM_40_LAG2"),
    ("MLP_40", "DEEPSET_40"),
    ("HYBRID_MLP40_DEEPSET40", "MLP_40"),
    ("HYBRID_MLP40_DEEPSET40", "DEEPSET_40"),
    ("DEEPSET_40", "DEEPSET_40_LAG1"),
    ("DEEPSET_40", "DEEPSET_40_DYNAMIC"),
    ("DEEPSET_40_LAG1", "DEEPSET_40_DYNAMIC"),
    ("DEEPSET_20", "DEEPSET_20_LAG1"),
    ("DEEPSET_20", "DEEPSET_20_DYNAMIC"),
    ("DEEPSET_20_LAG1", "DEEPSET_20_DYNAMIC"),
    ("HYBRID_LGBM20_DEEPSET20", "LGBM_20"),
    ("HYBRID_LGBM20_DEEPSET20", "DEEPSET_20"),
    ("HYBRID_LGBM40_DEEPSET40_DYNAMIC", "LGBM_40"),
    ("HYBRID_LGBM40_DEEPSET40_DYNAMIC", "LGBM_20"),
    ("HYBRID_LGBM40_DEEPSET40_DYNAMIC", "DEEPSET_40_DYNAMIC"),
    ("HYBRID_LGBM40_DEEPSET40_DYNAMIC", "HYBRID_LGBM40_DEEPSET40"),
    ("HYBRID_LGBM40_DEEPSET40_DYNAMIC", "HYBRID_MLP40_DEEPSET40"),
)


def _normal_two_sided_pvalue(t_stat: float) -> float:
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0))) if np.isfinite(t_stat) else np.nan


def _holm_adjust(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    running = 0.0
    m = len(valid)
    for order, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, (m - order) * float(value)))
        adjusted.loc[index] = running
    return adjusted


def _annualized_sharpe(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    volatility = values.std(ddof=1)
    return float(math.sqrt(12.0) * values.mean() / volatility) if volatility > 0 else np.nan


def _year_block_sharpe_interval(
    paired: pd.DataFrame, draws: int, seed: int
) -> tuple[float, float]:
    years = paired["test_year"].dropna().unique()
    if len(years) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    differences = []
    blocks = {year: paired[paired["test_year"] == year] for year in years}
    for _ in range(draws):
        sampled = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat([blocks[year] for year in sampled], ignore_index=True)
        differences.append(
            _annualized_sharpe(sample["value_a"]) - _annualized_sharpe(sample["value_b"])
        )
    return tuple(float(x) for x in np.nanquantile(differences, [0.025, 0.975]))


def _paired_series(path_a: Path, path_b: Path, value: str) -> pd.DataFrame:
    a = pd.read_parquet(path_a, columns=["eom", "test_year", value]).rename(columns={value: "value_a"})
    b = pd.read_parquet(path_b, columns=["eom", "test_year", value]).rename(columns={value: "value_b"})
    return a.merge(b, on=["eom", "test_year"], validate="one_to_one").dropna()


def run_paired_model_comparisons(
    run_dir: Path,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_MODEL_PAIRS,
    newey_west_lags: int = 6,
    bootstrap_draws: int = 2000,
    seed: int = 42,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Run the declared paired tests using saved monthly OOS diagnostics only."""
    from .runner import ExperimentRunner

    run_dir = Path(run_dir)
    rows = []
    for pair_number, (model_a, model_b) in enumerate(pairs):
        for model_id in (model_a, model_b):
            metric_path = run_dir / "metrics" / f"{model_id}.json"
            prediction_path = run_dir / "predictions" / f"{model_id}.parquet"
            if not metric_path.exists() or not prediction_path.exists():
                raise FileNotFoundError(f"Missing current metric/prediction artifact for {model_id}.")
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            if metric.get("diagnostics_version") != ExperimentRunner.DIAGNOSTICS_VERSION:
                raise RuntimeError(f"{model_id} does not have current signal-aware diagnostics.")
            signatures = pd.read_parquet(prediction_path, columns=["model_signature"])[
                "model_signature"
            ].dropna().astype(str).unique()
            if len(signatures) != 1 or metric.get("model_signature") != signatures[0]:
                raise RuntimeError(f"Signature mismatch for {model_id} comparison artifacts.")
        mechanism_a = run_dir / "diagnostics" / f"{model_a}_monthly_mechanisms.parquet"
        mechanism_b = run_dir / "diagnostics" / f"{model_b}_monthly_mechanisms.parquet"
        rank_a = run_dir / "diagnostics" / f"{model_a}_monthly_rank_ic.parquet"
        rank_b = run_dir / "diagnostics" / f"{model_b}_monthly_rank_ic.parquet"
        portfolio_a = run_dir / "portfolios" / f"{model_a}_variants.parquet"
        portfolio_b = run_dir / "portfolios" / f"{model_b}_variants.parquet"
        required = (mechanism_a, mechanism_b, rank_a, rank_b, portfolio_a, portfolio_b)
        if not all(path.exists() for path in required):
            if require_complete:
                missing = [str(path) for path in required if not path.exists()]
                raise FileNotFoundError(
                    f"Model pair {model_a} vs {model_b} is incomplete: {missing}"
                )
            continue

        mse = _paired_series(mechanism_a, mechanism_b, "monthly_mse")
        mse["difference"] = mse["value_a"] - mse["value_b"]
        rank = _paired_series(rank_a, rank_b, "rank_ic")
        rank["difference"] = rank["value_a"] - rank["value_b"]
        pa = pd.read_parquet(portfolio_a).query("strategy == 'TAIL_10PCT'")
        pb = pd.read_parquet(portfolio_b).query("strategy == 'TAIL_10PCT'")
        returns = pa[["eom", "test_year", "long_short_ret"]].rename(columns={"long_short_ret": "value_a"}).merge(
            pb[["eom", "test_year", "long_short_ret"]].rename(columns={"long_short_ret": "value_b"}),
            on=["eom", "test_year"], validate="one_to_one",
        ).dropna()
        returns["difference"] = returns["value_a"] - returns["value_b"]
        mse_t = newey_west_tstat(mse["difference"], newey_west_lags)
        return_t = newey_west_tstat(returns["difference"], newey_west_lags)
        rank_t = clustered_mean_tstat(rank["difference"], rank["test_year"])
        ci_low, ci_high = _year_block_sharpe_interval(
            returns, bootstrap_draws, seed + pair_number
        )
        sharpe_difference = (
            _annualized_sharpe(returns["value_a"])
            - _annualized_sharpe(returns["value_b"])
        )
        bootstrap_standard_error = (ci_high - ci_low) / (2.0 * 1.959964)
        rows.append({
            "model_a": model_a, "model_b": model_b,
            "difference_definition": "model_a_minus_model_b",
            "mean_monthly_mse_difference": float(mse["difference"].mean()),
            "mse_difference_nw_t_stat": mse_t,
            "mse_difference_p_value": _normal_two_sided_pvalue(mse_t),
            "mean_monthly_return_difference": float(returns["difference"].mean()),
            "return_difference_nw_t_stat": return_t,
            "return_difference_p_value": _normal_two_sided_pvalue(return_t),
            "mean_rank_ic_difference": float(rank["difference"].mean()),
            "rank_ic_difference_clustered_t_stat": rank_t,
            "rank_ic_difference_p_value": _normal_two_sided_pvalue(rank_t),
            "sharpe_difference": sharpe_difference,
            "sharpe_difference_bootstrap_standard_error": bootstrap_standard_error,
            "sharpe_difference_exceeds_one_standard_error": bool(
                sharpe_difference > bootstrap_standard_error
            ),
            "sharpe_difference_block_bootstrap_ci_low": ci_low,
            "sharpe_difference_block_bootstrap_ci_high": ci_high,
            "simpler_model_within_sharpe_interval": bool(ci_low <= 0.0 <= ci_high),
            "n_paired_months": int(len(returns)),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for family in ("mse_difference", "return_difference", "rank_ic_difference"):
        result[f"{family}_holm_p_value"] = _holm_adjust(result[f"{family}_p_value"])
    output = run_dir / "comparisons" / "paired_model_tests.parquet"
    write_parquet_atomic(result, output)
    result.to_csv(output.with_suffix(".csv"), index=False)
    return result
