from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json_atomic
from .config import ExperimentConfig
from .evaluation import form_portfolio_variants
from .model_comparison import DEFAULT_MODEL_PAIRS
from .models import MODEL_REGISTRY
from .runner import ExperimentRunner


AUDIT_VERSION = "final_project_audit_v2_deepset"
REQUIRED_PREDICTION_COLUMNS = {
    "eom", "target_date", "id", "permno", "country", "y_true", "y_pred",
    "me", "size_grp", "target_available", "model_id", "model_signature",
    "test_year", "refit_id",
}


def _check(rows: list[dict], area: str, check: str, passed: bool, detail: str) -> None:
    rows.append({"area": area, "check": check, "passed": bool(passed), "detail": detail})


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_final_project_audit(
    config: ExperimentConfig, model_ids: tuple[str, ...], chosen_model_id: str
) -> pd.DataFrame:
    """Validate final saved artifacts and write a compact audit report."""
    runner = ExperimentRunner(config)
    rows: list[dict] = []
    run_dir = config.run_dir
    expected_years = list(range(
        config.universe.start_year + config.windows.train_years + config.windows.validation_years,
        config.universe.end_year + 1,
    ))
    expected_months = 12 * len(expected_years)
    schedule_path = run_dir / "rolling_schedule.parquet"
    if schedule_path.exists():
        schedule = pd.read_parquet(schedule_path)
        no_overlap = bool((schedule.train_end_year < schedule.validation_start_year).all()
                          and (schedule.validation_end_year < schedule.test_start_year).all())
        exact_windows = bool(
            ((schedule.train_end_year - schedule.train_start_year + 1) == config.windows.train_years).all()
            and ((schedule.validation_end_year - schedule.validation_start_year + 1) == config.windows.validation_years).all()
            and ((schedule.test_end_year - schedule.test_start_year + 1) == 1).all()
        )
        _check(rows, "schedule", "no_split_overlap", no_overlap, f"{len(schedule)} annual rows")
        _check(rows, "schedule", "exact_window_lengths", exact_windows, "15/4/1 expected")
    else:
        _check(rows, "schedule", "rolling_schedule_exists", False, str(schedule_path))

    availability_path = run_dir / "target_availability_summary.csv"
    _check(
        rows, "data", "target_availability_summary_exists",
        availability_path.exists(), str(availability_path),
    )

    for model_id in model_ids:
        area = f"model:{model_id}"
        if model_id not in MODEL_REGISTRY:
            _check(rows, area, "registered", False, "unknown model")
            continue
        signature = runner._model_signature(model_id)
        prediction_path = run_dir / "predictions" / f"{model_id}.parquet"
        metric_path = run_dir / "metrics" / f"{model_id}.json"
        _check(rows, area, "prediction_exists", prediction_path.exists(), str(prediction_path))
        _check(rows, area, "metric_exists", metric_path.exists(), str(metric_path))
        if not prediction_path.exists() or not metric_path.exists():
            continue
        predictions = pd.read_parquet(prediction_path)
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        columns_ok = REQUIRED_PREDICTION_COLUMNS.issubset(predictions.columns)
        _check(rows, area, "prediction_schema", columns_ok,
               f"missing={sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))}")
        unique_keys = not predictions.duplicated(["eom", "permno"]).any()
        _check(rows, area, "unique_month_permno", unique_keys, f"rows={len(predictions)}")
        signatures = predictions.model_signature.dropna().astype(str).unique()
        signature_ok = len(signatures) == 1 and signatures[0] == signature and metric.get("model_signature") == signature
        _check(rows, area, "signature_current", signature_ok, f"expected={signature}")
        _check(rows, area, "diagnostics_current",
               metric.get("diagnostics_version") == runner.DIAGNOSTICS_VERSION,
               str(metric.get("diagnostics_version")))
        _check(
            rows, area, "missing_return_stress_reported",
            "tail_10pct_missing_return_stress_annualized_return" in metric,
            "adverse within-month 1st/99th percentile sensitivity",
        )
        months = pd.to_datetime(predictions.eom).nunique()
        _check(rows, area, "complete_oos_calendar", months == expected_months,
               f"months={months}, expected={expected_months}")
        years = sorted(pd.Series(predictions.test_year).dropna().astype(int).unique())
        _check(rows, area, "complete_test_years", years == expected_years,
               f"years={years[0] if years else None}-{years[-1] if years else None}")
        timing = (
            pd.to_datetime(predictions.target_date)
            == pd.to_datetime(predictions.eom) + pd.offsets.MonthEnd(1)
        ).all()
        _check(rows, area, "one_month_target_timing", bool(timing), "target_date=eom+MonthEnd(1)")
        availability = predictions.target_available.astype(bool).eq(
            pd.to_numeric(predictions.y_true, errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
        ).all()
        _check(rows, area, "target_availability_consistent", bool(availability), "flag matches finite y_true")
        refit_root = run_dir / "models" / model_id / signature
        completed = [year for year in expected_years if (refit_root / f"refit_{year}" / "COMPLETED").exists()]
        _check(rows, area, "all_refits_complete", completed == expected_years,
               f"complete={len(completed)}, expected={len(expected_years)}")
        variants_path = run_dir / "portfolios" / f"{model_id}_variants.parquet"
        if variants_path.exists() and columns_ok:
            rebuilt = form_portfolio_variants(predictions)
            saved = pd.read_parquet(variants_path)
            keys = ["eom", "strategy"]
            paired = saved[keys + ["long_short_ret"]].merge(
                rebuilt[keys + ["long_short_ret"]], on=keys, suffixes=("_saved", "_rebuilt"), validate="one_to_one"
            )
            difference = np.nanmax(np.abs(paired.long_short_ret_saved - paired.long_short_ret_rebuilt))
            portfolio_ok = len(paired) == len(saved) == len(rebuilt) and difference < 1e-12
            _check(rows, area, "portfolio_reconstructs", portfolio_ok, f"max_abs_diff={difference:.3g}")
        else:
            _check(rows, area, "portfolio_reconstructs", False, "missing variants or prediction columns")

    comparison_path = run_dir / "comparisons" / "paired_model_tests.csv"
    if comparison_path.exists():
        comparisons = pd.read_csv(comparison_path)
        observed = list(zip(comparisons.model_a, comparisons.model_b))
        _check(rows, "comparisons", "declared_pairs_complete", observed == list(DEFAULT_MODEL_PAIRS),
               f"observed={observed}")
    else:
        _check(rows, "comparisons", "declared_pairs_complete", False, str(comparison_path))

    chosen_dir = run_dir / "chosen_model_analysis" / chosen_model_id
    chosen_required = (
        "regime_stability_summary.csv", "feature_importance_by_regime.csv",
        "feature_importance_method.json", "ff5_momentum_decomposition.csv",
        "long_short_attribution.csv", "transaction_cost_robustness.csv",
    )
    for name in chosen_required:
        _check(rows, "chosen_model", name, (chosen_dir / name).exists(), str(chosen_dir / name))

    result = pd.DataFrame(rows)
    output = run_dir / "finalization"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "final_project_audit.csv", index=False)
    write_json_atomic({
        "audit_version": AUDIT_VERSION,
        "passed": bool(result.passed.all()),
        "n_checks": int(len(result)),
        "n_failed": int((~result.passed).sum()),
    }, output / "final_project_audit_summary.json")
    return result


def write_frozen_manifest(
    config: ExperimentConfig, model_ids: tuple[str, ...], chosen_model_id: str
) -> dict:
    runner = ExperimentRunner(config)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.project_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=config.project_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        git_status = "git status unavailable"
    requirements_path = config.project_dir / "requirements.txt"
    source_files = sorted((config.project_dir / "src").glob("*.py"))
    payload = {
        "manifest_version": "final_research_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_model_id": chosen_model_id,
        "chosen_model_signature": runner._model_signature(chosen_model_id),
        "model_signatures": {model: runner._model_signature(model) for model in model_ids},
        "data_file": config.data_path.name,
        "data_sha256": _sha256(config.data_path),
        "experiment": config.to_dict(),
        "diagnostics_version": runner.DIAGNOSTICS_VERSION,
        "comparison_pairs": list(DEFAULT_MODEL_PAIRS),
        "requirements_sha256": _sha256(requirements_path),
        "git_commit": commit,
        "git_worktree_clean": git_status == "",
        "source_sha256": {
            str(path.relative_to(config.project_dir)): _sha256(path)
            for path in source_files
        },
    }
    output = config.run_dir / "finalization" / "frozen_research_manifest.json"
    write_json_atomic(payload, output)
    return payload


def generate_report_outputs(
    config: ExperimentConfig, chosen_model_id: str,
    comparison_models: tuple[str, ...] = (
        "LASSO_20", "LGBM_20", "XGBOOST_20", "NN2_20", "NN2_40", "NN3_20", "NN4_20", "NN4_40",
        "DEEPSET_20", "DEEPSET_20_LAG1", "DEEPSET_20_DYNAMIC",
        "LGBM_40",
        "LGBM_20_LAG1", "LGBM_20_LAG2", "LGBM_40_LAG1", "LGBM_40_LAG2",
        "MLP_40", "DEEPSET_40", "HYBRID_MLP40_DEEPSET40",
        "DEEPSET_40_LAG1", "DEEPSET_40_DYNAMIC",
    ),
) -> dict[str, Path]:
    """Create a compact, frozen set of report tables and figures from saved OOS files."""
    import matplotlib.pyplot as plt

    run_dir = config.run_dir
    output = run_dir / "report_outputs"
    output.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(run_dir / "model_comparison.csv")
    main_columns = [
        "model_id", "pooled_oos_r2", "robust_oos_r2", "mean_monthly_rank_ic",
        "annualized_return", "annualized_volatility", "sharpe",
        "newey_west_t_stat", "max_drawdown", "hit_rate", "n_months",
    ]
    main = comparison[comparison.model_id.isin(comparison_models)][main_columns].sort_values("sharpe", ascending=False)
    main.to_csv(output / "table_1_main_model_comparison.csv", index=False)

    ladder_ids = ["LGBM_20", "LGBM_40", "LGBM_60", "LGBM_80", "LGBM_100"]
    ladder = comparison.set_index("model_id").loc[ladder_ids].reset_index()
    ladder.to_csv(output / "table_2_feature_expansion_ladder.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot([20, 40, 60, 80, 100], ladder.pooled_oos_r2 * 100, marker="o")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(xlabel="Number of characteristics", ylabel="OOS R² (%)", title="Statistical fit")
    axes[1].plot([20, 40, 60, 80, 100], ladder.sharpe, marker="o", color="#b45309")
    axes[1].set(xlabel="Number of characteristics", ylabel="Sharpe ratio", title="Economic performance")
    fig.tight_layout(); fig.savefig(output / "figure_1_feature_expansion.png", dpi=200); plt.close(fig)

    cumulative_ids = ("LGBM_20", "LGBM_40", "DEEPSET_20", "DEEPSET_20_DYNAMIC")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    cumulative_table = None
    for model_id in cumulative_ids:
        returns = pd.read_parquet(run_dir / "portfolios" / f"{model_id}_variants.parquet").query("strategy == 'TAIL_10PCT'")
        series = returns.set_index("eom").long_short_ret.sort_index()
        wealth = (1 + series).cumprod()
        ax.plot(wealth.index, wealth, label=model_id)
        named = wealth.rename(model_id)
        cumulative_table = named.to_frame() if cumulative_table is None else cumulative_table.join(named, how="outer")
    cumulative_table.to_csv(output / "figure_2_cumulative_returns_data.csv")
    ax.set(title="Gross equal-weight 10% tail portfolios", ylabel="Growth of 1", xlabel="Signal month")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(output / "figure_2_cumulative_returns.png", dpi=200); plt.close(fig)

    chosen = run_dir / "chosen_model_analysis" / chosen_model_id
    regime = pd.read_csv(chosen / "regime_stability_summary.csv")
    importance = pd.read_csv(chosen / "feature_importance_by_regime.csv")
    top_features = (
        importance.groupby("feature")["importance_share"].mean().nlargest(10).index
    )
    importance_table = importance[importance.feature.isin(top_features)].copy()
    regime.to_csv(output / "table_3_regime_stability.csv", index=False)
    importance_table.to_csv(output / "table_4_feature_importance_by_regime.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    regime.set_index("regime")[["annualized_return", "sharpe"]].plot.bar(ax=axes[0])
    axes[0].set(title="Performance by volatility regime", xlabel="", ylabel="Value")
    pivot = importance_table.pivot(
        index="feature", columns="regime", values="importance_share"
    )
    pivot.plot.barh(ax=axes[1])
    axes[1].set(title="Grouped permutation importance", xlabel="Importance share")
    fig.tight_layout(); fig.savefig(output / "figure_3_regime_and_feature_importance.png", dpi=200); plt.close(fig)

    factors = pd.read_csv(chosen / "ff5_momentum_decomposition.csv")
    costs = pd.read_csv(chosen / "transaction_cost_robustness.csv")
    factors.to_csv(output / "table_5_factor_decomposition.csv", index=False)
    costs.to_csv(output / "table_6_implementability.csv", index=False)
    return {path.name: path for path in sorted(output.iterdir())}
