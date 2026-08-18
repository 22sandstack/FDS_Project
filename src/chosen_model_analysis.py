from __future__ import annotations

import io
import json
import math
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json_atomic, write_parquet_atomic
from .config import (
    CORE20, FEATURES_40, FEATURES_40_LAG1, FEATURES_40_LAG1_AVAILABLE,
    FEATURES_40_VELOCITY, ExperimentConfig,
)
from .data import load_and_prepare_panel
from .evaluation import newey_west_tstat, performance_stats
from .models import MODEL_FEATURES, MODEL_REGISTRY, build_deepset_core


ANALYSIS_VERSION = "chosen_model_analysis_v2_deepset_permutation"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def chosen_output_dir(config: ExperimentConfig, model_id: str) -> Path:
    path = config.run_dir / "chosen_model_analysis" / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_chosen_model_artifacts(config: ExperimentConfig, model_id: str) -> dict:
    """Fail early unless the chosen model has current, matching OOS artifacts."""
    from .runner import ExperimentRunner

    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown chosen model: {model_id}")
    run_dir = config.run_dir
    prediction_path = run_dir / "predictions" / f"{model_id}.parquet"
    metric_path = run_dir / "metrics" / f"{model_id}.json"
    robustness_path = run_dir / "robustness" / f"{model_id}_summary.json"
    required = (prediction_path, metric_path, robustness_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Chosen-model analysis prerequisites are missing: {missing}")
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    signature = ExperimentRunner(config)._model_signature(model_id)
    signatures = pd.read_parquet(prediction_path, columns=["model_signature"])[
        "model_signature"
    ].dropna().astype(str).unique()
    if metric.get("diagnostics_version") != ExperimentRunner.DIAGNOSTICS_VERSION:
        raise RuntimeError("Chosen model does not have current diagnostics.")
    if len(signatures) != 1 or signatures[0] != signature or metric.get("model_signature") != signature:
        raise RuntimeError("Chosen-model signatures do not match the current specification.")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if robustness.get("model_signature") != signature:
        raise RuntimeError("Chosen-model portfolio robustness is stale.")
    return {"model_id": model_id, "model_signature": signature, "metrics": metric}


def build_signal_date_regimes(config: ExperimentConfig) -> pd.DataFrame:
    """High/low trailing market volatility, classified using signal-date information only."""
    columns = ["eom", "excntry", "permno", "size_grp", "me", "ret_exc"]
    data = pd.read_parquet(config.data_path, columns=columns)
    data["eom"] = pd.to_datetime(data["eom"])
    size = data["size_grp"].astype("string").str.strip().str.lower()
    data = data[
        data["eom"].dt.year.between(config.universe.start_year, config.universe.end_year)
        & data["excntry"].eq(config.universe.country)
        & data["permno"].notna()
        & size.isin(config.universe.allowed_size_groups)
    ].copy()
    data["size_grp"] = size.loc[data.index]
    data["me"] = pd.to_numeric(data["me"], errors="coerce")
    data["ret_exc"] = pd.to_numeric(data["ret_exc"], errors="coerce")
    data = data.sort_values(["permno", "eom"])
    previous_eom = data.groupby("permno")["eom"].shift(1)
    exact_lag = previous_eom.eq(data["eom"] - pd.offsets.MonthEnd(1))
    data["beginning_me"] = data.groupby("permno")["me"].shift(1).where(exact_lag)
    usable = data[
        data["ret_exc"].notna() & np.isfinite(data["ret_exc"])
        & data["beginning_me"].notna() & (data["beginning_me"] > 0)
    ].copy()
    usable["weighted_return"] = usable["ret_exc"] * usable["beginning_me"]
    market = usable.groupby("eom").agg(
        weighted_return=("weighted_return", "sum"), market_weight=("beginning_me", "sum")
    ).reset_index()
    market["market_ret_exc"] = market["weighted_return"] / market["market_weight"]
    market = market.sort_values("eom")
    market["trailing_12m_market_vol"] = market["market_ret_exc"].rolling(12, min_periods=12).std(ddof=1)
    # The threshold excludes the current month and therefore cannot use future/test-period volatility.
    market["past_expanding_median_vol"] = market["trailing_12m_market_vol"].expanding(min_periods=60).median().shift(1)
    market["regime"] = np.where(
        market["trailing_12m_market_vol"] > market["past_expanding_median_vol"],
        "HIGH_VOL", "LOW_VOL",
    )
    market.loc[market["past_expanding_median_vol"].isna(), "regime"] = pd.NA
    return market[["eom", "market_ret_exc", "trailing_12m_market_vol", "past_expanding_median_vol", "regime"]]


def regime_stability(config: ExperimentConfig, model_id: str, regimes: pd.DataFrame) -> pd.DataFrame:
    portfolio_path = config.run_dir / "portfolios" / f"{model_id}_variants.parquet"
    ic_path = config.run_dir / "diagnostics" / f"{model_id}_monthly_rank_ic.parquet"
    portfolio = pd.read_parquet(portfolio_path).query("strategy == 'TAIL_10PCT'")
    ic = pd.read_parquet(ic_path)[["eom", "rank_ic"]]
    monthly = portfolio.merge(ic, on="eom", validate="one_to_one").merge(
        regimes, on="eom", validate="many_to_one"
    ).dropna(subset=["regime"])
    rows = []
    for regime, group in monthly.groupby("regime", sort=True):
        stats = performance_stats(group["long_short_ret"])
        rank_ic = group["rank_ic"].dropna()
        rows.append({
            "model_id": model_id, "regime": regime,
            **stats,
            "mean_rank_ic": float(rank_ic.mean()),
            "rank_ic_newey_west_t_stat": newey_west_tstat(rank_ic, 6),
            "rank_ic_positive_rate": float((rank_ic > 0).mean()),
            "mean_trailing_market_vol": float(group["trailing_12m_market_vol"].mean()),
        })
    output = chosen_output_dir(config, model_id)
    write_parquet_atomic(monthly, output / "monthly_regime_results.parquet")
    result = pd.DataFrame(rows)
    result.to_csv(output / "regime_stability_summary.csv", index=False)
    return result


def feature_importance_by_regime(
    config: ExperimentConfig, model_id: str, regimes: pd.DataFrame
) -> pd.DataFrame:
    """Compute model-appropriate OOS feature importance by signal-date regime."""
    spec = MODEL_REGISTRY[model_id]
    if spec.trainer_id == "deepset":
        return _deepset_permutation_importance_by_regime(
            config, model_id, regimes
        )
    if spec.trainer_id != "lightgbm":
        raise ValueError(
            "Regime importance supports standalone LightGBM and DeepSets models only."
        )
    features = MODEL_FEATURES.get(model_id, CORE20)
    raw_features = tuple(
        feature for feature in features
        if not feature.endswith("_lag1") and not feature.endswith("_velocity1")
        and not feature.endswith("_lag1_available")
    )
    panel, _ = load_and_prepare_panel(config, raw_features)
    runner_signature = _model_signature(config, model_id)
    regime_map = regimes.set_index("eom")["regime"]
    totals: dict[tuple[str, str], float] = {}
    counts: dict[str, int] = {}
    expected_years = list(range(
        config.universe.start_year + config.windows.train_years + config.windows.validation_years,
        config.universe.end_year + 1,
    ))
    missing_years = []
    for test_year in expected_years:
        model_path = config.run_dir / "models" / model_id / runner_signature / f"refit_{test_year}" / "model.bin"
        metadata_path = model_path.with_name("metadata.json")
        if not model_path.exists() or not metadata_path.exists():
            missing_years.append(test_year)
            continue
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fit_details = metadata.get("fit_details", {})
        selected_iteration = int(
            fit_details.get("selected_iteration", fit_details.get("best_iteration", model.best_iteration_))
        )
        test = panel[panel["eom"].dt.year.eq(test_year)]
        for eom, month in test.groupby("eom", sort=True):
            regime = regime_map.get(eom)
            if pd.isna(regime):
                continue
            contributions = model.booster_.predict(
                month[list(features)].to_numpy(np.float32),
                pred_contrib=True, num_iteration=selected_iteration,
            )[:, :-1]
            counts[str(regime)] = counts.get(str(regime), 0) + len(month)
            for feature, value in zip(features, np.abs(contributions).sum(axis=0)):
                key = (str(regime), feature)
                totals[key] = totals.get(key, 0.0) + float(value)
    if missing_years:
        raise FileNotFoundError(
            f"Missing OOS model checkpoints for {model_id}: {missing_years}"
        )
    rows = [
        {"model_id": model_id, "regime": regime, "feature": feature,
         "mean_abs_contribution": total / counts[regime]}
        for (regime, feature), total in totals.items()
    ]
    result = pd.DataFrame(rows)
    if not result.empty:
        result["importance_share"] = result.groupby("regime")["mean_abs_contribution"].transform(
            lambda values: values / values.sum()
        )
        result["importance_rank"] = result.groupby("regime")["mean_abs_contribution"].rank(
            ascending=False, method="first"
        ).astype(int)
        result = result.sort_values(["regime", "importance_rank"])
    output = chosen_output_dir(config, model_id) / "feature_importance_by_regime.csv"
    result.to_csv(output, index=False)
    return result


def _deepset_permutation_importance_by_regime(
    config: ExperimentConfig, model_id: str, regimes: pd.DataFrame
) -> pd.DataFrame:
    """Grouped OOS permutation importance for a fitted dynamic DeepSets model.

    Current, lagged and velocity coordinates belonging to one underlying
    characteristic are permuted together within each month. This preserves the
    monthly marginal distribution and measures the fitted model's total reliance
    on that characteristic family without refitting or affecting model selection.
    """
    import torch

    features = MODEL_FEATURES[model_id]
    if not set(FEATURES_40).issubset(features):
        raise ValueError("DeepSets regime importance expects the frozen 40-characteristic set.")
    panel, _ = load_and_prepare_panel(
        config,
        FEATURES_40,
        include_core_dynamics=False,
        include_feature40_lag1=True,
        include_core_lag2=False,
        include_feature40_lag2=False,
    )
    first_test_year = (
        config.universe.start_year
        + config.windows.train_years
        + config.windows.validation_years
    )
    panel = panel.loc[panel["eom"].dt.year.ge(first_test_year)].copy()
    regime_map = regimes.set_index("eom")["regime"]
    signature = _model_signature(config, model_id)
    spec = MODEL_REGISTRY[model_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_position = {feature: index for index, feature in enumerate(features)}
    families = {
        characteristic: tuple(
            feature_position[name]
            for name in (
                characteristic,
                FEATURES_40_LAG1[index],
                FEATURES_40_VELOCITY[index],
            )
            if name in feature_position
        )
        for index, characteristic in enumerate(FEATURES_40)
    }
    if FEATURES_40_LAG1_AVAILABLE in feature_position:
        families["lag1_availability"] = (
            feature_position[FEATURES_40_LAG1_AVAILABLE],
        )

    aggregates: dict[tuple[str, str], dict[str, float]] = {}
    expected_years = range(first_test_year, config.universe.end_year + 1)
    for test_year in expected_years:
        model_path = (
            config.run_dir / "models" / model_id / signature
            / f"refit_{test_year}" / "model.bin"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing OOS checkpoint: {model_path}")
        model = build_deepset_core(len(features), spec.params).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        test = panel.loc[panel["eom"].dt.year.eq(test_year)]
        with torch.inference_mode():
            for month_number, (eom, month) in enumerate(
                test.groupby("eom", sort=True)
            ):
                regime = regime_map.get(eom)
                if pd.isna(regime):
                    continue
                x_array = month[list(features)].to_numpy(np.float32)
                y = month[config.target_col].to_numpy(np.float64)
                valid = np.isfinite(y)
                if valid.sum() < 20:
                    continue
                x = torch.from_numpy(x_array).to(device)
                baseline = model(x).float().cpu().numpy().astype(np.float64)
                baseline_squared_error = float(
                    np.square(y[valid] - baseline[valid]).sum()
                )
                for family_number, (family, columns) in enumerate(families.items()):
                    rng = np.random.default_rng(
                        config.seed + test_year * 10_000
                        + month_number * 100 + family_number
                    )
                    permutation = rng.permutation(len(month))
                    permuted = x_array.copy()
                    permuted[:, columns] = x_array[permutation][:, columns]
                    prediction = model(
                        torch.from_numpy(permuted).to(device)
                    ).float().cpu().numpy().astype(np.float64)
                    permuted_squared_error = float(
                        np.square(y[valid] - prediction[valid]).sum()
                    )
                    key = (str(regime), family)
                    item = aggregates.setdefault(
                        key,
                        {"delta_squared_error": 0.0, "baseline_squared_error": 0.0,
                         "n": 0.0, "months": 0.0},
                    )
                    item["delta_squared_error"] += (
                        permuted_squared_error - baseline_squared_error
                    )
                    item["baseline_squared_error"] += baseline_squared_error
                    item["n"] += int(valid.sum())
                    item["months"] += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for (regime, feature), item in aggregates.items():
        rows.append({
            "model_id": model_id,
            "regime": regime,
            "feature": feature,
            "permutation_mse_increase": item["delta_squared_error"] / item["n"],
            "relative_mse_increase": (
                item["delta_squared_error"] / item["baseline_squared_error"]
                if item["baseline_squared_error"] > 0 else np.nan
            ),
            "n_stock_observations": int(item["n"]),
            "n_months": int(item["months"]),
        })
    result = pd.DataFrame(rows)
    result["positive_importance"] = result["permutation_mse_increase"].clip(lower=0.0)
    positive_total = result.groupby("regime")["positive_importance"].transform("sum")
    result["importance_share"] = np.where(
        positive_total > 0,
        result["positive_importance"] / positive_total,
        0.0,
    )
    result["importance_rank"] = result.groupby("regime")[
        "permutation_mse_increase"
    ].rank(ascending=False, method="first").astype(int)
    result = result.sort_values(["regime", "importance_rank"])
    output = chosen_output_dir(config, model_id)
    result.to_csv(output / "feature_importance_by_regime.csv", index=False)
    write_json_atomic(
        {
            "analysis_version": ANALYSIS_VERSION,
            "method": "within_month_grouped_permutation_mse",
            "grouping": "current_lag1_velocity_by_underlying_characteristic",
            "permutations_per_month": 1,
            "seed": config.seed,
            "uses_test_returns_for_ex_post_interpretation_only": True,
        },
        output / "feature_importance_method.json",
    )
    return result


def _model_signature(config: ExperimentConfig, model_id: str) -> str:
    # Imported locally to avoid duplicating or changing the training signature definition.
    from .runner import ExperimentRunner
    return ExperimentRunner(config)._model_signature(model_id)


def _download_french_zip(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")


def _parse_french_monthly(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.lstrip().startswith(","))
    header = "date" + lines[header_index]
    monthly = [line for line in lines[header_index + 1:] if re.match(r"^\s*\d{6}\s*,", line)]
    data = pd.read_csv(io.StringIO("\n".join([header] + monthly)))
    data.columns = [str(column).strip().lower().replace("-", "_") for column in data.columns]
    data["eom"] = pd.to_datetime(data["date"].astype(str).str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    for column in data.columns.difference(["date", "eom"]):
        data[column] = pd.to_numeric(data[column], errors="coerce") / 100.0
    return data.drop(columns="date")


def download_ff5_momentum() -> pd.DataFrame:
    ff5 = _parse_french_monthly(_download_french_zip(FF5_URL))
    momentum = _parse_french_monthly(_download_french_zip(MOM_URL))
    momentum_column = next(column for column in momentum.columns if column != "eom")
    momentum = momentum.rename(columns={momentum_column: "mom"})
    factors = ff5.merge(momentum[["eom", "mom"]], on="eom", validate="one_to_one")
    required = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    missing = [column for column in required if column not in factors]
    if missing:
        raise ValueError(f"French factor download is missing columns: {missing}")
    return factors[["eom"] + required].sort_values("eom")


def _hac_ols(y: np.ndarray, x: np.ndarray, lags: int = 6) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, len(y) - 1) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1.0)) * (cross + cross.T)
    covariance = bread @ meat @ bread
    return beta, np.sqrt(np.maximum(np.diag(covariance), 0.0))


def factor_decomposition(
    config: ExperimentConfig, model_id: str, factors: pd.DataFrame
) -> pd.DataFrame:
    variants = pd.read_parquet(config.run_dir / "portfolios" / f"{model_id}_variants.parquet")
    returns = variants.query("strategy == 'TAIL_10PCT'")[["eom", "long_short_ret"]]
    # A signal dated at month-end t earns y_true during t+1, so factor timing
    # must use the following calendar month rather than the signal month.
    returns["factor_eom"] = returns["eom"] + pd.offsets.MonthEnd(1)
    data = returns.merge(
        factors.rename(columns={"eom": "factor_eom"}),
        on="factor_eom", validate="one_to_one",
    ).dropna()
    names = ["alpha", "mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    x = np.column_stack([np.ones(len(data)), data[names[1:]].to_numpy(float)])
    beta, standard_error = _hac_ols(data["long_short_ret"].to_numpy(float), x, 6)
    result = pd.DataFrame({
        "coefficient": names, "monthly_estimate": beta,
        "hac_standard_error": standard_error, "hac_t_stat": beta / standard_error,
    })
    result["annualized_estimate"] = result["monthly_estimate"]
    result.loc[result["coefficient"].eq("alpha"), "annualized_estimate"] *= 12.0
    output = chosen_output_dir(config, model_id)
    result.to_csv(output / "ff5_momentum_decomposition.csv", index=False)
    write_parquet_atomic(data, output / "factor_regression_monthly_data.parquet")
    return result


def long_short_and_cost_attribution(config: ExperimentConfig, model_id: str) -> tuple[pd.DataFrame, str]:
    variants = pd.read_parquet(config.run_dir / "portfolios" / f"{model_id}_variants.parquet")
    tails = variants.query("strategy == 'TAIL_10PCT'")
    attribution = pd.DataFrame([
        {"component": "long_leg", "mean_monthly_contribution": tails["long_ret"].fillna(0.0).mean()},
        {"component": "short_leg", "mean_monthly_contribution": -tails["short_ret"].fillna(0.0).mean()},
        {"component": "long_short", "mean_monthly_contribution": tails["long_short_ret"].mean()},
    ])
    attribution["annualized_contribution"] = 12.0 * attribution["mean_monthly_contribution"]
    robustness_path = config.run_dir / "robustness" / f"{model_id}_summary.json"
    if not robustness_path.exists():
        raise FileNotFoundError("Run the portfolio implementability robustness stage first.")
    robustness = pd.DataFrame(json.loads(robustness_path.read_text(encoding="utf-8"))["rows"])
    full_equal = robustness.query("universe == 'FULL' and weighting == 'EQUAL'").iloc[0]
    break_even_bps = (
        10_000.0 * full_equal["gross_mean_monthly_return"] / full_equal["mean_monthly_turnover"]
        if full_equal["mean_monthly_turnover"] > 0 else np.nan
    )
    scenario_text = "; ".join(
        f"{row.universe}/{row.weighting}: 25 bps annualized return "
        f"{row.net_25bps_annualized_return:.2%}, Sharpe {row.net_25bps_sharpe:.2f}"
        for row in robustness.itertuples(index=False)
    )
    discussion = (
        f"The full-universe equal-weight 10% portfolio turns over "
        f"{full_equal['mean_monthly_turnover']:.3f} per month. Its gross annualized return is "
        f"{full_equal['gross_annualized_return']:.2%}; the 25 bps scenario produces "
        f"{full_equal['net_25bps_annualized_return']:.2%} with a Sharpe of "
        f"{full_equal['net_25bps_sharpe']:.2f}. The simple proportional-cost break-even estimate is "
        f"{break_even_bps:.1f} bps per unit of turnover. These scenarios exclude stock-borrow fees, "
        "bid-ask heterogeneity and nonlinear market impact, so they are sensitivity analysis rather than "
        f"a claim of fully implementable net performance. At 25 bps, the robustness matrix is: {scenario_text}."
    )
    output = chosen_output_dir(config, model_id)
    attribution.to_csv(output / "long_short_attribution.csv", index=False)
    robustness.to_csv(output / "transaction_cost_robustness.csv", index=False)
    (output / "transaction_cost_discussion.md").write_text(discussion, encoding="utf-8")
    return attribution, discussion
