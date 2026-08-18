from __future__ import annotations

import math

import numpy as np
import pandas as pd


MINIMUM_UNIQUE_PREDICTIONS = 2


def has_cross_sectional_signal(
    month: pd.DataFrame, minimum_unique: int = MINIMUM_UNIQUE_PREDICTIONS
) -> bool:
    """Return whether a month contains a usable cross-sectional forecast."""
    prediction = pd.to_numeric(month["y_pred"], errors="coerce")
    prediction = prediction[np.isfinite(prediction)]
    return prediction.nunique() >= minimum_unique


def fractional_tail_membership(
    month: pd.DataFrame, tail_fraction: float,
) -> pd.DataFrame:
    """Assign equal fractional membership when a prediction tie crosses a tail boundary."""
    if not 0.0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must lie between zero and one half.")
    result = month.copy()
    n = len(result)
    target_mass = float(max(1, math.ceil(n * tail_fraction)))
    result["short_membership"] = 0.0
    result["long_membership"] = 0.0
    groups = [index for _, index in result.groupby("y_pred", sort=True).groups.items()]
    remaining = target_mass
    for index in groups:
        allocation = min(remaining, float(len(index)))
        if allocation > 0:
            result.loc[index, "short_membership"] = allocation / len(index)
            remaining -= allocation
        if remaining <= 0:
            break
    remaining = target_mass
    for index in reversed(groups):
        allocation = min(remaining, float(len(index)))
        if allocation > 0:
            result.loc[index, "long_membership"] = allocation / len(index)
            remaining -= allocation
        if remaining <= 0:
            break
    return result


def tied_rank_score(prediction: pd.Series) -> pd.Series:
    """Map average prediction ranks to [-1, 1] without identifier-based ordering."""
    n = len(prediction)
    if n < 2:
        return pd.Series(0.0, index=prediction.index)
    rank = prediction.rank(method="average") - 1.0
    return 2.0 * rank / (n - 1.0) - 1.0


def fractional_quantile_membership(
    month: pd.DataFrame, n_groups: int,
) -> pd.DataFrame:
    """Allocate tied prediction groups fractionally across quantile boundaries."""
    result = month.copy()
    columns = [f"membership_{group}" for group in range(1, n_groups + 1)]
    result[columns] = 0.0
    n = len(result)
    cursor = 0.0
    for _, index in result.groupby("y_pred", sort=True).groups.items():
        group_size = float(len(index))
        group_start, group_end = cursor, cursor + group_size
        for group in range(1, n_groups + 1):
            bucket_start = n * (group - 1) / n_groups
            bucket_end = n * group / n_groups
            overlap = max(0.0, min(group_end, bucket_end) - max(group_start, bucket_start))
            if overlap > 0:
                result.loc[index, f"membership_{group}"] = overlap / group_size
        cursor = group_end
    return result


def monthly_signal_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Record forecast dispersion and explicitly identify no-signal months."""
    rows = []
    for eom, month in data.groupby("eom", sort=True):
        prediction = pd.to_numeric(month["y_pred"], errors="coerce")
        finite = prediction[np.isfinite(prediction)]
        unique = int(finite.nunique())
        rows.append({
            "eom": eom,
            "test_year": int(month["test_year"].iloc[0]) if "test_year" in month else int(pd.Timestamp(eom).year),
            "n_finite_predictions": int(len(finite)),
            "n_unique_predictions": unique,
            "prediction_std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            "signal_available": bool(unique >= MINIMUM_UNIQUE_PREDICTIONS),
        })
    monthly = pd.DataFrame(rows)
    valid = int(monthly["signal_available"].sum()) if not monthly.empty else 0
    total = int(len(monthly))
    return monthly, {
        "n_signal_months": valid,
        "n_no_signal_months": total - valid,
        "n_constant_prediction_months": int(
            monthly["n_unique_predictions"].lt(2).sum()
        ) if total else 0,
        "signal_month_rate": float(valid / total) if total else np.nan,
        "minimum_monthly_unique_predictions": (
            int(monthly["n_unique_predictions"].min()) if total else np.nan
        ),
    }

def merge_predictions(predictions: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    keys = ["eom", "id", "permno", "test_year", "refit_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("Predictions contain duplicate observation keys.")
    if universe.duplicated(keys).any():
        raise ValueError("OOS universe contains duplicate observation keys.")
    # New pooled files are self-contained, while legacy annual artifacts hold
    # only forecasts. Always take evaluation metadata from the frozen universe
    # so both formats remain compatible and cannot silently disagree.
    overlap = [
        column for column in universe.columns
        if column in predictions.columns and column not in keys
    ]
    forecast = predictions.drop(columns=overlap)
    merged = forecast.merge(
        universe, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        counts = merged.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(f"Prediction/universe key coverage mismatch: {counts}")
    return merged.drop(columns="_merge")


def oos_r2(data: pd.DataFrame) -> float:
    valid = data[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna()
    denominator = np.square(valid["y_true"]).sum()
    if denominator <= 0:
        return np.nan
    return float(1.0 - np.square(valid["y_true"] - valid["y_pred"]).sum() / denominator)


def robust_oos_r2(data: pd.DataFrame, lower: float = 0.005, upper: float = 0.995) -> float:
    """GKX R2 after fixed within-month clipping of outcomes and predictions."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Winsorization bounds must satisfy 0 <= lower < upper <= 1.")
    pieces = []
    for _, month in data.groupby("eom", sort=True):
        valid = month[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if valid.empty:
            continue
        for column in ("y_true", "y_pred"):
            bounds = valid[column].quantile([lower, upper])
            valid[column] = valid[column].clip(bounds.iloc[0], bounds.iloc[1])
        pieces.append(valid)
    return oos_r2(pd.concat(pieces, ignore_index=True)) if pieces else np.nan


def clustered_mean_tstat(values: pd.Series, clusters: pd.Series) -> float:
    """CR1 cluster-robust t-statistic for an intercept-only mean regression."""
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    n, groups = len(frame), frame["cluster"].nunique()
    if n < 2 or groups < 2:
        return np.nan
    demeaned = frame["value"] - frame["value"].mean()
    cluster_scores = demeaned.groupby(frame["cluster"]).sum()
    variance = (groups / (groups - 1.0)) * float(np.square(cluster_scores).sum()) / (n * n)
    standard_error = math.sqrt(max(variance, 0.0))
    return float(frame["value"].mean() / standard_error) if standard_error > 0 else np.nan


def safe_autocorrelation(values: pd.Series, lag: int) -> float:
    paired = pd.concat([values, values.shift(lag)], axis=1).dropna()
    if len(paired) < 2 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def monthly_rank_ic(
    data: pd.DataFrame, newey_west_lags: int = 6, minimum_stocks: int = 20
) -> tuple[pd.DataFrame, dict]:
    """Calculate monthly cross-sectional Spearman IC and summary statistics."""
    rows = []
    for eom, month in data.groupby("eom", sort=True):
        valid = (
            month[["y_true", "y_pred"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(valid) < minimum_stocks:
            continue
        # Ranking followed by Pearson correlation is Spearman correlation and
        # avoids an additional SciPy dependency. Average ranks handle ties.
        prediction_rank = valid["y_pred"].rank(method="average")
        return_rank = valid["y_true"].rank(method="average")
        ic = (
            prediction_rank.corr(return_rank, method="pearson")
            if prediction_rank.nunique() > 1 and return_rank.nunique() > 1
            else np.nan
        )
        test_year = int(month["test_year"].iloc[0]) if "test_year" in month else int(pd.Timestamp(eom).year)
        rows.append({"eom": eom, "test_year": test_year, "rank_ic": float(ic), "n_realized": int(len(valid))})

    monthly = pd.DataFrame(rows, columns=["eom", "test_year", "rank_ic", "n_realized"])
    valid_months = monthly.replace({"rank_ic": [np.inf, -np.inf]}).dropna(subset=["rank_ic"])
    ic = valid_months["rank_ic"]
    empty = {
        "mean_monthly_rank_ic": np.nan, "rank_ic_std": np.nan,
        "stock_count_weighted_mean_rank_ic": np.nan,
        "rank_ic_information_ratio": np.nan, "rank_ic_t_stat": np.nan,
        "rank_ic_newey_west_t_stat": np.nan, "rank_ic_clustered_t_stat": np.nan,
        "rank_ic_nw0_t_stat": np.nan, "rank_ic_nw3_t_stat": np.nan,
        "rank_ic_nw6_t_stat": np.nan, "rank_ic_nw12_t_stat": np.nan,
        "rank_ic_positive_rate": np.nan, "rank_ic_n_months": 0,
        "rank_ic_min_stocks": np.nan, "rank_ic_median_stocks": np.nan,
        "rank_ic_max_stocks": np.nan, "rank_ic_autocorr_lag1": np.nan,
        "rank_ic_autocorr_lag2": np.nan, "rank_ic_autocorr_lag3": np.nan,
        "rank_ic_autocorr_lag6": np.nan,
        "rank_ic_autocorr_lag12": np.nan,
    }
    if ic.empty:
        return monthly, empty
    standard_deviation = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    conventional_t = (
        float(ic.mean() / (standard_deviation / math.sqrt(len(ic))))
        if np.isfinite(standard_deviation) and standard_deviation > 0
        else np.nan
    )
    return monthly, {
        "mean_monthly_rank_ic": float(ic.mean()),
        "stock_count_weighted_mean_rank_ic": float(
            np.average(ic, weights=valid_months["n_realized"])
        ),
        "rank_ic_std": standard_deviation,
        "rank_ic_information_ratio": (
            float(math.sqrt(12.0) * ic.mean() / standard_deviation)
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else np.nan
        ),
        "rank_ic_t_stat": conventional_t,
        "rank_ic_newey_west_t_stat": newey_west_tstat(ic, newey_west_lags),
        "rank_ic_nw0_t_stat": newey_west_tstat(ic, 0),
        "rank_ic_nw3_t_stat": newey_west_tstat(ic, 3),
        "rank_ic_nw6_t_stat": newey_west_tstat(ic, 6),
        "rank_ic_nw12_t_stat": newey_west_tstat(ic, 12),
        "rank_ic_clustered_t_stat": clustered_mean_tstat(ic, valid_months["test_year"]),
        "rank_ic_positive_rate": float((ic > 0).mean()),
        "rank_ic_n_months": int(len(ic)),
        "rank_ic_min_stocks": int(valid_months["n_realized"].min()),
        "rank_ic_median_stocks": float(valid_months["n_realized"].median()),
        "rank_ic_max_stocks": int(valid_months["n_realized"].max()),
        "rank_ic_autocorr_lag1": safe_autocorrelation(ic, 1),
        "rank_ic_autocorr_lag2": safe_autocorrelation(ic, 2),
        "rank_ic_autocorr_lag3": safe_autocorrelation(ic, 3),
        "rank_ic_autocorr_lag6": safe_autocorrelation(ic, 6),
        "rank_ic_autocorr_lag12": safe_autocorrelation(ic, 12),
    }


def monthly_mechanism_diagnostics(
    data: pd.DataFrame, n_groups: int = 10, minimum_stocks: int = 20
) -> tuple[pd.DataFrame, dict]:
    """Measure middle-80% ranking skill and cross-sectional calibration."""
    rows = []
    for eom, month in data.groupby("eom", sort=True):
        valid = month[["y_true", "y_pred"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(valid) < max(minimum_stocks, n_groups):
            continue
        valid["prediction_percentile"] = valid["y_pred"].rank(
            method="average", pct=True
        )
        trimmed = valid[valid["prediction_percentile"].between(
            1.0 / n_groups, 1.0 - 1.0 / n_groups, inclusive="neither"
        )]
        pred_rank = trimmed["y_pred"].rank(method="average")
        true_rank = trimmed["y_true"].rank(method="average")
        trimmed_ic = (
            pred_rank.corr(true_rank)
            if pred_rank.nunique() > 1 and true_rank.nunique() > 1 else np.nan
        )
        x, y = valid["y_pred"].to_numpy(float), valid["y_true"].to_numpy(float)
        x_variance = float(np.var(x, ddof=0))
        slope = float(np.mean((x - x.mean()) * (y - y.mean())) / x_variance) if x_variance > 0 else np.nan
        intercept = float(y.mean() - slope * x.mean()) if np.isfinite(slope) else np.nan
        rows.append({
            "eom": eom, "test_year": int(pd.Timestamp(eom).year), "n_realized": int(len(valid)),
            "trimmed_rank_ic": float(trimmed_ic), "prediction_mean": float(x.mean()),
            "realized_mean": float(y.mean()), "prediction_std": float(np.std(x, ddof=1)),
            "realized_std": float(np.std(y, ddof=1)), "calibration_intercept": intercept,
            "calibration_slope": slope,
            "monthly_mse": float(np.mean(np.square(y - x))),
        })
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly, {}
    realized_std = float(monthly["realized_std"].mean())
    return monthly, {
        "mean_monthly_trimmed_rank_ic": float(monthly["trimmed_rank_ic"].mean()),
        "mean_prediction": float(monthly["prediction_mean"].mean()),
        "mean_realized_return": float(monthly["realized_mean"].mean()),
        "mean_prediction_std": float(monthly["prediction_std"].mean()),
        "mean_realized_std": realized_std,
        "prediction_realized_dispersion_ratio": (
            float(monthly["prediction_std"].mean() / realized_std) if realized_std > 0 else np.nan
        ),
        "mean_monthly_calibration_intercept": float(monthly["calibration_intercept"].mean()),
        "mean_monthly_calibration_slope": float(monthly["calibration_slope"].mean()),
    }


def form_portfolio_variants(data: pd.DataFrame) -> pd.DataFrame:
    """Form fixed tail and rank portfolios before observing future returns.

    ``missing_return_stress_ret`` is a deliberately adverse sensitivity check:
    within each month, missing long returns receive the realized cross-sectional
    1st percentile and missing short returns receive the 99th percentile. It is
    not an estimated delisting return and is reported separately from baseline
    performance.
    """
    eligible = data.dropna(subset=["eom", "id", "y_pred"]).copy()
    rows = []
    for eom, month in eligible.groupby("eom", sort=True):
        month = month.copy()
        n = len(month)
        if n < 20:
            continue
        if not has_cross_sectional_signal(month):
            for tail_fraction in (0.05, 0.10, 0.20):
                rows.append({
                    "eom": eom, "test_year": int(pd.Timestamp(eom).year),
                    "strategy": f"TAIL_{int(tail_fraction * 100)}PCT",
                    "long_ret": np.nan, "short_ret": np.nan,
                    "long_short_ret": 0.0, "n_eligible": n,
                    "missing_return_stress_ret": 0.0,
                    "n_long": 0, "n_short": 0,
                    "n_long_assigned": 0, "n_short_assigned": 0,
                    "long_coverage": np.nan,
                    "short_coverage": np.nan, "signal_available": False,
                })
            rows.append({
                "eom": eom, "test_year": int(pd.Timestamp(eom).year),
                "strategy": "RANK_WEIGHTED", "long_ret": np.nan,
                "short_ret": np.nan, "long_short_ret": 0.0,
                "missing_return_stress_ret": 0.0,
                "n_eligible": n, "n_long": 0, "n_short": 0,
                "n_long_assigned": 0, "n_short_assigned": 0,
                "long_coverage": np.nan, "short_coverage": np.nan,
                "signal_available": False,
            })
            continue
        month["rank_score"] = tied_rank_score(month["y_pred"])
        realized = month[month["target_available"] & month["y_true"].notna()].copy()
        if realized.empty:
            adverse_long_return = adverse_short_return = np.nan
        else:
            adverse_long_return, adverse_short_return = realized["y_true"].quantile([0.01, 0.99])
        for tail_fraction in (0.05, 0.10, 0.20):
            assigned = fractional_tail_membership(month, tail_fraction)
            leg_size = float(assigned["long_membership"].sum())
            realized_assigned = assigned[
                assigned["target_available"] & assigned["y_true"].notna()
            ]
            long_weight = realized_assigned["long_membership"]
            short_weight = realized_assigned["short_membership"]
            long_return = (
                float((long_weight * realized_assigned["y_true"]).sum() / long_weight.sum())
                if long_weight.sum() > 0 else np.nan
            )
            short_return = (
                float((short_weight * realized_assigned["y_true"]).sum() / short_weight.sum())
                if short_weight.sum() > 0 else np.nan
            )
            stressed_long_y = assigned["y_true"].where(
                assigned["target_available"] & assigned["y_true"].notna(),
                adverse_long_return,
            )
            stressed_short_y = assigned["y_true"].where(
                assigned["target_available"] & assigned["y_true"].notna(),
                adverse_short_return,
            )
            stressed_long = float(
                (assigned["long_membership"] * stressed_long_y).sum() / leg_size
            )
            stressed_short = float(
                (assigned["short_membership"] * stressed_short_y).sum() / leg_size
            )
            long_realized_mass = float(long_weight.sum())
            short_realized_mass = float(short_weight.sum())
            rows.append({
                "eom": eom, "test_year": int(pd.Timestamp(eom).year),
                "strategy": f"TAIL_{int(tail_fraction * 100)}PCT",
                "long_ret": long_return, "short_ret": short_return,
                "long_short_ret": long_return - short_return,
                "missing_return_stress_ret": float(stressed_long - stressed_short),
                "n_eligible": n, "n_long": long_realized_mass,
                "n_short": short_realized_mass,
                "n_long_assigned": leg_size, "n_short_assigned": leg_size,
                "long_coverage": long_realized_mass / leg_size,
                "short_coverage": short_realized_mass / leg_size,
                "signal_available": True,
            })
        rank_realized = realized.copy()
        positive = rank_realized["rank_score"].clip(lower=0.0)
        negative = -rank_realized["rank_score"].clip(upper=0.0)
        long_weights = positive / positive.sum() if positive.sum() > 0 else positive * np.nan
        short_weights = negative / negative.sum() if negative.sum() > 0 else negative * np.nan
        long_return = float((long_weights * rank_realized["y_true"]).sum())
        short_return = float((short_weights * rank_realized["y_true"]).sum())
        all_positive = month["rank_score"].clip(lower=0.0)
        all_negative = -month["rank_score"].clip(upper=0.0)
        stressed_long_y = month["y_true"].where(
            month["target_available"] & month["y_true"].notna(), adverse_long_return
        )
        stressed_short_y = month["y_true"].where(
            month["target_available"] & month["y_true"].notna(), adverse_short_return
        )
        stressed_long = float((all_positive / all_positive.sum() * stressed_long_y).sum())
        stressed_short = float((all_negative / all_negative.sum() * stressed_short_y).sum())
        rows.append({
            "eom": eom, "test_year": int(pd.Timestamp(eom).year),
            "strategy": "RANK_WEIGHTED", "long_ret": long_return,
            "short_ret": short_return, "long_short_ret": long_return - short_return,
            "missing_return_stress_ret": stressed_long - stressed_short,
            "n_eligible": n, "n_long": int((positive > 0).sum()),
            "n_short": int((negative > 0).sum()),
            "n_long_assigned": int((all_positive > 0).sum()),
            "n_short_assigned": int((all_negative > 0).sum()),
            "long_coverage": float((positive > 0).sum() / max(1, (month["rank_score"] > 0).sum())),
            "short_coverage": float((negative > 0).sum() / max(1, (month["rank_score"] < 0).sum())),
            "signal_available": True,
        })
    return pd.DataFrame(rows)


def portfolio_variant_stats(variants: pd.DataFrame, newey_west_lags: int = 6) -> dict:
    output = {}
    if variants.empty or "strategy" not in variants:
        return output
    for strategy, group in variants.groupby("strategy", sort=True):
        prefix = strategy.lower()
        stats = performance_stats(group["long_short_ret"], newey_west_lags)
        for key, value in stats.items():
            output[f"{prefix}_{key}"] = value
        stress_stats = performance_stats(
            group["missing_return_stress_ret"], newey_west_lags
        )
        for key, value in stress_stats.items():
            output[f"{prefix}_missing_return_stress_{key}"] = value
        output[f"{prefix}_mean_long_return"] = float(group["long_ret"].mean())
        output[f"{prefix}_mean_short_return"] = float(group["short_ret"].mean())
        output[f"{prefix}_mean_n_long"] = float(group["n_long"].mean())
        output[f"{prefix}_mean_n_short"] = float(group["n_short"].mean())
        output[f"{prefix}_mean_n_long_assigned"] = float(group["n_long_assigned"].mean())
        output[f"{prefix}_mean_n_short_assigned"] = float(group["n_short_assigned"].mean())
        output[f"{prefix}_mean_long_coverage"] = float(group["long_coverage"].mean())
        output[f"{prefix}_mean_short_coverage"] = float(group["short_coverage"].mean())
    return output


def decile_summary(deciles: pd.DataFrame, n_groups: int = 10) -> dict:
    active = (
        deciles.loc[deciles["signal_available"]].copy()
        if "signal_available" in deciles else deciles
    )
    means = pd.Series({
        f"D{i}": active[f"D{i}"].mean() if f"D{i}" in active else np.nan
        for i in range(1, n_groups + 1)
    })
    valid = means.dropna()
    output = {f"mean_{key}_return": float(value) for key, value in means.items()}
    if len(valid) > 1 and valid.nunique() > 1:
        decile_numbers = pd.Series(
            [int(str(label).removeprefix("D")) for label in valid.index],
            index=valid.index, dtype=float,
        )
        output["decile_monotonicity_spearman"] = float(
            decile_numbers.rank().corr(valid.rank(method="average"))
        )
        output["decile_adjacent_increase_rate"] = float((valid.diff().dropna() > 0).mean())
    else:
        output["decile_monotonicity_spearman"] = np.nan
        output["decile_adjacent_increase_rate"] = np.nan
    if not deciles.empty:
        output.update({
            "mean_monthly_eligible_stocks": float(deciles["n_eligible"].mean()),
            "mean_monthly_realized_stocks": float(deciles["n_realized"].mean()),
            "mean_monthly_return_coverage": float(deciles["return_coverage"].mean()),
            "minimum_assigned_decile_stock_count": (
                int(active[[f"N{i}" for i in range(1, n_groups + 1)]].min().min())
                if not active.empty else np.nan
            ),
            "minimum_realized_decile_stock_count": (
                int(active[[f"R{i}" for i in range(1, n_groups + 1)]].min().min())
                if not active.empty else np.nan
            ),
        })
    return output


def form_equal_weight_deciles(data: pd.DataFrame, n_groups: int = 10) -> pd.DataFrame:
    """Assign portfolios before checking whether next-month returns are available."""
    eligible = data.dropna(subset=["eom", "id", "y_pred"]).copy()
    rows = []
    for eom, month in eligible.groupby("eom", sort=True):
        if len(month) < n_groups:
            continue
        if not has_cross_sectional_signal(month):
            realized = month.loc[month["target_available"] & month["y_true"].notna()]
            row = {
                "eom": eom, "n_eligible": int(len(month)),
                "n_realized": int(len(realized)),
                "return_coverage": float(len(realized) / len(month)),
                "signal_available": False,
            }
            for portfolio in range(1, n_groups + 1):
                row[f"D{portfolio}"] = np.nan
                row[f"N{portfolio}"] = 0
                row[f"R{portfolio}"] = 0
            row.update({"long_ret": np.nan, "short_ret": np.nan, "long_short_ret": 0.0})
            rows.append(row)
            continue
        month = fractional_quantile_membership(month, n_groups)
        realized = month.loc[month["target_available"] & month["y_true"].notna()].copy()
        row = {
            "eom": eom,
            "n_eligible": int(len(month)),
            "n_realized": int(len(realized)),
            "return_coverage": float(len(realized) / len(month)),
            "signal_available": True,
        }
        for portfolio in range(1, n_groups + 1):
            membership = f"membership_{portfolio}"
            assigned_mass = float(month[membership].sum())
            realized_mass = float(realized[membership].sum())
            row[f"D{portfolio}"] = (
                float((realized[membership] * realized["y_true"]).sum() / realized_mass)
                if realized_mass > 0 else np.nan
            )
            row[f"N{portfolio}"] = assigned_mass
            row[f"R{portfolio}"] = realized_mass
        row["long_ret"] = row[f"D{n_groups}"]
        row["short_ret"] = row["D1"]
        row["long_short_ret"] = row["long_ret"] - row["short_ret"]
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values("eom").reset_index(drop=True) if not result.empty else result


def newey_west_tstat(values: pd.Series, lags: int = 6) -> float:
    x = values.dropna().to_numpy(float)
    n = len(x)
    if n < 2:
        return np.nan
    demeaned = x - x.mean()
    long_run_variance = np.dot(demeaned, demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = np.dot(demeaned[lag:], demeaned[:-lag]) / n
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    return float(x.mean() / standard_error) if standard_error > 0 else np.nan


def performance_stats(returns: pd.Series, newey_west_lags: int = 6) -> dict:
    r = returns.dropna().astype(float)
    if r.empty:
        return {}
    mean_monthly = float(r.mean())
    monthly_volatility = float(r.std(ddof=1))
    annualized_return = 12.0 * mean_monthly
    annualized_volatility = math.sqrt(12.0) * monthly_volatility
    cumulative = (1.0 + r).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    conventional_t = mean_monthly / (monthly_volatility / math.sqrt(len(r))) if monthly_volatility > 0 else np.nan
    return {
        "mean_monthly_return": mean_monthly,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan,
        "t_stat": float(conventional_t),
        "newey_west_t_stat": newey_west_tstat(r, newey_west_lags),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((r > 0).mean()),
        "n_months": int(len(r)),
    }
