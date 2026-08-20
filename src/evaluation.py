from __future__ import annotations

import math

import numpy as np
import pandas as pd


MINIMUM_UNIQUE_PREDICTIONS = 2


def has_cross_sectional_signal(
    month: pd.DataFrame,
    minimum_unique: int = MINIMUM_UNIQUE_PREDICTIONS,
) -> bool:
    """Return whether a month contains a usable cross-sectional forecast."""
    prediction = pd.to_numeric(
        month["y_pred"],
        errors="coerce",
    )
    prediction = prediction[
        np.isfinite(prediction)
    ]
    return (
        prediction.nunique()
        >= minimum_unique
    )


def fractional_quantile_membership(
    month: pd.DataFrame,
    n_groups: int,
) -> pd.DataFrame:
    """
    Allocate tied forecast groups fractionally across equal-sized quantiles.

    This avoids arbitrary identifier-based tie breaking.
    """
    if n_groups < 2:
        raise ValueError(
            "n_groups must be at least 2."
        )

    result = month.copy()
    membership_columns = [
        f"membership_{group}"
        for group in range(
            1,
            n_groups + 1,
        )
    ]

    result[membership_columns] = 0.0

    n = len(result)
    if n == 0:
        return result

    cursor = 0.0

    for _, index in result.groupby(
        "y_pred",
        sort=True,
    ).groups.items():
        tie_size = float(len(index))
        tie_start = cursor
        tie_end = cursor + tie_size

        for group in range(
            1,
            n_groups + 1,
        ):
            bucket_start = (
                n * (group - 1)
                / n_groups
            )
            bucket_end = (
                n * group
                / n_groups
            )

            overlap = max(
                0.0,
                min(tie_end, bucket_end)
                - max(
                    tie_start,
                    bucket_start,
                ),
            )

            if overlap > 0:
                result.loc[
                    index,
                    f"membership_{group}",
                ] = overlap / tie_size

        cursor = tie_end

    return result


def merge_predictions(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Merge forecasts onto the frozen common OOS observation universe."""
    keys = [
        "eom",
        "id",
        "security_id",
        "test_year",
        "refit_id",
    ]

    if predictions.duplicated(keys).any():
        raise ValueError(
            "Predictions contain duplicate observation keys."
        )

    if universe.duplicated(keys).any():
        raise ValueError(
            "OOS universe contains duplicate observation keys."
        )

    overlap = [
        column
        for column in universe.columns
        if (
            column in predictions.columns
            and column not in keys
        )
    ]

    forecast = predictions.drop(
        columns=overlap
    )

    merged = forecast.merge(
        universe,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged["_merge"].ne(
        "both"
    )

    if unmatched.any():
        counts = (
            merged.loc[
                unmatched,
                "_merge",
            ]
            .value_counts()
            .to_dict()
        )

        raise ValueError(
            "Prediction/universe key coverage "
            f"mismatch: {counts}"
        )

    return merged.drop(
        columns="_merge"
    )


def oos_r2(
    data: pd.DataFrame,
) -> float:
    """
    Pooled GKX-style out-of-sample R² relative to a zero-return forecast.
    """
    valid = (
        data[["y_true", "y_pred"]]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if valid.empty:
        return np.nan

    denominator = float(
        np.square(
            valid["y_true"]
        ).sum()
    )

    if denominator <= 0:
        return np.nan

    numerator = float(
        np.square(
            valid["y_true"]
            - valid["y_pred"]
        ).sum()
    )

    return float(
        1.0 - numerator / denominator
    )


def newey_west_tstat(
    values: pd.Series,
    lags: int = 6,
) -> float:
    """
    Newey-West t-statistic for the mean of a monthly series.

    Uses Bartlett weights and a fixed lag count.
    """
    values = (
        pd.Series(values)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(float)
        .to_numpy()
    )

    n = len(values)

    if n < 2:
        return np.nan

    if lags < 0:
        raise ValueError(
            "lags cannot be negative."
        )

    mean = float(values.mean())
    demeaned = values - mean

    gamma0 = float(
        np.dot(
            demeaned,
            demeaned,
        )
        / n
    )

    long_run_variance = gamma0
    max_lag = min(
        int(lags),
        n - 1,
    )

    for lag in range(
        1,
        max_lag + 1,
    ):
        covariance = float(
            np.dot(
                demeaned[lag:],
                demeaned[:-lag],
            )
            / n
        )

        weight = (
            1.0
            - lag
            / (max_lag + 1.0)
        )

        long_run_variance += (
            2.0
            * weight
            * covariance
        )

    variance_of_mean = (
        long_run_variance / n
    )

    if (
        not np.isfinite(
            variance_of_mean
        )
        or variance_of_mean <= 0
    ):
        return np.nan

    return float(
        mean
        / math.sqrt(
            variance_of_mean
        )
    )


def monthly_rank_ic(
    data: pd.DataFrame,
    newey_west_lags: int = 6,
    minimum_stocks: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate monthly Spearman Rank IC and the compact common summary.

    Returned summary:
    - mean monthly Rank IC
    - Newey-West t-statistic
    - number of valid IC months
    """
    rows: list[dict] = []

    for eom, month in data.groupby(
        "eom",
        sort=True,
    ):
        valid = (
            month[
                [
                    "y_true",
                    "y_pred",
                ]
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if len(valid) < minimum_stocks:
            continue

        prediction_rank = valid[
            "y_pred"
        ].rank(
            method="average"
        )

        return_rank = valid[
            "y_true"
        ].rank(
            method="average"
        )

        if (
            prediction_rank.nunique() < 2
            or return_rank.nunique() < 2
        ):
            rank_ic = np.nan
        else:
            rank_ic = float(
                prediction_rank.corr(
                    return_rank
                )
            )

        rows.append(
            {
                "eom": eom,
                "test_year": int(
                    month["test_year"].iloc[0]
                )
                if "test_year"
                in month.columns
                else int(
                    pd.Timestamp(
                        eom
                    ).year
                ),
                "rank_ic": rank_ic,
                "n_stocks": int(
                    len(valid)
                ),
            }
        )

    monthly = pd.DataFrame(
        rows,
        columns=[
            "eom",
            "test_year",
            "rank_ic",
            "n_stocks",
        ],
    )

    valid_ic = (
        monthly["rank_ic"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if valid_ic.empty:
        summary = {
            "mean_monthly_rank_ic": np.nan,
            "rank_ic_newey_west_t_stat": np.nan,
            "rank_ic_n_months": 0,
        }
    else:
        summary = {
            "mean_monthly_rank_ic": float(
                valid_ic.mean()
            ),
            "rank_ic_newey_west_t_stat": (
                newey_west_tstat(
                    valid_ic,
                    newey_west_lags,
                )
            ),
            "rank_ic_n_months": int(
                len(valid_ic)
            ),
        }

    return monthly, summary


def form_equal_weight_deciles(
    data: pd.DataFrame,
    n_groups: int = 10,
) -> pd.DataFrame:
    """
    Form one equal-weight long-short decile portfolio each month.

    Portfolio formation uses the full finite forecast cross-section. Ties that
    cross quantile boundaries receive fractional membership. Realized returns
    are averaged over stocks with available outcomes within each side.

    If a month has no usable cross-sectional ranking signal, the portfolio
    return is set to zero rather than imposing an arbitrary stock ordering.
    """
    if n_groups < 2:
        raise ValueError(
            "n_groups must be at least 2."
        )

    rows: list[dict] = []

    for eom, month in data.groupby(
        "eom",
        sort=True,
    ):
        forecast = month.loc[
            np.isfinite(
                pd.to_numeric(
                    month["y_pred"],
                    errors="coerce",
                )
            )
        ].copy()

        test_year = (
            int(
                month["test_year"].iloc[0]
            )
            if "test_year"
            in month.columns
            else int(
                pd.Timestamp(
                    eom
                ).year
            )
        )

        if (
            forecast.empty
            or not has_cross_sectional_signal(
                forecast
            )
        ):
            rows.append(
                {
                    "eom": eom,
                    "test_year": test_year,
                    "long_short_ret": 0.0,
                }
            )
            continue

        assigned = (
            fractional_quantile_membership(
                forecast,
                n_groups,
            )
        )

        realized = pd.to_numeric(
            assigned["y_true"],
            errors="coerce",
        )

        finite_return = np.isfinite(
            realized
        )

        bottom_weight = (
            assigned[
                "membership_1"
            ]
            .where(
                finite_return,
                0.0,
            )
            .astype(float)
        )

        top_weight = (
            assigned[
                f"membership_{n_groups}"
            ]
            .where(
                finite_return,
                0.0,
            )
            .astype(float)
        )

        bottom_mass = float(
            bottom_weight.sum()
        )
        top_mass = float(
            top_weight.sum()
        )

        bottom_return = (
            float(
                np.dot(
                    bottom_weight,
                    realized.fillna(0.0),
                )
                / bottom_mass
            )
            if bottom_mass > 0
            else np.nan
        )

        top_return = (
            float(
                np.dot(
                    top_weight,
                    realized.fillna(0.0),
                )
                / top_mass
            )
            if top_mass > 0
            else np.nan
        )

        long_short_return = (
            top_return
            - bottom_return
            if (
                np.isfinite(
                    top_return
                )
                and np.isfinite(
                    bottom_return
                )
            )
            else np.nan
        )

        rows.append(
            {
                "eom": eom,
                "test_year": test_year,
                "long_short_ret": (
                    long_short_return
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "eom",
            "test_year",
            "long_short_ret",
        ],
    )


def performance_stats(
    monthly_returns: pd.Series,
    newey_west_lags: int = 6,
) -> dict:
    """
    Summarize the common equal-weight long-short monthly return series.

    Returned summary:
    - annualized return
    - annualized volatility
    - Sharpe ratio
    - Newey-West t-statistic
    - maximum drawdown
    """
    returns = (
        pd.Series(
            monthly_returns
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(float)
    )

    if returns.empty:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "portfolio_newey_west_t_stat": np.nan,
            "max_drawdown": np.nan,
        }

    mean_monthly = float(
        returns.mean()
    )

    monthly_volatility = (
        float(
            returns.std(ddof=1)
        )
        if len(returns) > 1
        else np.nan
    )

    annualized_return = (
        12.0 * mean_monthly
    )

    annualized_volatility = (
        math.sqrt(12.0)
        * monthly_volatility
        if np.isfinite(
            monthly_volatility
        )
        else np.nan
    )

    sharpe_ratio = (
        annualized_return
        / annualized_volatility
        if (
            np.isfinite(
                annualized_volatility
            )
            and annualized_volatility
            > 0
        )
        else np.nan
    )

    wealth = (
        1.0 + returns
    ).cumprod()

    running_peak = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_peak
        - 1.0
    )

    return {
        "annualized_return": float(
            annualized_return
        ),
        "annualized_volatility": float(
            annualized_volatility
        )
        if np.isfinite(
            annualized_volatility
        )
        else np.nan,
        "sharpe_ratio": float(
            sharpe_ratio
        )
        if np.isfinite(
            sharpe_ratio
        )
        else np.nan,
        "portfolio_newey_west_t_stat": (
            newey_west_tstat(
                returns,
                newey_west_lags,
            )
        ),
        "max_drawdown": float(
            drawdown.min()
        ),
    }
