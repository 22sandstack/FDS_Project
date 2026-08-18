from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CORE20,
    CORE20_LAG1,
    CORE20_LAG2,
    CORE20_VELOCITY,
    FEATURES_40,
    FEATURES_40_LAG1,
    FEATURES_40_LAG1_AVAILABLE,
    FEATURES_40_LAG2,
    FEATURES_40_LAG2_AVAILABLE,
    FEATURES_40_VELOCITY,
    LAG1_AVAILABLE,
    LAG2_AVAILABLE,
    ExperimentConfig,
)


IDENTIFIER_COLUMNS = ["id", "eom", "excntry", "size_grp", "me"]


def _add_exact_calendar_lag1(
    df: pd.DataFrame,
    features: tuple[str, ...],
    lagged_names: tuple[str, ...],
    velocity_names: tuple[str, ...],
    availability_name: str,
    missing_fill: float,
    security_id_col: str = "id",
) -> pd.DataFrame:
    """Add exact one-month lags, velocities, and an availability indicator."""
    df = df.sort_values([security_id_col, "eom"]).reset_index(drop=True)
    previous_eom = df.groupby(security_id_col, sort=False)["eom"].shift(1)
    exact_previous_month = previous_eom.eq(df["eom"] - pd.offsets.MonthEnd(1))
    previous = df.groupby(security_id_col, sort=False)[list(features)].shift(1)
    additions = {availability_name: exact_previous_month.astype("float32")}
    for current, lagged, velocity in zip(features, lagged_names, velocity_names):
        previous_value = previous[current].where(exact_previous_month)
        lagged_value = previous_value.fillna(missing_fill).astype("float32")
        additions[lagged] = lagged_value
        additions[velocity] = (
            df[current] - lagged_value
        ).where(exact_previous_month, missing_fill).astype("float32")
    return pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)


def add_core20_dynamics(
    df: pd.DataFrame, missing_fill: float, security_id_col: str = "id"
) -> pd.DataFrame:
    """Add exact-calendar Core20 lag ranks and velocities."""
    return _add_exact_calendar_lag1(
        df, CORE20, CORE20_LAG1, CORE20_VELOCITY, LAG1_AVAILABLE,
        missing_fill, security_id_col
    )


def add_feature40_lag1(
    df: pd.DataFrame, missing_fill: float, security_id_col: str = "id"
) -> pd.DataFrame:
    """Add exact-calendar lags for the frozen 40-characteristic set."""
    return _add_exact_calendar_lag1(
        df,
        FEATURES_40,
        FEATURES_40_LAG1,
        FEATURES_40_VELOCITY,
        FEATURES_40_LAG1_AVAILABLE,
        missing_fill,
        security_id_col,
    )


def add_exact_calendar_lag2(
    df: pd.DataFrame,
    features: tuple[str, ...],
    lagged_names: tuple[str, ...],
    availability_name: str,
    missing_fill: float,
    security_id_col: str = "id",
) -> pd.DataFrame:
    """Join values from exactly two calendar months earlier by security identifier."""
    source = df[[security_id_col, "eom", *features]].copy()
    source["eom"] = source["eom"] + pd.offsets.MonthEnd(2)
    source = source.rename(columns=dict(zip(features, lagged_names)))
    source[availability_name] = np.float32(1.0)
    result = df.merge(
        source, on=[security_id_col, "eom"], how="left", validate="one_to_one"
    )
    result[availability_name] = result[availability_name].fillna(0.0).astype("float32")
    result[list(lagged_names)] = result[list(lagged_names)].fillna(missing_fill).astype("float32")
    return result.sort_values([security_id_col, "eom"]).reset_index(drop=True)


def load_and_prepare_panel(
    config: ExperimentConfig,
    rank_features: tuple[str, ...] = CORE20,
    *,
    include_core_dynamics: bool = True,
    include_feature40_lag1: bool | None = None,
    include_core_lag2: bool = False,
    include_feature40_lag2: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load, validate, filter, and rank the single configured research universe."""
    rank_features = tuple(dict.fromkeys(rank_features))
    if include_feature40_lag1 is None:
        include_feature40_lag1 = set(FEATURES_40).issubset(rank_features)
    if include_feature40_lag1 and not set(FEATURES_40).issubset(rank_features):
        raise ValueError("Feature40 derived blocks require all FEATURES_40 inputs.")
    if include_feature40_lag2 and not set(FEATURES_40).issubset(rank_features):
        raise ValueError("Feature40 lag-2 blocks require all FEATURES_40 inputs.")
    if not set(CORE20).issubset(rank_features):
        raise ValueError("rank_features must include Core20 for dynamic model support.")
    security_id_col = config.universe.security_id_col
    required = list(dict.fromkeys(
        IDENTIFIER_COLUMNS + [security_id_col, config.target_col] + list(rank_features)
    ))
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["eom"] = pd.to_datetime(df["eom"], errors="coerce")
    if df["eom"].isna().any():
        raise ValueError("eom contains values that cannot be parsed as dates.")

    years = df["eom"].dt.year
    size_group = df["size_grp"].astype("string").str.strip().str.lower()
    eligible = (
        years.between(config.universe.start_year, config.universe.end_year)
        & df["excntry"].eq(config.universe.country)
        & df[security_id_col].notna()
        & size_group.isin(config.universe.allowed_size_groups)
    )
    df = df.loc[eligible].copy()
    df["size_grp"] = size_group.loc[eligible]
    df["security_id"] = df[security_id_col].astype("string")

    if df["id"].isna().any():
        raise ValueError("Eligible observations contain missing id values.")

    for col in list(rank_features) + [config.target_col, "me"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_security = df.duplicated(["eom", "security_id"], keep=False)
    if duplicate_security.any():
        sample = df.loc[
            duplicate_security, ["eom", "id", "security_id"]
        ].head(20)
        raise ValueError(f"Duplicate (eom, security_id) observations found:\n{sample}")

    duplicate_id = df.duplicated(["eom", "id"], keep=False)
    if duplicate_id.any():
        sample = df.loc[duplicate_id, ["eom", "id", "security_id"]].head(20)
        raise ValueError(f"Duplicate (eom, id) observations found:\n{sample}")

    # Ranking is performed on the eligible month-t universe before inspecting
    # whether the next-month target is available.
    for col in rank_features:
        grouped = df.groupby("eom", sort=False)[col]
        rank = grouped.rank(method="average", na_option="keep")
        count = grouped.transform("count")
        denominator = (count - 1).replace(0, np.nan)
        normalized = 2.0 * (rank - 1.0) / denominator - 1.0
        df[col] = normalized.fillna(config.preprocessing.missing_feature_fill).astype("float32")

    # Lags refer to the exact preceding calendar month. A stock returning after
    # a gap must not have its last observed row treated as a one-month lag.
    if include_core_dynamics:
        df = add_core20_dynamics(
            df, config.preprocessing.missing_feature_fill, "security_id"
        )
    if include_feature40_lag1:
        df = add_feature40_lag1(
            df, config.preprocessing.missing_feature_fill, "security_id"
        )
    if include_core_lag2:
        df = add_exact_calendar_lag2(
            df, CORE20, CORE20_LAG2, LAG2_AVAILABLE,
            config.preprocessing.missing_feature_fill,
            "security_id",
        )
    if include_feature40_lag2:
        df = add_exact_calendar_lag2(
            df, FEATURES_40, FEATURES_40_LAG2, FEATURES_40_LAG2_AVAILABLE,
            config.preprocessing.missing_feature_fill,
            "security_id",
        )

    df["target_date"] = df["eom"] + pd.offsets.MonthEnd(1)
    # Always store a non-null, plain Boolean. Pandas nullable Float64 can make
    # np.isfinite return <NA> for missing targets rather than False.
    df["target_available"] = (
        df[config.target_col]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .astype(bool)
    )
    df = df.sort_values(["eom", "security_id"]).reset_index(drop=True)

    audit = {
        "rows": int(len(df)),
        "unique_stocks": int(df["security_id"].nunique()),
        "months": int(df["eom"].nunique()),
        "start": str(df["eom"].min().date()),
        "end": str(df["eom"].max().date()),
        "target_available": int(df["target_available"].sum()),
        "target_missing": int((~df["target_available"]).sum()),
        "size_groups": {str(k): int(v) for k, v in df["size_grp"].value_counts().items()},
    }
    return df, audit


def build_oos_universe(panel: pd.DataFrame, schedule: pd.DataFrame, target_col: str) -> pd.DataFrame:
    year_map = schedule.set_index("test_year")["refit_id"].to_dict()
    out = panel.loc[panel["eom"].dt.year.isin(year_map)].copy()
    out["test_year"] = out["eom"].dt.year.astype(int)
    out["refit_id"] = out["test_year"].map(year_map).astype(int)
    columns = [
        "eom", "target_date", "id", "security_id", "excntry", "me", "size_grp",
        target_col, "target_available", "test_year", "refit_id",
    ]
    out = out[columns].rename(columns={"excntry": "country", target_col: "y_true"})
    return out.sort_values(["eom", "security_id"]).reset_index(drop=True)


def target_availability_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing next-month returns without changing the research sample."""
    rows = []

    def append(dimension: str, group: str, frame: pd.DataFrame) -> None:
        n = len(frame)
        available = int(frame["target_available"].sum())
        rows.append({
            "dimension": dimension,
            "group": str(group),
            "n_observations": n,
            "n_target_available": available,
            "n_target_missing": n - available,
            "target_missing_rate": float((n - available) / n) if n else np.nan,
        })

    append("overall", "all", panel)
    for size_group, group in panel.groupby("size_grp", observed=True, sort=True):
        append("size_group", size_group, group)
    for year, group in panel.groupby(panel["eom"].dt.year, sort=True):
        append("signal_year", int(year), group)
    return pd.DataFrame(rows)
