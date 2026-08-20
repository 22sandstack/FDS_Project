from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CORE20,
    FEATURES_40,
    FEATURES_40_LAG1,
    FEATURES_40_LAG1_AVAILABLE,
    FEATURES_40_LAG2,
    FEATURES_40_LAG2_AVAILABLE,
    FEATURES_40_VELOCITY,
    ExperimentConfig,
)


BASE_COLUMNS: tuple[str, ...] = (
    "id",
    "eom",
    "excntry",
    "size_grp",
    "me",
)


# ----------------------------------------------------------------------
# Cross-sectional preprocessing
# ----------------------------------------------------------------------

def _monthly_rank(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    missing_fill: float,
) -> pd.DataFrame:
    """
    Rank characteristics within each eligible month and map ranks to [-1, 1].

    Missing or non-finite characteristics are assigned the neutral value
    `missing_fill` after ranking.

    Importantly, this function uses the complete eligible month-t cross section.
    Target availability is not used to determine the ranking universe.
    """
    for column in columns:
        grouped = df.groupby("eom", sort=False)[column]

        ranks = grouped.rank(
            method="average",
            na_option="keep",
        )
        counts = grouped.transform("count")

        denominator = (
            counts - 1
        ).replace(0, np.nan)

        ranked = (
            2.0 * (ranks - 1.0)
            / denominator
            - 1.0
        )

        df[column] = (
            ranked
            .fillna(missing_fill)
            .astype("float32")
        )

    return df


# ----------------------------------------------------------------------
# Exact-calendar lagged and dynamic characteristics
# ----------------------------------------------------------------------

def add_feature40_lag1(
    df: pd.DataFrame,
    missing_fill: float,
    security_id_col: str = "security_id",
) -> pd.DataFrame:
    """
    Add exact one-month lags and one-month characteristic changes.

    If a stock was not observed in the immediately preceding calendar month,
    all lagged/dynamic features are set to the neutral fill value and the
    availability flag is zero.
    """
    df = (
        df.sort_values(
            [security_id_col, "eom"]
        )
        .reset_index(drop=True)
    )

    grouped = df.groupby(
        security_id_col,
        sort=False,
    )

    previous_eom = grouped["eom"].shift(1)

    exact_previous_month = previous_eom.eq(
        df["eom"] - pd.offsets.MonthEnd(1)
    )

    previous_values = grouped[
        list(FEATURES_40)
    ].shift(1)

    additions: dict[str, pd.Series] = {
        FEATURES_40_LAG1_AVAILABLE: (
            exact_previous_month
            .astype("float32")
        )
    }

    for feature, lag_name, velocity_name in zip(
        FEATURES_40,
        FEATURES_40_LAG1,
        FEATURES_40_VELOCITY,
    ):
        lagged = previous_values[feature].where(
            exact_previous_month
        )

        lagged_filled = (
            lagged
            .fillna(missing_fill)
            .astype("float32")
        )

        velocity = (
            df[feature] - lagged
        ).where(
            exact_previous_month,
            missing_fill,
        )

        additions[lag_name] = lagged_filled
        additions[velocity_name] = (
            velocity
            .fillna(missing_fill)
            .astype("float32")
        )

    return pd.concat(
        [
            df,
            pd.DataFrame(
                additions,
                index=df.index,
            ),
        ],
        axis=1,
    )


def add_feature40_lag2(
    df: pd.DataFrame,
    missing_fill: float,
    security_id_col: str = "security_id",
) -> pd.DataFrame:
    """
    Add characteristics from exactly two calendar months earlier.

    This uses an explicit date-keyed merge rather than row shifting, so a gap
    in a stock's observation history cannot be mistaken for a two-month lag.
    """
    source = df[
        [
            security_id_col,
            "eom",
            *FEATURES_40,
        ]
    ].copy()

    source["eom"] = (
        source["eom"]
        + pd.offsets.MonthEnd(2)
    )

    source = source.rename(
        columns=dict(
            zip(
                FEATURES_40,
                FEATURES_40_LAG2,
            )
        )
    )

    source[
        FEATURES_40_LAG2_AVAILABLE
    ] = np.float32(1.0)

    result = df.merge(
        source,
        on=[security_id_col, "eom"],
        how="left",
        validate="one_to_one",
    )

    result[
        FEATURES_40_LAG2_AVAILABLE
    ] = (
        result[
            FEATURES_40_LAG2_AVAILABLE
        ]
        .fillna(0.0)
        .astype("float32")
    )

    result[
        list(FEATURES_40_LAG2)
    ] = (
        result[
            list(FEATURES_40_LAG2)
        ]
        .fillna(missing_fill)
        .astype("float32")
    )

    return (
        result.sort_values(
            [security_id_col, "eom"]
        )
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# Panel construction
# ----------------------------------------------------------------------

def load_and_prepare_panel(
    config: ExperimentConfig,
    rank_features: tuple[str, ...] = CORE20,
    *,
    include_feature40_lag1: bool = False,
    include_feature40_lag2: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Load and prepare the configured research panel.

    Steps
    -----
    1. Read only required parquet columns.
    2. Filter to the frozen universe.
    3. Validate observation identifiers.
    4. Convert characteristics to numeric and remove infinities.
    5. Rank characteristics within the complete eligible month.
    6. Add requested exact-calendar lag/dynamic features.
    7. Record target timing and availability.
    """
    rank_features = tuple(
        dict.fromkeys(rank_features)
    )

    if not rank_features:
        raise ValueError(
            "rank_features cannot be empty."
        )

    if include_feature40_lag1 and not set(
        FEATURES_40
    ).issubset(rank_features):
        raise ValueError(
            "Feature40 lag/dynamic features require "
            "all FEATURES_40 inputs."
        )

    if include_feature40_lag2 and not set(
        FEATURES_40
    ).issubset(rank_features):
        raise ValueError(
            "Feature40 lag-2 features require "
            "all FEATURES_40 inputs."
        )

    security_id_col = (
        config.universe.security_id_col
    )

    required_columns = list(
        dict.fromkeys(
            [
                *BASE_COLUMNS,
                security_id_col,
                config.target_col,
                *rank_features,
            ]
        )
    )

    path = Path(config.data_path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(
        path,
        columns=required_columns,
    )

    # ------------------------------------------------------------------
    # Dates and universe filter
    # ------------------------------------------------------------------

    df["eom"] = pd.to_datetime(
        df["eom"],
        errors="coerce",
    )

    if df["eom"].isna().any():
        raise ValueError(
            "eom contains values that cannot "
            "be parsed as dates."
        )

    size_group = (
        df["size_grp"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    eligible = (
        df["eom"].dt.year.between(
            config.universe.start_year,
            config.universe.end_year,
        )
        & df["excntry"].eq(
            config.universe.country
        )
        & df[security_id_col].notna()
        & size_group.isin(
            config.universe
            .allowed_size_groups
        )
    )

    df = df.loc[eligible].copy()

    if df.empty:
        raise ValueError(
            "No observations remain after "
            "applying the universe filter."
        )

    df["size_grp"] = size_group.loc[
        eligible
    ]

    df["security_id"] = (
        df[security_id_col]
        .astype("string")
    )

    # ------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------

    numeric_columns = [
        *rank_features,
        config.target_col,
        "me",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[list(rank_features)] = (
        df[list(rank_features)]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    df[config.target_col] = (
        df[config.target_col]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ------------------------------------------------------------------
    # Identifier validation
    # ------------------------------------------------------------------

    duplicate_security = df.duplicated(
        ["eom", "security_id"],
        keep=False,
    )

    if duplicate_security.any():
        sample = df.loc[
            duplicate_security,
            [
                "eom",
                "id",
                "security_id",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate (eom, security_id) "
            f"observations found:\n{sample}"
        )

    duplicate_id = df.duplicated(
        ["eom", "id"],
        keep=False,
    )

    if duplicate_id.any():
        sample = df.loc[
            duplicate_id,
            [
                "eom",
                "id",
                "security_id",
            ],
        ].head(20)

        raise ValueError(
            "Duplicate (eom, id) "
            f"observations found:\n{sample}"
        )

    # ------------------------------------------------------------------
    # Cross-sectional rank normalization
    # ------------------------------------------------------------------

    # Rank characteristics before target-availability filtering.
    df = _monthly_rank(
        df,
        rank_features,
        config.preprocessing
        .missing_feature_fill,
    )

    # ------------------------------------------------------------------
    # Exact-calendar derived characteristics
    # ------------------------------------------------------------------

    if include_feature40_lag1:
        df = add_feature40_lag1(
            df,
            missing_fill=(
                config.preprocessing
                .missing_feature_fill
            ),
            security_id_col="security_id",
        )

    if include_feature40_lag2:
        df = add_feature40_lag2(
            df,
            missing_fill=(
                config.preprocessing
                .missing_feature_fill
            ),
            security_id_col="security_id",
        )

    # ------------------------------------------------------------------
    # Target timing
    # ------------------------------------------------------------------

    df["target_date"] = (
        df["eom"]
        + pd.offsets.MonthEnd(1)
    )

    df["target_available"] = (
        df[config.target_col]
        .notna()
        .astype(bool)
    )

    df = (
        df.sort_values(
            ["eom", "security_id"]
        )
        .reset_index(drop=True)
    )

    audit = {
        "rows": int(len(df)),
        "unique_stocks": int(
            df["security_id"].nunique()
        ),
        "months": int(
            df["eom"].nunique()
        ),
        "start": str(
            df["eom"].min().date()
        ),
        "end": str(
            df["eom"].max().date()
        ),
        "target_available": int(
            df["target_available"].sum()
        ),
        "target_missing": int(
            (~df["target_available"]).sum()
        ),
        "size_groups": {
            str(group): int(count)
            for group, count
            in df["size_grp"]
            .value_counts()
            .items()
        },
    }

    return df, audit


# ----------------------------------------------------------------------
# Frozen OOS evaluation universe
# ----------------------------------------------------------------------

def build_oos_universe(
    panel: pd.DataFrame,
    schedule: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Build the common OOS observation universe used by every model.

    The universe is fixed independently of model predictions so every model is
    evaluated on the same stock-month keys.
    """
    year_to_refit = (
        schedule
        .set_index("test_year")["refit_id"]
        .to_dict()
    )

    out = panel.loc[
        panel["eom"].dt.year.isin(
            year_to_refit
        )
    ].copy()

    out["test_year"] = (
        out["eom"]
        .dt.year
        .astype(int)
    )

    out["refit_id"] = (
        out["test_year"]
        .map(year_to_refit)
        .astype(int)
    )

    columns = [
        "eom",
        "target_date",
        "id",
        "security_id",
        "excntry",
        "me",
        "size_grp",
        target_col,
        "target_available",
        "test_year",
        "refit_id",
    ]

    out = out[columns].rename(
        columns={
            "excntry": "country",
            target_col: "y_true",
        }
    )

    return (
        out.sort_values(
            ["eom", "security_id"]
        )
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# Target-availability audit
# ----------------------------------------------------------------------

def target_availability_summary(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize missing next-month returns without changing the research sample.
    """
    rows: list[dict] = []

    def add_group(
        dimension: str,
        group: str,
        frame: pd.DataFrame,
    ) -> None:
        n = len(frame)
        available = int(
            frame["target_available"].sum()
        )
        missing = n - available

        rows.append(
            {
                "dimension": dimension,
                "group": str(group),
                "n_observations": n,
                "n_target_available": (
                    available
                ),
                "n_target_missing": missing,
                "target_missing_rate": (
                    float(missing / n)
                    if n
                    else np.nan
                ),
            }
        )

    add_group(
        "overall",
        "all",
        panel,
    )

    for size_group, group in panel.groupby(
        "size_grp",
        observed=True,
        sort=True,
    ):
        add_group(
            "size_group",
            str(size_group),
            group,
        )

    for year, group in panel.groupby(
        panel["eom"].dt.year,
        sort=True,
    ):
        add_group(
            "signal_year",
            str(int(year)),
            group,
        )

    return pd.DataFrame(rows)
