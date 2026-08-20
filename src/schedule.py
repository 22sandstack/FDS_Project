from __future__ import annotations

import pandas as pd

from .config import WindowConfig


def make_rolling_schedule(
    panel: pd.DataFrame,
    config: WindowConfig,
) -> pd.DataFrame:
    """
    Build the annual rolling train/validation/test schedule.

    With the default 15/4/1 design:

        test year 1999
        train      1980-1994
        validation 1995-1998
        test       1999

        test year 2000
        train      1981-1995
        validation 1996-1999
        test       2000

    The window advances by one calendar year at each refit.
    """
    if panel.empty:
        raise ValueError(
            "Cannot build a rolling schedule from an empty panel."
        )

    years = panel["eom"].dt.year

    min_year = int(years.min())
    max_year = int(years.max())

    first_test_year = (
        min_year
        + config.train_years
        + config.validation_years
    )

    rows: list[dict[str, int]] = []

    for refit_id, test_year in enumerate(
        range(first_test_year, max_year + 1)
    ):
        validation_start_year = (
            test_year - config.validation_years
        )
        validation_end_year = test_year - 1

        train_end_year = (
            validation_start_year - 1
        )
        train_start_year = (
            train_end_year
            - config.train_years
            + 1
        )

        rows.append(
            {
                "refit_id": refit_id,
                "test_year": test_year,
                "train_start_year": train_start_year,
                "train_end_year": train_end_year,
                "validation_start_year": validation_start_year,
                "validation_end_year": validation_end_year,
                "test_start_year": test_year,
                "test_end_year": test_year,
            }
        )

    schedule = pd.DataFrame(rows)

    if schedule.empty:
        raise ValueError(
            "The panel is too short for the configured rolling schedule."
        )

    # Validate rolling-window structure.
    train_lengths = (
        schedule["train_end_year"]
        - schedule["train_start_year"]
        + 1
    )

    validation_lengths = (
        schedule["validation_end_year"]
        - schedule["validation_start_year"]
        + 1
    )

    if not train_lengths.eq(
        config.train_years
    ).all():
        raise ValueError(
            "Rolling schedule produced an invalid training-window length."
        )

    if not validation_lengths.eq(
        config.validation_years
    ).all():
        raise ValueError(
            "Rolling schedule produced an invalid validation-window length."
        )

    if not (
        schedule["train_end_year"] + 1
        == schedule["validation_start_year"]
    ).all():
        raise ValueError(
            "Training and validation windows are not contiguous."
        )

    if not (
        schedule["validation_end_year"] + 1
        == schedule["test_start_year"]
    ).all():
        raise ValueError(
            "Validation and test windows are not contiguous."
        )

    return schedule


def year_slice(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Return a copy of observations whose signal dates fall within the year range."""
    mask = df["eom"].dt.year.between(
        int(start_year),
        int(end_year),
    )

    return df.loc[mask].copy()
