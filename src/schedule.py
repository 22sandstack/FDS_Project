from __future__ import annotations

import pandas as pd

from .config import WindowConfig


def make_rolling_schedule(panel: pd.DataFrame, config: WindowConfig) -> pd.DataFrame:
    min_year = int(panel["eom"].dt.year.min())
    max_year = int(panel["eom"].dt.year.max())
    first_test_year = min_year + config.train_years + config.validation_years
    rows = []
    for refit_id, test_year in enumerate(range(first_test_year, max_year + 1)):
        rows.append({
            "refit_id": refit_id,
            "test_year": test_year,
            "train_start_year": test_year - config.validation_years - config.train_years,
            "train_end_year": test_year - config.validation_years - 1,
            "validation_start_year": test_year - config.validation_years,
            "validation_end_year": test_year - 1,
            "test_start_year": test_year,
            "test_end_year": test_year,
        })
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise ValueError("The panel is too short for the configured rolling schedule.")
    return schedule


def year_slice(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    mask = df["eom"].dt.year.between(int(start_year), int(end_year))
    return df.loc[mask]

