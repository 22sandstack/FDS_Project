from __future__ import annotations

import json
import pandas as pd

from .config import ExperimentConfig
from .ensemble import ENSEMBLE_ID, build_fixed_fifty_fifty


COMPONENT_IDS = ("LGBM_40", "DEEPSET_40_DYNAMIC")
COUNTRY_NAMES = {
    "GBR": "United Kingdom",
    "AUS": "Australia",
    "DEU": "Germany",
    "FRA": "France",
}
RESULT_COLUMNS = (
    "pooled_oos_r2",
    "mean_monthly_rank_ic",
    "rank_ic_newey_west_t_stat",
    "rank_ic_n_months",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "portfolio_newey_west_t_stat",
    "max_drawdown",
)


def country_validation_table(
    config: ExperimentConfig,
    *,
    build_chosen_model: bool = True,
) -> pd.DataFrame:
    """Return the frozen ensemble and its two components for one country."""
    country = config.universe.country
    if country not in COUNTRY_NAMES:
        raise ValueError(f"Unsupported developed-market country: {country}")

    if build_chosen_model:
        build_fixed_fifty_fifty(config)

    rows = []
    for model_id in (ENSEMBLE_ID, *COMPONENT_IDS):
        path = config.run_dir / "metrics" / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "country": country,
                "country_name": COUNTRY_NAMES[country],
                "model_id": model_id,
                **{column: metrics.get(column) for column in RESULT_COLUMNS},
            }
        )
    return pd.DataFrame(rows)
