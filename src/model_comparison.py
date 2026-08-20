from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import MODEL_REGISTRY


COMPARISON_COLUMNS: tuple[str, ...] = (
    "model_id",
    "model_type",
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


def load_saved_model_metrics(
    run_dir: Path,
) -> pd.DataFrame:
    """
    Load every saved model metric JSON in the experiment directory.

    This intentionally does not restrict results to MODEL_REGISTRY. Derived
    models created later, such as the fixed 50/50 chosen model, are included
    automatically once their metric JSON has been saved in `metrics/`.
    """
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"

    if not metrics_dir.exists():
        return pd.DataFrame(
            columns=COMPARISON_COLUMNS
        )

    rows: list[dict] = []

    for path in sorted(
        metrics_dir.glob("*.json")
    ):
        metric = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        model_id = str(
            metric.get(
                "model_id",
                path.stem,
            )
        )

        rows.append(
            {
                **metric,
                "model_id": model_id,
                "model_type": (
                    "standalone"
                    if model_id
                    in MODEL_REGISTRY
                    else "derived"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=COMPARISON_COLUMNS
        )

    result = pd.DataFrame(rows)

    for column in COMPARISON_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    return result.loc[
        :,
        list(COMPARISON_COLUMNS),
    ]


def build_model_comparison_table(
    run_dir: Path,
) -> pd.DataFrame:
    """
    Build the descriptive common model-comparison table.

    No pairwise significance tests are run here. Formal tests for the fixed
    chosen model belong in chosen_model_analysis.py.

    The table is sorted primarily by pooled OOS R², with Rank IC and Sharpe
    used only as deterministic secondary sort keys.
    """
    run_dir = Path(run_dir)

    comparison = load_saved_model_metrics(
        run_dir
    )

    if comparison.empty:
        comparison.to_csv(
            run_dir
            / "model_comparison.csv",
            index=False,
        )
        return comparison

    comparison = (
        comparison.sort_values(
            by=[
                "pooled_oos_r2",
                "mean_monthly_rank_ic",
                "sharpe_ratio",
                "model_id",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    comparison.to_csv(
        run_dir
        / "model_comparison.csv",
        index=False,
    )

    return comparison
