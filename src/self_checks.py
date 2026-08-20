from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURES_40,
    FINAL_MODEL_ROSTER,
)
from .data import (
    add_feature40_lag1,
    add_feature40_lag2,
)
from .models import (
    MODEL_FEATURES,
    MODEL_REGISTRY,
    TRAINERS,
    build_deepset_core,
)


EXPECTED_FEATURE_COUNTS: dict[str, int] = {
    "LASSO_20": 20,
    "LGBM_20": 20,
    "XGBOOST_20": 20,
    "NN2_20": 20,
    "NN3_20": 20,
    "NN4_20": 20,
    "LGBM_40": 40,
    "LGBM_60": 60,
    "LGBM_80": 80,
    "LGBM_100": 100,
    "LGBM_40_LAG1": 81,
    "LGBM_40_LAG12": 122,
    "MLP_40": 40,
    "MLP_40_LAG1": 81,
    "DEEPSET_40": 40,
    "DEEPSET_40_LAG1": 81,
    "DEEPSET_40_DYNAMIC": 121,
}


def _check_model_registry() -> None:
    """Verify the frozen standalone model roster and trainer wiring."""
    roster = set(
        FINAL_MODEL_ROSTER
    )

    registry = set(
        MODEL_REGISTRY
    )

    missing_models = (
        roster - registry
    )

    if missing_models:
        raise AssertionError(
            "Final roster models are missing "
            f"from MODEL_REGISTRY: "
            f"{sorted(missing_models)}"
        )

    unexpected_models = (
        registry - roster
    )

    if unexpected_models:
        raise AssertionError(
            "MODEL_REGISTRY contains models "
            "outside the final standalone "
            f"roster: "
            f"{sorted(unexpected_models)}"
        )

    missing_feature_sets = [
        model_id
        for model_id
        in FINAL_MODEL_ROSTER
        if model_id
        not in MODEL_FEATURES
    ]

    if missing_feature_sets:
        raise AssertionError(
            "Models are missing feature "
            f"definitions: "
            f"{missing_feature_sets}"
        )

    missing_trainers = {
        MODEL_REGISTRY[
            model_id
        ].trainer_id
        for model_id
        in FINAL_MODEL_ROSTER
        if MODEL_REGISTRY[
            model_id
        ].trainer_id
        not in TRAINERS
    }

    if missing_trainers:
        raise AssertionError(
            "Registered models reference "
            f"missing trainers: "
            f"{sorted(missing_trainers)}"
        )


def _check_feature_counts() -> None:
    """Verify that every final model uses the intended number of features."""
    if set(
        EXPECTED_FEATURE_COUNTS
    ) != set(
        FINAL_MODEL_ROSTER
    ):
        raise AssertionError(
            "EXPECTED_FEATURE_COUNTS does "
            "not exactly match "
            "FINAL_MODEL_ROSTER."
        )

    for model_id in (
        FINAL_MODEL_ROSTER
    ):
        observed = len(
            MODEL_FEATURES[
                model_id
            ]
        )

        expected = (
            EXPECTED_FEATURE_COUNTS[
                model_id
            ]
        )

        if observed != expected:
            raise AssertionError(
                f"{model_id} has "
                f"{observed} features; "
                f"expected {expected}."
            )

        if len(
            set(
                MODEL_FEATURES[
                    model_id
                ]
            )
        ) != observed:
            raise AssertionError(
                f"{model_id} contains "
                "duplicate feature names."
            )


def _lag_test_panel() -> pd.DataFrame:
    """
    Build a tiny panel with both consecutive observations and calendar gaps.
    """
    rows = []

    observations = (
        (
            "a",
            "2020-01-31",
            1.0,
        ),
        (
            "a",
            "2020-02-29",
            2.0,
        ),
        (
            "a",
            "2020-03-31",
            3.0,
        ),
        (
            "b",
            "2020-01-31",
            10.0,
        ),
        (
            "b",
            "2020-03-31",
            30.0,
        ),
    )

    for (
        security_id,
        date,
        value,
    ) in observations:
        row = {
            "security_id": (
                security_id
            ),
            "eom": pd.Timestamp(
                date
            ),
        }

        for feature in (
            FEATURES_40
        ):
            row[feature] = np.float32(
                value
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _check_exact_calendar_lag1() -> None:
    """
    Verify lag-1 uses the immediately preceding calendar month only.
    """
    panel = _lag_test_panel()

    lagged = add_feature40_lag1(
        panel,
        missing_fill=0.0,
        security_id_col=(
            "security_id"
        ),
    )

    first_feature = (
        FEATURES_40[0]
    )

    lag_name = (
        f"{first_feature}"
        "_feature40_lag1"
    )

    velocity_name = (
        f"{first_feature}"
        "_feature40_velocity1"
    )

    march = (
        lagged.loc[
            lagged["eom"].eq(
                pd.Timestamp(
                    "2020-03-31"
                )
            )
        ]
        .set_index(
            "security_id"
        )
    )

    if not np.isclose(
        march.loc[
            "a",
            lag_name,
        ],
        2.0,
    ):
        raise AssertionError(
            "Lag-1 failed for a "
            "consecutive observation."
        )

    if not np.isclose(
        march.loc[
            "a",
            velocity_name,
        ],
        1.0,
    ):
        raise AssertionError(
            "Lag-1 velocity is incorrect."
        )

    if not np.isclose(
        march.loc[
            "b",
            lag_name,
        ],
        0.0,
    ):
        raise AssertionError(
            "Lag-1 incorrectly carried "
            "a value across a calendar gap."
        )

    if not np.isclose(
        march.loc[
            "b",
            "features40_lag1_available",
        ],
        0.0,
    ):
        raise AssertionError(
            "Lag-1 availability flag "
            "failed for a calendar gap."
        )


def _check_exact_calendar_lag2() -> None:
    """
    Verify lag-2 joins values from exactly two calendar months earlier.
    """
    panel = _lag_test_panel()

    lagged = add_feature40_lag2(
        panel,
        missing_fill=0.0,
        security_id_col=(
            "security_id"
        ),
    )

    first_feature = (
        FEATURES_40[0]
    )

    lag_name = (
        f"{first_feature}"
        "_feature40_lag2"
    )

    march = (
        lagged.loc[
            lagged["eom"].eq(
                pd.Timestamp(
                    "2020-03-31"
                )
            )
        ]
        .set_index(
            "security_id"
        )
    )

    if not np.isclose(
        march.loc[
            "a",
            lag_name,
        ],
        1.0,
    ):
        raise AssertionError(
            "Lag-2 failed for a stock "
            "with consecutive observations."
        )

    if not np.isclose(
        march.loc[
            "b",
            lag_name,
        ],
        10.0,
    ):
        raise AssertionError(
            "Lag-2 failed to recover the "
            "observation from exactly two "
            "calendar months earlier."
        )

    if not np.isclose(
        march.loc[
            "b",
            "features40_lag2_available",
        ],
        1.0,
    ):
        raise AssertionError(
            "Lag-2 availability flag "
            "is incorrect."
        )


def _check_neural_architectures() -> None:
    """
    Verify the defining structural properties of the MLP and DeepSet models.

    Skipped when PyTorch is not installed.
    """
    try:
        import torch
    except ImportError:
        return

    params = {
        "encoder_hidden_dim": 8,
        "embedding_dim": 4,
        "predictor_hidden_dims": [
            8,
            4,
        ],
        "dropout": 0.0,
        "include_market_context": (
            False
        ),
    }

    first_set = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )

    second_set = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [-0.9, 0.8, -0.7],
        ],
        dtype=torch.float32,
    )

    torch.manual_seed(1)

    mlp = build_deepset_core(
        3,
        params,
    ).eval()

    with torch.inference_mode():
        first_prediction = (
            mlp(first_set)[0]
        )

        second_prediction = (
            mlp(second_set)[0]
        )

    if not torch.allclose(
        first_prediction,
        second_prediction,
        atol=1e-7,
        rtol=0.0,
    ):
        raise AssertionError(
            "An MLP model depends on other "
            "stocks despite market "
            "context being disabled."
        )

    params = {
        **params,
        "include_market_context": True,
    }

    torch.manual_seed(1)

    deepset = build_deepset_core(
        3,
        params,
    ).eval()

    permutation = torch.tensor(
        [1, 0]
    )

    with torch.inference_mode():
        original = deepset(
            first_set
        )

        permuted = deepset(
            first_set[
                permutation
            ]
        )

    if not torch.allclose(
        permuted,
        original[
            permutation
        ],
        atol=1e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "DeepSet predictions are not "
            "permutation equivariant."
        )


def run_framework_self_checks() -> None:
    """
    Run fast structural checks before loading the full research panel.

    These checks validate code/design assumptions only. Saved-artifact
    completeness is handled separately by post_train_audit.py.
    """
    _check_model_registry()
    _check_feature_counts()
    _check_exact_calendar_lag1()
    _check_exact_calendar_lag2()
    _check_neural_architectures()
