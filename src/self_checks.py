from __future__ import annotations

import numpy as np
import pandas as pd

from .data import add_exact_calendar_lag2
from .models import (
    MODEL_FEATURES, MODEL_REGISTRY, TRAINERS, _convex_validation_weight,
    build_deepset_core,
)


def run_framework_self_checks() -> None:
    """Run fast structural checks before loading the full research panel."""
    missing_trainers = {
        spec.trainer_id for spec in MODEL_REGISTRY.values()
        if spec.trainer_id not in TRAINERS
    }
    if missing_trainers:
        raise AssertionError(f"Registered models have missing trainers: {missing_trainers}")
    expected_counts = {
        "NN2_20": 20,
        "NN4_20": 20,
        "LGBM_40_LAG2": 122,
        "MLP_40": 40,
        "DEEPSET_40": 40,
        "HYBRID_MLP40_DEEPSET40": 40,
        "DEEPSET_40_LAG1": 81,
        "DEEPSET_40_DYNAMIC": 121,
    }
    for model_id, count in expected_counts.items():
        if model_id not in MODEL_REGISTRY:
            raise AssertionError(f"Missing model registration: {model_id}")
        if MODEL_REGISTRY[model_id].trainer_id not in TRAINERS:
            raise AssertionError(f"Missing trainer for: {model_id}")
        if len(MODEL_FEATURES[model_id]) != count:
            raise AssertionError(
                f"{model_id} has {len(MODEL_FEATURES[model_id])} features, expected {count}."
            )

    sample = pd.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b"],
            "eom": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-01-31", "2020-03-31"]
            ),
            "x": np.array([1.0, 2.0, 3.0, 10.0, 30.0], dtype=np.float32),
        }
    )
    lagged = add_exact_calendar_lag2(sample, ("x",), ("x_lag2",), "available", 0.0)
    march = lagged.loc[lagged.eom.eq(pd.Timestamp("2020-03-31"))].set_index("id")
    if march.loc["a", "x_lag2"] != 1.0 or march.loc["b", "x_lag2"] != 10.0:
        raise AssertionError("Lag-2 construction is not an exact-calendar join.")

    # The validation blend should recover an interior optimum and respect
    # convex boundaries without ever consulting test outcomes.
    y = np.array([0.0, 1.0], dtype=np.float64)
    mlp_prediction = np.array([0.0, 0.0], dtype=np.float64)
    deepset_prediction = np.array([0.0, 2.0], dtype=np.float64)
    weight, n_weight, fallback = _convex_validation_weight(
        y, mlp_prediction, deepset_prediction
    )
    if not np.isclose(weight, 0.5) or n_weight != 2 or fallback:
        raise AssertionError("Convex validation weighting failed its interior test.")

    try:
        import torch
    except ImportError:
        return

    params = {
        "encoder_hidden_dim": 8,
        "embedding_dim": 4,
        "predictor_hidden_dims": [8, 4],
        "dropout": 0.0,
        "include_market_context": False,
    }
    torch.manual_seed(1)
    mlp = build_deepset_core(3, params).eval()
    first_set = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    second_set = torch.tensor([[0.1, 0.2, 0.3], [-0.9, 0.8, -0.7]])
    if not torch.allclose(mlp(first_set)[0], mlp(second_set)[0]):
        raise AssertionError("MLP_40 depends on other stocks despite context being disabled.")

    params["include_market_context"] = True
    torch.manual_seed(1)
    deepset = build_deepset_core(3, params).eval()
    permutation = torch.tensor([1, 0])
    if not torch.allclose(
        deepset(first_set[permutation]), deepset(first_set)[permutation], atol=1e-6
    ):
        raise AssertionError("DeepSets predictions are not permutation equivariant.")
