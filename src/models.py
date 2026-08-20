from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import (
    FEATURES_20,
    FEATURES_40,
    FEATURES_40_WITH_LAG1,
    FEATURES_40_WITH_LAG2,
    FEATURES_40_DYNAMIC,
    FEATURES_60,
    FEATURES_80,
    FEATURES_100,
)


# Increment when training changes invalidate fitted artifacts.
TRAINING_VERSION = "training_v3_final"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    trainer_id: str
    feature_set_id: str
    data_layout: str = "flat"
    params: dict[str, Any] = field(default_factory=dict)


LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 100,
}

NN_PARAMS = {
    "dropout": 0.05,
    "batch_size": 32768,
    "max_epochs": 50,
    "patience": 10,
    "min_delta": 1e-6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "mixed_precision": True,
}

DEEPSET_PARAMS = {
    "encoder_hidden_dim": 32,
    "embedding_dim": 8,
    "predictor_hidden_dims": [32, 16],
    "dropout": 0.05,
    "months_per_batch": 4,
    "max_epochs": 100,
    "patience": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


def with_params(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a copy of shared parameters with explicit overrides."""
    return {**base, **overrides}


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Core20 benchmark models.
    "LASSO_20": ModelSpec(
        model_id="LASSO_20",
        trainer_id="lasso",
        feature_set_id="RANKED_CHARACTERISTICS",
        params={
            "alphas": [
                1e-6,
                3e-6,
                1e-5,
                3e-5,
                1e-4,
                3e-4,
                1e-3,
                3e-3,
                1e-2,
            ],
            "max_iter": 10_000,
        },
    ),
    "LGBM_20": ModelSpec(
        model_id="LGBM_20",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),
    "XGBOOST_20": ModelSpec(
        model_id="XGBOOST_20",
        trainer_id="xgboost",
        feature_set_id="RANKED_CHARACTERISTICS",
        params={
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 10.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 100,
        },
    ),
    "NN2_20": ModelSpec(
        model_id="NN2_20",
        trainer_id="feedforward_nn",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(NN_PARAMS, hidden_dims=[32, 16]),
    ),
    "NN3_20": ModelSpec(
        model_id="NN3_20",
        trainer_id="feedforward_nn",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(NN_PARAMS, hidden_dims=[32, 16, 8]),
    ),
    "NN4_20": ModelSpec(
        model_id="NN4_20",
        trainer_id="feedforward_nn",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(NN_PARAMS, hidden_dims=[32, 16, 8, 4]),
    ),

    # Characteristic-breadth models: 20 -> 100.
    "LGBM_40": ModelSpec(
        model_id="LGBM_40",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),
    "LGBM_60": ModelSpec(
        model_id="LGBM_60",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),
    "LGBM_80": ModelSpec(
        model_id="LGBM_80",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),
    "LGBM_100": ModelSpec(
        model_id="LGBM_100",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),

    # Lagged-information models.
    "LGBM_40_LAG1": ModelSpec(
        model_id="LGBM_40_LAG1",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),
    "LGBM_40_LAG12": ModelSpec(
        model_id="LGBM_40_LAG12",
        trainer_id="lightgbm",
        feature_set_id="RANKED_CHARACTERISTICS",
        params=with_params(LGBM_PARAMS),
    ),

    # Cross-sectional cloud and neural models.
    "MLP_40": ModelSpec(
        model_id="MLP_40",
        trainer_id="deepset",
        feature_set_id="RANKED_CHARACTERISTICS",
        data_layout="monthly_panel",
        params=with_params(
            DEEPSET_PARAMS,
            include_market_context=False,
        ),
    ),
    "MLP_40_LAG1": ModelSpec(
        model_id="MLP_40_LAG1",
        trainer_id="deepset",
        feature_set_id="RANKED_CHARACTERISTICS",
        data_layout="monthly_panel",
        params=with_params(
            DEEPSET_PARAMS,
            include_market_context=False,
        ),
    ),
    "DEEPSET_40": ModelSpec(
        model_id="DEEPSET_40",
        trainer_id="deepset",
        feature_set_id="RANKED_CHARACTERISTICS",
        data_layout="monthly_panel",
        params=with_params(
            DEEPSET_PARAMS,
            include_market_context=True,
        ),
    ),
    "DEEPSET_40_LAG1": ModelSpec(
        model_id="DEEPSET_40_LAG1",
        trainer_id="deepset",
        feature_set_id="RANKED_CHARACTERISTICS",
        data_layout="monthly_panel",
        params=with_params(
            DEEPSET_PARAMS,
            include_market_context=True,
        ),
    ),
    "DEEPSET_40_DYNAMIC": ModelSpec(
        model_id="DEEPSET_40_DYNAMIC",
        trainer_id="deepset",
        feature_set_id="RANKED_CHARACTERISTICS",
        data_layout="monthly_panel",
        params=with_params(
            DEEPSET_PARAMS,
            include_market_context=True,
        ),
    ),
}


MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "LASSO_20": FEATURES_20,
    "LGBM_20": FEATURES_20,
    "XGBOOST_20": FEATURES_20,
    "NN2_20": FEATURES_20,
    "NN3_20": FEATURES_20,
    "NN4_20": FEATURES_20,
    "LGBM_40": FEATURES_40,
    "LGBM_60": FEATURES_60,
    "LGBM_80": FEATURES_80,
    "LGBM_100": FEATURES_100,
    "LGBM_40_LAG1": FEATURES_40_WITH_LAG1,
    "LGBM_40_LAG12": FEATURES_40_WITH_LAG2,
    "MLP_40": FEATURES_40,
    "MLP_40_LAG1": FEATURES_40_WITH_LAG1,
    "DEEPSET_40": FEATURES_40,
    "DEEPSET_40_LAG1": FEATURES_40_WITH_LAG1,
    "DEEPSET_40_DYNAMIC": FEATURES_40_DYNAMIC,
}


def set_seed(seed: int) -> None:
    """Set common random seeds used by the custom neural trainers."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _print_epoch_progress(
    epoch: int,
    max_epochs: int,
    train_mse: float,
    validation_mse: float,
    best_validation_mse: float,
    bad_epochs: int,
    patience: int,
    seconds: float,
) -> None:
    print(
        f"    epoch {epoch:03d}/{max_epochs:03d} | "
        f"train_mse={train_mse:.8f} | val_mse={validation_mse:.8f} | "
        f"best_val_mse={best_validation_mse:.8f} | "
        f"early_stop={bad_epochs:02d}/{patience:02d} | {seconds:.1f}s"
    )


def _finite_target_mask(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    """Return the common finite-label mask used by every trainer."""
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    return np.isfinite(target)


def _flat_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build train, validation, and test matrices for flat estimators."""
    train = train.loc[_finite_target_mask(train, target_col)]
    validation = validation.loc[_finite_target_mask(validation, target_col)]
    columns = list(features)

    return (
        train[columns].to_numpy(dtype=np.float32),
        train[target_col].to_numpy(dtype=np.float64),
        validation[columns].to_numpy(dtype=np.float32),
        validation[target_col].to_numpy(dtype=np.float64),
        test[columns].to_numpy(dtype=np.float32),
    )


def train_lasso(
    train,
    validation,
    test,
    features,
    target_col,
    params,
    paths,
    seed,
    device,
):
    """Fit LASSO and select alpha using validation MSE."""
    del device

    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error

    x_train, y_train, x_validation, y_validation, x_test = _flat_arrays(
        train,
        validation,
        test,
        features,
        target_col,
    )

    best_model = None
    best_alpha = None
    best_validation_mse = np.inf

    for alpha in params["alphas"]:
        model = Lasso(
            alpha=alpha,
            max_iter=params["max_iter"],
            fit_intercept=True,
            random_state=seed,
        )
        model.fit(x_train, y_train)

        validation_prediction = model.predict(x_validation)
        validation_mse = mean_squared_error(y_validation, validation_prediction)

        if validation_mse < best_validation_mse:
            best_model = model
            best_alpha = alpha
            best_validation_mse = validation_mse

    if best_model is None:
        raise RuntimeError("LASSO did not produce a valid fitted model.")

    with paths["model"].open("wb") as handle:
        pickle.dump(best_model, handle)

    return best_model.predict(x_test), {
        "best_alpha": float(best_alpha),
        "validation_mse": float(best_validation_mse),
        "selection_rule": "minimum_validation_mse",
    }


def train_lightgbm(
    train,
    validation,
    test,
    features,
    target_col,
    params,
    paths,
    seed,
    device,
):
    """Fit LightGBM with validation-MSE early stopping."""
    del device

    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    x_train, y_train, x_validation, y_validation, x_test = _flat_arrays(
        train,
        validation,
        test,
        features,
        target_col,
    )

    model_params = {
        key: value
        for key, value in params.items()
        if key not in {"n_estimators", "early_stopping_rounds"}
    }

    model = LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        n_estimators=params["n_estimators"],
        **model_params,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(
                params["early_stopping_rounds"],
                verbose=False,
            ),
            lgb.log_evaluation(0),
        ],
    )

    best_iteration = int(model.best_iteration_)

    with paths["model"].open("wb") as handle:
        pickle.dump(model, handle)

    return model.predict(x_test, num_iteration=best_iteration), {
        "best_iteration": best_iteration,
        "validation_mse": float(model.best_score_["valid_0"]["l2"]),
        "selection_rule": "validation_mse_early_stopping",
    }


def train_xgboost(
    train,
    validation,
    test,
    features,
    target_col,
    params,
    paths,
    seed,
    device,
):
    """Fit XGBoost with validation-set early stopping."""
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor

    x_train, y_train, x_validation, y_validation, x_test = _flat_arrays(
        train,
        validation,
        test,
        features,
        target_col,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if device == "cuda" else "cpu",
        **params,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    prediction_kwargs = (
        {}
        if best_iteration is None
        else {"iteration_range": (0, int(best_iteration) + 1)}
    )

    validation_prediction = model.predict(x_validation, **prediction_kwargs)
    validation_mse = mean_squared_error(y_validation, validation_prediction)

    model.save_model(paths["model"])

    return model.predict(x_test, **prediction_kwargs), {
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "validation_mse": float(validation_mse),
        "selection_rule": "validation_mse_early_stopping",
    }


def build_feedforward_network(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
):
    """Build the common feed-forward architecture used by NN2/NN3/NN4."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, 1))
    return nn.Sequential(*layers)


def train_feedforward_nn(
    train,
    validation,
    test,
    features,
    target_col,
    params,
    paths,
    seed,
    device,
):
    """Train NN2/NN3/NN4 with one common validation-MSE procedure."""
    import torch
    import torch.nn.functional as F

    set_seed(seed)

    x_train, y_train, x_validation, y_validation, x_test = _flat_arrays(
        train,
        validation,
        test,
        features,
        target_col,
    )

    torch_device = torch.device(device)
    model = build_feedforward_network(
        input_dim=len(features),
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    ).to(torch_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=torch_device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=torch_device)
    x_validation = torch.as_tensor(
        x_validation,
        dtype=torch.float32,
        device=torch_device,
    )
    y_validation = torch.as_tensor(
        y_validation,
        dtype=torch.float32,
        device=torch_device,
    )

    batch_size = int(params["batch_size"])
    amp_enabled = bool(
        params.get("mixed_precision", False) and torch_device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_validation_mse = np.inf
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, params["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()

        permutation = torch.randperm(len(x_train), device=torch_device)
        train_squared_error = 0.0
        train_observations = 0

        for start in range(0, len(x_train), batch_size):
            indices = permutation[start : start + batch_size]
            if len(indices) < 2:
                continue

            xb = x_train.index_select(0, indices)
            yb = y_train.index_select(0, indices)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=torch_device.type,
                enabled=amp_enabled,
            ):
                prediction = model(xb).squeeze(-1)
                loss = F.mse_loss(prediction, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_squared_error += float(
                ((prediction.detach() - yb) ** 2).sum().cpu()
            )
            train_observations += len(yb)

        train_mse = (
            train_squared_error / train_observations
            if train_observations
            else np.nan
        )

        model.eval()
        with torch.inference_mode():
            validation_prediction = model(x_validation).squeeze(-1)
            validation_mse = float(
                F.mse_loss(validation_prediction, y_validation).cpu()
            )

        improved = validation_mse < best_validation_mse - params["min_delta"]
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            bad_epochs = 0
            _torch_save_atomic(model.state_dict(), paths["best"])
        else:
            bad_epochs += 1

        _print_epoch_progress(
            epoch=epoch,
            max_epochs=params["max_epochs"],
            train_mse=train_mse,
            validation_mse=validation_mse,
            best_validation_mse=best_validation_mse,
            bad_epochs=bad_epochs,
            patience=params["patience"],
            seconds=time.perf_counter() - epoch_start,
        )

        if bad_epochs >= params["patience"]:
            break

    if not paths["best"].exists():
        raise RuntimeError(
            "Neural-network training did not produce a valid best checkpoint."
        )

    model.load_state_dict(torch.load(paths["best"], map_location=torch_device))
    model.eval()

    x_test = torch.as_tensor(x_test, dtype=torch.float32, device=torch_device)
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(x_test), batch_size):
            prediction = model(x_test[start : start + batch_size]).squeeze(-1)
            predictions.append(prediction.float().cpu().numpy())

    _torch_save_atomic(model.state_dict(), paths["model"])

    return np.concatenate(predictions), {
        "hidden_dims": list(params["hidden_dims"]),
        "best_epoch": int(best_epoch),
        "validation_mse": float(best_validation_mse),
        "selection_rule": "validation_mse_early_stopping",
    }


def build_deepset_core(n_features: int, params: dict[str, Any]):
    """Build a permutation-equivariant model for one monthly stock set."""
    import torch
    import torch.nn as nn

    class DeepSetCore(nn.Module):
        def __init__(self):
            super().__init__()
            embedding_dim = params["embedding_dim"]
            self.stock_encoder = nn.Sequential(
                nn.Linear(n_features, params["encoder_hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(params["dropout"]),
                nn.Linear(params["encoder_hidden_dim"], embedding_dim),
                nn.ReLU(),
            )

            layers = []
            previous = n_features + embedding_dim
            for width in params["predictor_hidden_dims"]:
                layers += [
                    nn.Linear(previous, width),
                    nn.ReLU(),
                    nn.Dropout(params["dropout"]),
                ]
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.return_predictor = nn.Sequential(*layers)

        def forward(self, x):
            if params.get("include_market_context", True):
                embedding = self.stock_encoder(x)
                n_stocks = embedding.shape[0]
                if n_stocks > 1:
                    cloud = (
                        embedding.sum(dim=0, keepdim=True) - embedding
                    ) / (n_stocks - 1)
                else:
                    cloud = torch.zeros_like(embedding)
            else:
                cloud = x.new_zeros((x.shape[0], params["embedding_dim"]))

            return self.return_predictor(
                torch.cat([x, cloud], dim=1)
            ).squeeze(-1)

    return DeepSetCore()


def _torch_save_atomic(payload, path: Path) -> None:
    """Keep the previous checkpoint valid if a runtime stops during saving."""
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_deepset(
    train,
    validation,
    test,
    features,
    target_col,
    params,
    paths,
    seed,
    device,
):
    """Train on complete monthly sets while masking labels only in the loss."""
    import torch

    set_seed(seed)
    torch_device = torch.device(device)
    model = build_deepset_core(len(features), params).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    def month_indices(frame):
        return [
            np.asarray(index, dtype=np.int64)
            for index in frame.groupby("eom", sort=True).indices.values()
        ]

    train_months = month_indices(train)
    validation_months = month_indices(validation)
    test_months = month_indices(test)

    def month_tensors(frame, index):
        monthly = frame.iloc[index]
        x = torch.from_numpy(
            monthly[list(features)].to_numpy(np.float32)
        ).to(torch_device)
        y_array = monthly[target_col].to_numpy(np.float32)
        mask_array = np.isfinite(y_array)
        y = torch.from_numpy(np.where(mask_array, y_array, 0.0)).to(torch_device)
        mask = torch.from_numpy(mask_array).to(torch_device)
        return x, y, mask

    def train_epoch(epoch):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_months))
        total_squared_error = 0.0
        total_count = 0
        months_per_batch = params["months_per_batch"]

        for start in range(0, len(order), months_per_batch):
            optimizer.zero_grad(set_to_none=True)
            batch_squared_error = None
            batch_count = 0

            for month_number in order[start : start + months_per_batch]:
                x, y, mask = month_tensors(train, train_months[month_number])
                if not bool(mask.any()):
                    continue

                errors = (model(x)[mask] - y[mask]) ** 2
                month_squared_error = errors.sum()
                batch_squared_error = (
                    month_squared_error
                    if batch_squared_error is None
                    else batch_squared_error + month_squared_error
                )
                batch_count += int(mask.sum().item())

            if batch_count == 0:
                continue

            loss = batch_squared_error / batch_count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_squared_error += float(batch_squared_error.detach().cpu())
            total_count += batch_count

        return total_squared_error / max(total_count, 1)

    def evaluate(frame, months):
        model.eval()
        total_squared_error = 0.0
        total_count = 0

        with torch.no_grad():
            for index in months:
                x, y, mask = month_tensors(frame, index)
                if not bool(mask.any()):
                    continue

                errors = (model(x)[mask] - y[mask]) ** 2
                total_squared_error += float(errors.sum().cpu())
                total_count += int(mask.sum().item())

        return total_squared_error / max(total_count, 1)

    start_epoch = 0
    best_loss = np.inf
    bad_epochs = 0
    best_epoch = -1

    if paths["latest"].exists():
        checkpoint = torch.load(paths["latest"], map_location=torch_device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]
        best_epoch = checkpoint["best_epoch"]

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )

        print(f"    resuming at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, params["max_epochs"]):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(epoch)
        validation_loss = evaluate(validation, validation_months)

        improved = validation_loss < best_loss - 1e-12
        if improved:
            best_loss = validation_loss
            bad_epochs = 0
            best_epoch = epoch
            _torch_save_atomic(model.state_dict(), paths["best"])
        else:
            bad_epochs += 1

        _torch_save_atomic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            paths["latest"],
        )

        _print_epoch_progress(
            epoch=epoch + 1,
            max_epochs=params["max_epochs"],
            train_mse=train_loss,
            validation_mse=validation_loss,
            best_validation_mse=best_loss,
            bad_epochs=bad_epochs,
            patience=params["patience"],
            seconds=time.perf_counter() - epoch_start,
        )

        if bad_epochs >= params["patience"]:
            break

    if not paths["best"].exists():
        raise RuntimeError(
            "DeepSets training did not produce a valid best checkpoint."
        )

    model.load_state_dict(torch.load(paths["best"], map_location=torch_device))
    model.eval()

    predictions = np.empty(len(test), dtype=np.float32)
    with torch.no_grad():
        for index in test_months:
            x, _, _ = month_tensors(test, index)
            predictions[index] = model(x).cpu().numpy()

    _torch_save_atomic(model.state_dict(), paths["model"])

    return predictions, {
        "best_validation_mse": float(best_loss),
        "best_epoch": int(best_epoch + 1),
        "embedding_dim": int(params["embedding_dim"]),
        "uses_leave_one_out_cloud": bool(
            params.get("include_market_context", True)
        ),
        "training_loss_weighting": "pooled_stock_mse",
        "selection_rule": "validation_mse_early_stopping",
    }


TRAINERS: dict[str, Callable] = {
    "lasso": train_lasso,
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "feedforward_nn": train_feedforward_nn,
    "deepset": train_deepset,
}
