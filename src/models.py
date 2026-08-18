from __future__ import annotations

import pickle
import random
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import (
    CORE20_LAG1_FEATURES,
    CORE20_LAG2_FEATURES,
    FEATURES_20,
    FEATURES_40,
    FEATURES_40_WITH_LAG1,
    FEATURES_40_WITH_LAG2,
    FEATURES_40_DYNAMIC,
    FEATURES_60,
    FEATURES_80,
    FEATURES_100,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    trainer_id: str
    feature_set_id: str
    data_layout: str = "flat"
    params: dict[str, Any] = field(default_factory=dict)


_LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 100,
}
_NN_PARAMS = {
    "dropout": 0.05,
    "batch_size": 32768,
    "max_epochs": 50,
    "patience": 10,
    "min_delta": 1e-6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "l1_penalty": 1e-5,
    "mixed_precision": True,
    "device_resident_data": True,
}
_DEEPSET_PARAMS = {
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


def _params(
    base: dict[str, Any],
    leading: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build parameters while preserving the historical signature order."""
    return {**(leading or {}), **base, **overrides}


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LASSO_20": ModelSpec(
        "LASSO_20", "lasso", "RANKED_CHARACTERISTICS",
        params={"alphas": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "max_iter": 10000},
    ),
    "LGBM_20": ModelSpec(
        "LGBM_20", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "XGBOOST_20": ModelSpec(
        "XGBOOST_20", "xgboost", "RANKED_CHARACTERISTICS",
        params={"n_estimators": 3000, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "early_stopping_rounds": 100},
    ),
    "NN2_20": ModelSpec(
        "NN2_20", "feedforward_nn", "RANKED_CHARACTERISTICS",
        params=_params(_NN_PARAMS, {"architecture_version": "nn2_20_v1_device_resident", "hidden_dims": [32, 16]}),
    ),
    "NN2_40": ModelSpec(
        "NN2_40", "feedforward_nn", "RANKED_CHARACTERISTICS",
        params=_params(_NN_PARAMS, {"architecture_version": "nn2_40_v1_device_resident", "hidden_dims": [32, 16]}),
    ),
    "NN3_20": ModelSpec(
        "NN3_20", "feedforward_nn", "RANKED_CHARACTERISTICS",
        params=_params(_NN_PARAMS, {"architecture_version": "nn3_core_v3_device_resident", "hidden_dims": [32, 16, 8]}),
    ),
    "NN4_20": ModelSpec(
        "NN4_20", "feedforward_nn", "RANKED_CHARACTERISTICS",
        params=_params(_NN_PARAMS, {"architecture_version": "nn4_20_v1_device_resident", "hidden_dims": [32, 16, 8, 4]}),
    ),
    "LGBM_20_LAG1": ModelSpec(
        "LGBM_20_LAG1", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "LGBM_40": ModelSpec(
        "LGBM_40", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "LGBM_60": ModelSpec(
        "LGBM_60", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "LGBM_80": ModelSpec(
        "LGBM_80", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "LGBM_100": ModelSpec(
        "LGBM_100", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS),
    ),
    "LGBM_40_LAG1": ModelSpec(
        "LGBM_40_LAG1", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS, {"architecture_version": "lgbm_40_lag1_v1"}),
    ),
    "LGBM_20_LAG2": ModelSpec(
        "LGBM_20_LAG2", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS, {"architecture_version": "lgbm_20_lag2_v1"}),
    ),
    "LGBM_40_LAG2": ModelSpec(
        "LGBM_40_LAG2", "lightgbm", "RANKED_CHARACTERISTICS",
        params=_params(_LGBM_PARAMS, {"architecture_version": "lgbm_40_lag2_v1"}),
    ),
    "MLP_40": ModelSpec(
        "MLP_40", "deepset", "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params=_params(_DEEPSET_PARAMS, {"architecture_version": "mlp_40_matched_v1"}, include_market_context=False),
    ),
    "DEEPSET_40": ModelSpec(
        "DEEPSET_40", "deepset", "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params=_params(_DEEPSET_PARAMS, {"architecture_version": "deepset_40_v1"}, include_market_context=True),
    ),
    "HYBRID_MLP40_DEEPSET40": ModelSpec(
        "HYBRID_MLP40_DEEPSET40", "strict_validation_hybrid",
        "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params={
            "architecture_version": "mlp40_deepset40_strict_3plus1_v2",
            "base_validation_years": 3,
            "weight_validation_years": 1,
            "component_a_id": "MLP_40",
            "component_b_id": "DEEPSET_40",
            "fallback_weight_a": 0.5,
            "weight_objective": "pooled_validation_stock_mse",
            "weight_constraint": "convex_0_1",
        },
    ),
    "HYBRID_LGBM40_DEEPSET40": ModelSpec(
        "HYBRID_LGBM40_DEEPSET40", "strict_validation_hybrid",
        "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params={
            "architecture_version": "lgbm40_deepset40_strict_3plus1_v2",
            "base_validation_years": 3,
            "weight_validation_years": 1,
            "component_a_id": "LGBM_40",
            "component_b_id": "DEEPSET_40",
            "fallback_weight_a": 0.5,
            "weight_objective": "pooled_validation_stock_mse",
            "weight_constraint": "convex_0_1",
        },
    ),
    "HYBRID_LGBM40_DEEPSET40_DYNAMIC": ModelSpec(
        "HYBRID_LGBM40_DEEPSET40_DYNAMIC", "strict_validation_hybrid",
        "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params={
            "architecture_version": "lgbm40_deepset40_dynamic_strict_3plus1_v2",
            "base_validation_years": 3,
            "weight_validation_years": 1,
            "component_a_id": "LGBM_40",
            "component_b_id": "DEEPSET_40_DYNAMIC",
            "fallback_weight_a": 0.5,
            "weight_objective": "pooled_validation_stock_mse",
            "weight_constraint": "convex_0_1",
        },
    ),
    "DEEPSET_40_LAG1": ModelSpec(
        "DEEPSET_40_LAG1", "deepset", "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params=_params(_DEEPSET_PARAMS, {"architecture_version": "deepset_40_lag1_v1"}, include_market_context=True),
    ),
    "DEEPSET_40_DYNAMIC": ModelSpec(
        "DEEPSET_40_DYNAMIC", "deepset", "RANKED_CHARACTERISTICS", data_layout="monthly_panel",
        params=_params(_DEEPSET_PARAMS, {"architecture_version": "deepset_40_dynamic_v1"}, include_market_context=True),
    ),
}

MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "NN2_20": FEATURES_20,
    "NN2_40": FEATURES_40,
    "NN3_20": FEATURES_20,
    "NN4_20": FEATURES_20,
    "LGBM_20_LAG1": CORE20_LAG1_FEATURES,
    "LGBM_20": FEATURES_20,
    "LGBM_40": FEATURES_40,
    "LGBM_60": FEATURES_60,
    "LGBM_80": FEATURES_80,
    "LGBM_100": FEATURES_100,
    "LGBM_40_LAG1": FEATURES_40_WITH_LAG1,
    "LGBM_20_LAG2": CORE20_LAG2_FEATURES,
    "LGBM_40_LAG2": FEATURES_40_WITH_LAG2,
    "MLP_40": FEATURES_40,
    "DEEPSET_40": FEATURES_40,
    "HYBRID_MLP40_DEEPSET40": FEATURES_40,
    "HYBRID_LGBM40_DEEPSET40": FEATURES_40,
    "HYBRID_LGBM40_DEEPSET40_DYNAMIC": FEATURES_40_DYNAMIC,
    "DEEPSET_40_LAG1": FEATURES_40_WITH_LAG1,
    "DEEPSET_40_DYNAMIC": FEATURES_40_DYNAMIC,
}

def set_seed(seed: int) -> None:
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
    """Use one compact progress format for every iterative neural trainer."""
    print(
        f"    epoch {epoch:03d}/{max_epochs:03d} | "
        f"train_mse={train_mse:.8f} | val_mse={validation_mse:.8f} | "
        f"best_val_mse={best_validation_mse:.8f} | "
        f"early_stop={bad_epochs:02d}/{patience:02d} | {seconds:.1f}s"
    )


def _arrays(train, validation, test, features, target_col):
    train = train.loc[train[target_col].notna()]
    validation = validation.loc[validation[target_col].notna()]
    columns = list(features)
    return (
        train[columns].to_numpy(np.float32), train[target_col].to_numpy(np.float64),
        validation[columns].to_numpy(np.float32), validation[target_col].to_numpy(np.float64),
        test[columns].to_numpy(np.float32),
    )


def _validation_signal_stats(validation, target_col, prediction):
    """Secondary economic diagnostic; MSE remains the primary fit objective."""
    frame = validation.loc[validation[target_col].notna(), ["eom", target_col]].copy()
    frame["prediction"] = np.asarray(prediction, dtype=float)
    monthly_ic = []
    for _, month in frame.groupby("eom", sort=True):
        valid = month.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 20 or valid["prediction"].nunique() < 2 or valid[target_col].nunique() < 2:
            continue
        monthly_ic.append(
            valid["prediction"].rank(method="average").corr(
                valid[target_col].rank(method="average")
            )
        )
    prediction = pd.Series(np.asarray(prediction, dtype=float))
    finite_prediction = prediction[np.isfinite(prediction)]
    return {
        "validation_mean_monthly_rank_ic": float(np.nanmean(monthly_ic)) if monthly_ic else np.nan,
        "validation_rank_ic_months": int(np.isfinite(monthly_ic).sum()),
        "validation_unique_predictions": int(finite_prediction.nunique()),
        "validation_prediction_std": float(finite_prediction.std(ddof=1)),
    }


def train_lasso(train, validation, test, features, target_col, params, paths, seed, device):
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error

    xtr, ytr, xva, yva, xte = _arrays(train, validation, test, features, target_col)
    best_model, best_alpha, best_loss, best_signal = None, None, np.inf, None
    for alpha in params["alphas"]:
        model = Lasso(alpha=alpha, max_iter=params["max_iter"], fit_intercept=True, random_state=seed)
        model.fit(xtr, ytr)
        prediction = model.predict(xva)
        loss = mean_squared_error(yva, prediction)
        signal = _validation_signal_stats(validation, target_col, prediction)
        if loss < best_loss:
            best_model, best_alpha, best_loss, best_signal = model, alpha, loss, signal
    with paths["model"].open("wb") as handle:
        pickle.dump(best_model, handle)
    return best_model.predict(xte), {
        "best_alpha": best_alpha, "validation_mse": float(best_loss),
        "selection_rule": "minimum_validation_mse",
        **best_signal,
    }


def train_lightgbm(train, validation, test, features, target_col, params, paths, seed, device):
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    xtr, ytr, xva, yva, xte = _arrays(train, validation, test, features, target_col)
    model_kwargs = {
        k: v for k, v in params.items()
        if k not in {"early_stopping_rounds", "n_estimators"}
    }
    model = LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=-1, verbosity=-1,
        n_estimators=params["n_estimators"], **model_kwargs,
    )
    model.fit(
        xtr, ytr, eval_set=[(xva, yva)], eval_metric="l2",
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = int(model.best_iteration_)
    selected_iteration = best_iteration
    validation_prediction = model.predict(xva, num_iteration=selected_iteration)
    signal = _validation_signal_stats(validation, target_col, validation_prediction)
    with paths["model"].open("wb") as handle:
        pickle.dump(model, handle)
    return model.predict(xte, num_iteration=selected_iteration), {
        "best_mse_iteration": best_iteration,
        "best_iteration": selected_iteration,
        "selected_iteration": selected_iteration,
        "selection_rule": "minimum_validation_mse_early_stopping",
        **signal,
    }


def train_xgboost(train, validation, test, features, target_col, params, paths, seed, device):
    from xgboost import XGBRegressor

    xtr, ytr, xva, yva, xte = _arrays(train, validation, test, features, target_col)
    kwargs = dict(params)
    model = XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", random_state=seed,
        n_jobs=-1, tree_method="hist", device="cuda" if device == "cuda" else "cpu", **kwargs,
    )
    model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    model.save_model(paths["model"])
    best_iteration = getattr(model, "best_iteration", None)
    prediction_kwargs = (
        {} if best_iteration is None
        else {"iteration_range": (0, int(best_iteration) + 1)}
    )
    validation_prediction = model.predict(xva, **prediction_kwargs)
    signal = _validation_signal_stats(validation, target_col, validation_prediction)
    return model.predict(xte, **prediction_kwargs), {
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "selection_rule": "validation_mse_primary_rank_ic_reported", **signal,
    }


def train_feedforward_nn(train, validation, test, features, target_col, params, paths, seed, device):
    import torch
    import torch.nn as nn

    set_seed(seed)
    xtr, ytr, xva, yva, xte = _arrays(train, validation, test, features, target_col)
    hidden = params["hidden_dims"]

    class FeedForwardNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            previous = len(features)
            for width in hidden:
                layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(params["dropout"])]
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x).squeeze(-1)

    torch_device = torch.device(device)
    model = FeedForwardNetwork().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    batch_size = int(params["batch_size"])
    # Core20 is small enough to copy each refit's complete matrices once. All
    # epoch shuffling and slicing then happens on-device with no batch transfer.
    xtr_tensor = torch.from_numpy(xtr).to(torch_device)
    ytr_tensor = torch.from_numpy(ytr.astype(np.float32)).to(torch_device)
    xva_tensor = torch.from_numpy(xva).to(torch_device)
    yva_tensor = torch.from_numpy(yva.astype(np.float32)).to(torch_device)
    amp_enabled = bool(params.get("mixed_precision", False) and torch_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch, best_loss, bad_epochs, best_epoch = 0, np.inf, 0, -1
    if paths["latest"].exists():
        checkpoint = torch.load(paths["latest"], map_location=torch_device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        print(f"    resuming at epoch {start_epoch + 1}")

    def epoch_loss(x_tensor, y_tensor, training):
        model.train(training)
        total_squared_error, count = 0.0, 0
        if training:
            order = torch.randperm(len(x_tensor), device=torch_device)
        else:
            order = None
        for start in range(0, len(x_tensor), batch_size):
            end = min(start + batch_size, len(x_tensor))
            if training:
                index = order[start:end]
                # BatchNorm cannot train on a singleton final batch.
                if len(index) == 1:
                    continue
                xb = x_tensor.index_select(0, index)
                yb = y_tensor.index_select(0, index)
            else:
                xb = x_tensor[start:end]
                yb = y_tensor[start:end]
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=torch_device.type, enabled=amp_enabled):
                prediction = model(xb)
                squared_error = (prediction - yb) ** 2
                loss = squared_error.mean()
                if training and params["l1_penalty"] > 0:
                    l1 = sum(
                        layer.weight.abs().sum()
                        for layer in model.modules()
                        if isinstance(layer, nn.Linear)
                    )
                    loss = loss + params["l1_penalty"] * l1
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_squared_error += float(squared_error.detach().sum().cpu())
            count += len(yb)
        return total_squared_error / max(count, 1)

    for epoch in range(start_epoch, params["max_epochs"]):
        epoch_start = time.perf_counter()
        train_loss = epoch_loss(xtr_tensor, ytr_tensor, True)
        with torch.inference_mode():
            validation_loss = epoch_loss(xva_tensor, yva_tensor, False)
        improved = validation_loss < best_loss - params["min_delta"]
        if improved:
            best_loss, bad_epochs, best_epoch = validation_loss, 0, epoch
            _torch_save_atomic(model.state_dict(), paths["best"])
        else:
            bad_epochs += 1
        _torch_save_atomic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
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
        raise RuntimeError("Feedforward training did not produce a valid best checkpoint.")
    model.load_state_dict(torch.load(paths["best"], map_location=torch_device))
    model.eval()
    with torch.inference_mode():
        validation_prediction = model(xva_tensor).float().cpu().numpy()
    signal = _validation_signal_stats(validation, target_col, validation_prediction)
    del xtr_tensor, ytr_tensor, xva_tensor, yva_tensor
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    xte_tensor = torch.from_numpy(xte).to(torch_device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(xte_tensor), batch_size):
            batch = xte_tensor[start:start + batch_size]
            with torch.amp.autocast(device_type=torch_device.type, enabled=amp_enabled):
                predictions.append(model(batch).float().cpu().numpy())
    _torch_save_atomic(model.state_dict(), paths["model"])
    return np.concatenate(predictions), {
        "best_validation_mse": float(best_loss),
        "best_epoch": int(best_epoch + 1),
        "batch_size": batch_size,
        "mixed_precision": amp_enabled,
        "device_resident_data": True,
        "selection_rule": "validation_mse_primary_rank_ic_reported",
        **signal,
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
                layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(params["dropout"])]
                previous = width
            layers.append(nn.Linear(previous, 1))
            self.return_predictor = nn.Sequential(*layers)

        def forward(self, x):
            if params.get("include_market_context", True):
                embedding = self.stock_encoder(x)
                n_stocks = embedding.shape[0]
                if n_stocks > 1:
                    # Each stock receives the mean learned representation of all
                    # other eligible stocks in the same month.
                    cloud = (embedding.sum(dim=0, keepdim=True) - embedding) / (n_stocks - 1)
                else:
                    cloud = torch.zeros_like(embedding)
            else:
                # Matched no-context ablation: identical prediction head and
                # training loop, but no information from other stocks.
                cloud = x.new_zeros((x.shape[0], params["embedding_dim"]))
            return self.return_predictor(torch.cat([x, cloud], dim=1)).squeeze(-1)

    return DeepSetCore()


def _torch_save_atomic(payload, path) -> None:
    """Keep the previous checkpoint valid if a runtime stops during saving."""
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_deepset(train, validation, test, features, target_col, params, paths, seed, device):
    """Train on complete monthly sets while masking labels only in the loss."""
    import torch

    set_seed(seed)
    torch_device = torch.device(device)
    model = build_deepset_core(len(features), params).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    def month_indices(frame):
        return [np.asarray(index, dtype=np.int64) for index in frame.groupby("eom", sort=True).indices.values()]

    train_months = month_indices(train)
    validation_months = month_indices(validation)
    test_months = month_indices(test)

    def month_tensors(frame, index):
        monthly = frame.iloc[index]
        x = torch.from_numpy(monthly[list(features)].to_numpy(np.float32)).to(torch_device)
        y_array = monthly[target_col].to_numpy(np.float32)
        mask_array = np.isfinite(y_array)
        y = torch.from_numpy(np.where(mask_array, y_array, 0.0)).to(torch_device)
        mask = torch.from_numpy(mask_array).to(torch_device)
        return x, y, mask

    def train_epoch(epoch):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_months))
        total_squared_error, total_count = 0.0, 0
        months_per_batch = params["months_per_batch"]

        for start in range(0, len(order), months_per_batch):
            optimizer.zero_grad(set_to_none=True)
            batch_squared_error = None
            batch_count = 0

            for month_number in order[start:start + months_per_batch]:
                x, y, mask = month_tensors(train, train_months[month_number])
                if not bool(mask.any()):
                    continue
                errors = (model(x)[mask] - y[mask]) ** 2
                month_squared_error = errors.sum()
                batch_squared_error = month_squared_error if batch_squared_error is None else batch_squared_error + month_squared_error
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
        total_squared_error, total_count = 0.0, 0
        with torch.no_grad():
            for index in months:
                x, y, mask = month_tensors(frame, index)
                if not bool(mask.any()):
                    continue
                errors = (model(x)[mask] - y[mask]) ** 2
                total_squared_error += float(errors.sum().cpu())
                total_count += int(mask.sum().item())
        return total_squared_error / max(total_count, 1)

    start_epoch, best_loss, bad_epochs, best_epoch = 0, np.inf, 0, -1
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
            best_loss, bad_epochs, best_epoch = validation_loss, 0, epoch
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
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
        raise RuntimeError("DeepSets training did not produce a valid best checkpoint.")
    model.load_state_dict(torch.load(paths["best"], map_location=torch_device))
    model.eval()

    validation_predictions = np.empty(len(validation), dtype=np.float32)
    predictions = np.empty(len(test), dtype=np.float32)
    with torch.no_grad():
        for index in validation_months:
            x, _, _ = month_tensors(validation, index)
            validation_predictions[index] = model(x).cpu().numpy()
        for index in test_months:
            x, _, _ = month_tensors(test, index)
            predictions[index] = model(x).cpu().numpy()

    _torch_save_atomic(model.state_dict(), paths["model"])
    return predictions, {
        "best_validation_mse": float(best_loss),
        "best_epoch": int(best_epoch + 1),
        "embedding_dim": int(params["embedding_dim"]),
        "uses_leave_one_out_cloud": bool(params.get("include_market_context", True)),
        "training_loss_weighting": "pooled_stock_mse",
        "selection_rule": "validation_mse_primary_rank_ic_reported",
        **_validation_signal_stats(validation, target_col, validation_predictions[validation[target_col].notna().to_numpy()]),
    }



def _component_paths(parent_dir: Path, component_id: str) -> dict[str, Path]:
    """Keep composite-model checkpoints isolated inside their own refit."""
    directory = parent_dir / "components" / component_id
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "dir": directory,
        "model": directory / "model.bin",
        "latest": directory / "latest.pt",
        "best": directory / "best.pt",
    }


def _convex_validation_weight(
    y_true, mlp_prediction, deepset_prediction, fallback_weight_b=0.5,
):
    """Return the MSE-minimizing DeepSets weight in a convex two-model blend."""
    y = np.asarray(y_true, dtype=np.float64)
    mlp = np.asarray(mlp_prediction, dtype=np.float64)
    deepset = np.asarray(deepset_prediction, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(mlp) & np.isfinite(deepset)
    if not valid.any():
        raise ValueError("No finite validation observations for hybrid weighting.")
    y, mlp, deepset = y[valid], mlp[valid], deepset[valid]
    difference = deepset - mlp
    denominator = float(np.dot(difference, difference))
    if denominator <= np.finfo(np.float64).eps * max(len(difference), 1):
        weight_deepset = float(fallback_weight_b)
        used_fallback = True
    else:
        unconstrained = float(np.dot(difference, y - mlp) / denominator)
        weight_deepset = float(np.clip(unconstrained, 0.0, 1.0))
        used_fallback = False
    return weight_deepset, int(valid.sum()), used_fallback


def _fit_hybrid_component(
    component_id, train, validation, prediction_panel, target_col, paths, seed, device,
):
    spec = MODEL_REGISTRY[component_id]
    component_features = MODEL_FEATURES.get(component_id, FEATURES_20)
    if spec.trainer_id == "lightgbm":
        trainer = train_lightgbm
    elif spec.trainer_id == "deepset":
        trainer = train_deepset
    else:
        raise ValueError(
            f"Hybrid component {component_id} uses unsupported trainer {spec.trainer_id}."
        )
    return trainer(
        train,
        validation,
        prediction_panel,
        component_features,
        target_col,
        spec.params,
        paths,
        seed,
        device,
    )


def train_strict_validation_hybrid(
    train, validation, test, features, target_col, params, paths, seed, device,
):
    """Train components on a three-year validation and blend on the fourth year."""
    del features
    validation_years = sorted(validation["eom"].dt.year.unique())
    base_years = int(params["base_validation_years"])
    weight_years = int(params["weight_validation_years"])
    if len(validation_years) != base_years + weight_years:
        raise ValueError(
            f"Expected {base_years + weight_years} validation years, "
            f"found {validation_years}."
        )
    model_validation_years = validation_years[:base_years]
    weight_validation_years = validation_years[base_years:]
    model_validation = validation.loc[
        validation["eom"].dt.year.isin(model_validation_years)
    ].copy()
    weight_validation = validation.loc[
        validation["eom"].dt.year.isin(weight_validation_years)
    ].copy()
    prediction_panel = pd.concat([weight_validation, test], ignore_index=True)
    n_weight = len(weight_validation)

    component_a = str(params["component_a_id"])
    component_b = str(params["component_b_id"])
    prediction_a, details_a = _fit_hybrid_component(
        component_a, train, model_validation, prediction_panel, target_col,
        _component_paths(paths["dir"], component_a), seed, device,
    )
    prediction_b, details_b = _fit_hybrid_component(
        component_b, train, model_validation, prediction_panel, target_col,
        _component_paths(paths["dir"], component_b), seed, device,
    )
    prediction_a = np.asarray(prediction_a, dtype=np.float64)
    prediction_b = np.asarray(prediction_b, dtype=np.float64)
    weight_b, n_observations, used_fallback = _convex_validation_weight(
        weight_validation[target_col].to_numpy(np.float64),
        prediction_a[:n_weight],
        prediction_b[:n_weight],
        fallback_weight_b=1.0 - float(params.get("fallback_weight_a", 0.5)),
    )
    weight_a = 1.0 - weight_b
    weighted_validation = (
        weight_a * prediction_a[:n_weight] + weight_b * prediction_b[:n_weight]
    )
    weighted_test = (
        weight_a * prediction_a[n_weight:] + weight_b * prediction_b[n_weight:]
    )
    fifty_fifty_test = 0.5 * (
        prediction_a[n_weight:] + prediction_b[n_weight:]
    )

    aligned = test[["eom", "id", "security_id"]].reset_index(drop=True).copy()
    aligned["component_a_id"] = component_a
    aligned["component_b_id"] = component_b
    aligned["component_a_pred"] = prediction_a[n_weight:]
    aligned["component_b_pred"] = prediction_b[n_weight:]
    aligned["fifty_fifty_pred"] = fifty_fifty_test
    aligned["validation_weighted_pred"] = weighted_test
    aligned_path = paths["dir"] / "aligned_component_predictions.parquet"
    temporary = aligned_path.with_suffix(".parquet.tmp")
    aligned.to_parquet(temporary, index=False)
    os.replace(temporary, aligned_path)

    valid_weight = weight_validation[target_col].notna().to_numpy()
    y_weight = weight_validation.loc[valid_weight, target_col].to_numpy(np.float64)
    validation_mse = {
        "component_a_validation_mse": float(np.mean(
            np.square(y_weight - prediction_a[:n_weight][valid_weight])
        )),
        "component_b_validation_mse": float(np.mean(
            np.square(y_weight - prediction_b[:n_weight][valid_weight])
        )),
        "fifty_fifty_validation_mse": float(np.mean(np.square(
            y_weight - 0.5 * (
                prediction_a[:n_weight][valid_weight]
                + prediction_b[:n_weight][valid_weight]
            )
        ))),
        "weighted_validation_mse": float(np.mean(np.square(
            y_weight - weighted_validation[valid_weight]
        ))),
    }
    with paths["model"].open("wb") as handle:
        pickle.dump({
            "component_a_id": component_a,
            "component_b_id": component_b,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "model_validation_years": model_validation_years,
            "weight_validation_years": weight_validation_years,
        }, handle)
    return weighted_test, {
        "component_a_id": component_a,
        "component_b_id": component_b,
        "model_validation_years": [int(year) for year in model_validation_years],
        "weight_validation_years": [int(year) for year in weight_validation_years],
        "weight_a": weight_a,
        "weight_b": weight_b,
        "weight_lgbm": weight_a if component_a.startswith("LGBM") else np.nan,
        "weight_deepset": weight_b if component_b.startswith("DEEPSET") else np.nan,
        "weight_observations": n_observations,
        "used_fallback_weight": used_fallback,
        "weight_selection_sample": "fourth_validation_year_only",
        "weight_objective": params["weight_objective"],
        "weight_constraint": params["weight_constraint"],
        "component_models_retrained": True,
        "component_a_fit": details_a,
        "component_b_fit": details_b,
        **validation_mse,
        **_validation_signal_stats(
            weight_validation.loc[valid_weight], target_col,
            weighted_validation[valid_weight],
        ),
    }




TRAINERS: dict[str, Callable] = {
    "lasso": train_lasso,
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "feedforward_nn": train_feedforward_nn,
    "deepset": train_deepset,
    "strict_validation_hybrid": train_strict_validation_hybrid,
}

