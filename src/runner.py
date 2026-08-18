from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import ArtifactStore, stable_hash, write_json_atomic, write_parquet_atomic
from .config import (
    CORE20,
    CORE20_LAG1,
    CORE20_LAG2,
    CORE20_VELOCITY,
    FEATURES_40_LAG1,
    FEATURES_40_LAG2,
    ExperimentConfig,
)
from .data import build_oos_universe, load_and_prepare_panel, target_availability_summary
from .evaluation import (
    decile_summary,
    form_equal_weight_deciles,
    form_portfolio_variants,
    merge_predictions,
    monthly_mechanism_diagnostics,
    monthly_rank_ic,
    monthly_signal_diagnostics,
    oos_r2,
    performance_stats,
    portfolio_variant_stats,
    robust_oos_r2,
)
from .models import (
    MIGRATED_MODEL_SIGNATURES,
    MODEL_DEPENDENCIES,
    MODEL_FEATURES,
    MODEL_REGISTRY,
    TRAINERS,
    set_seed,
)
from .schedule import make_rolling_schedule, year_slice


class ExperimentRunner:
    DIAGNOSTICS_VERSION = "post_model_v5_missing_return_stress"

    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config
        self.store = ArtifactStore(config.run_dir)

    def _device(self) -> str:
        if not self.config.use_gpu:
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _model_signature(self, model_id: str) -> str:
        if model_id in MIGRATED_MODEL_SIGNATURES:
            return MIGRATED_MODEL_SIGNATURES[model_id]
        spec = MODEL_REGISTRY[model_id]
        model_features = MODEL_FEATURES.get(model_id, CORE20)
        payload = {
            "pipeline_version": self.config.pipeline_version,
            "feature_set_id": self.config.feature_set_id,
            "features": model_features,
            "target": self.config.target_col,
            "universe": self.config.universe,
            "windows": self.config.windows,
            "preprocessing": self.config.preprocessing,
            "seed": self.config.seed,
            "model": spec,
        }
        dependencies = MODEL_DEPENDENCIES.get(model_id, ())
        if dependencies:
            payload["dependency_signatures"] = {
                dependency: self._model_signature(dependency)
                for dependency in dependencies
            }
        return stable_hash(payload)

    def _write_manifest(self) -> None:
        manifest_path = self.config.run_dir / "experiment_manifest.json"
        payload = self.config.to_dict()
        # These settings control where/how code executes, not the research
        # experiment. Excluding them lets the same artifacts move safely
        # between Windows and Colab and between CPU and GPU runtimes.
        execution_fields = {
            "selected_models", "project_dir", "data_path", "output_dir",
            "use_gpu", "config_hash",
        }

        def experiment_identity(data):
            return {key: value for key, value in data.items() if key not in execution_fields}

        identity = experiment_identity(payload)
        payload.pop("selected_models", None)
        payload["config_hash"] = stable_hash(identity)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_identity = experiment_identity(existing)
            if stable_hash(existing_identity) != payload["config_hash"]:
                differing = sorted(
                    key for key in set(identity) | set(existing_identity)
                    if identity.get(key) != existing_identity.get(key)
                )
                raise ValueError(
                    "This experiment_id already exists with different immutable "
                    f"settings ({differing}). Choose a new experiment_id."
                )
            # Upgrade an older manifest and record the paths/device used by
            # the current runtime without invalidating completed model work.
            if existing != payload:
                write_json_atomic(payload, manifest_path)
        else:
            write_json_atomic(payload, manifest_path)

    def _write_runtime_versions(self) -> None:
        packages = (
            "numpy", "pandas", "pyarrow", "scikit-learn",
            "lightgbm", "xgboost", "torch",
        )
        resolved = {"python": platform.python_version()}
        for package in packages:
            try:
                resolved[package] = version(package)
            except PackageNotFoundError:
                resolved[package] = None
        write_json_atomic(resolved, self.config.run_dir / "runtime_versions.json")

    def _cumulative_comparison(self) -> pd.DataFrame:
        rows = []
        metrics_dir = self.config.run_dir / "metrics"
        for path in sorted(metrics_dir.glob("*.json")):
            metric = json.loads(path.read_text(encoding="utf-8"))
            # The filename is the canonical identity. This also lets renamed
            # historical artifacts retain their original immutable payload.
            model_id = path.stem
            if model_id not in MODEL_REGISTRY:
                continue
            prediction_path = self.config.run_dir / "predictions" / f"{model_id}.parquet"
            if not prediction_path.exists():
                continue
            predictions = pd.read_parquet(
                prediction_path, columns=["model_signature", "y_pred"]
            )
            signatures = predictions["model_signature"].dropna().unique()
            if len(signatures) != 1:
                raise ValueError(
                    f"{prediction_path} must contain exactly one model signature."
                )
            rows.append({
                "model_id": model_id,
                "model_signature": str(signatures[0]),
                "pooled_oos_r2": metric.get("pooled_oos_r2"),
                "n_predictions": int(len(predictions)),
                **{key: value for key, value in metric.items()
                   if key not in {"model_id", "pooled_oos_r2"}},
            })
        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("pooled_oos_r2", ascending=False)
        result.to_csv(self.config.run_dir / "model_comparison.csv", index=False)
        return result

    def _diagnostics_current(self, model_id: str, signature: str) -> bool:
        """Return True only when pooled predictions and every diagnostic are current."""
        prediction_path = self.config.run_dir / "predictions" / f"{model_id}.parquet"
        metric_path = self.config.run_dir / "metrics" / f"{model_id}.json"
        required = (
            prediction_path,
            metric_path,
            self.config.run_dir / "diagnostics" / f"{model_id}_monthly_rank_ic.parquet",
            self.config.run_dir / "diagnostics" / f"{model_id}_monthly_mechanisms.parquet",
            self.config.run_dir / "diagnostics" / f"{model_id}_monthly_signal.parquet",
            self.config.run_dir / "portfolios" / f"{model_id}_deciles.parquet",
            self.config.run_dir / "portfolios" / f"{model_id}_variants.parquet",
        )
        if not all(path.exists() for path in required):
            return False
        try:
            import pyarrow.parquet as pq

            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            if metric.get("diagnostics_version") != self.DIAGNOSTICS_VERSION:
                return False
            if metric.get("model_signature") != signature:
                return False
            required_prediction_columns = {
                "eom", "id", "country", "y_true", "y_pred", "me",
                "model_id", "model_signature", "test_year", "refit_id",
            }
            if not required_prediction_columns.issubset(
                set(pq.read_schema(prediction_path).names)
            ):
                return False
            saved_ids = pd.read_parquet(
                prediction_path, columns=["model_id"]
            )["model_id"].dropna().unique()
            if len(saved_ids) != 1 or str(saved_ids[0]) != model_id:
                return False
            expected_months = 12 * (
                self.config.universe.end_year
                - self.config.universe.start_year
                - self.config.windows.train_years
                - self.config.windows.validation_years
                + 1
            )
            signal = pd.read_parquet(
                self.config.run_dir / "diagnostics" / f"{model_id}_monthly_signal.parquet",
                columns=["eom", "signal_available"],
            )
            if signal["eom"].nunique() != expected_months or len(signal) != expected_months:
                return False
            if metric.get("n_signal_months", 0) + metric.get("n_no_signal_months", 0) != expected_months:
                return False
            variant_columns = set(pq.read_schema(
                self.config.run_dir / "portfolios" / f"{model_id}_variants.parquet"
            ).names)
            if not {
                "missing_return_stress_ret", "n_long_assigned", "n_short_assigned"
            }.issubset(variant_columns):
                return False
            if "tail_10pct_missing_return_stress_annualized_return" not in metric:
                return False
            signatures = pd.read_parquet(
                prediction_path, columns=["model_signature"]
            )["model_signature"].dropna().unique()
            return len(signatures) == 1 and str(signatures[0]) == signature
        except Exception:
            return False

    def run(self) -> pd.DataFrame:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()
        self._write_runtime_versions()
        set_seed(self.config.seed)
        device = self._device()
        print(f"Device: {device}")

        for model_id in self.config.selected_models:
            if model_id not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_id}")
        current_models = [
            model_id for model_id in self.config.selected_models
            if self._diagnostics_current(model_id, self._model_signature(model_id))
        ]
        if len(current_models) == len(self.config.selected_models):
            for model_id in current_models:
                print(
                    f"{model_id} [{self._model_signature(model_id)}]: "
                    "current diagnostics; skipping"
                )
            return self._cumulative_comparison()

        requested_features = tuple(dict.fromkeys(
            feature
            for model_id in self.config.selected_models
            for feature in MODEL_FEATURES.get(model_id, CORE20)
        ))
        selected_feature_union = tuple(
            feature for feature in requested_features
            if not feature.endswith("_lag1")
            and not feature.endswith("_lag2")
            and not feature.endswith("_velocity1")
            and not feature.endswith("_lag1_available")
            and not feature.endswith("_lag2_available")
        )
        panel, audit = load_and_prepare_panel(
            self.config,
            selected_feature_union,
            include_core_dynamics=bool(
                set(requested_features) & (set(CORE20_LAG1) | set(CORE20_VELOCITY))
            ),
            include_feature40_lag1=bool(
                set(requested_features) & set(FEATURES_40_LAG1)
            ),
            include_core_lag2=bool(
                set(requested_features) & set(CORE20_LAG2)
            ),
            include_feature40_lag2=bool(
                set(requested_features) & set(FEATURES_40_LAG2)
            ),
        )
        write_json_atomic(audit, self.config.run_dir / "data_audit.json")
        target_availability_summary(panel).to_csv(
            self.config.run_dir / "target_availability_summary.csv", index=False
        )
        schedule = make_rolling_schedule(panel, self.config.windows)
        write_parquet_atomic(schedule, self.config.run_dir / "rolling_schedule.parquet")

        universe = build_oos_universe(panel, schedule, self.config.target_col)
        write_parquet_atomic(universe, self.config.run_dir / "oos_universe.parquet")
        comparison = []

        for model_id in self.config.selected_models:
            if model_id not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_id}")
            spec = MODEL_REGISTRY[model_id]
            model_features = MODEL_FEATURES.get(model_id, CORE20)
            if spec.feature_set_id != self.config.feature_set_id:
                raise ValueError(f"{model_id} expects {spec.feature_set_id}, not {self.config.feature_set_id}")
            if spec.data_layout not in {"flat", "monthly_panel"}:
                raise NotImplementedError(f"Data layout {spec.data_layout} is not implemented yet.")

            signature = self._model_signature(model_id)
            if self._diagnostics_current(model_id, signature):
                print(f"\n{model_id} [{signature}]: current diagnostics; skipping")
                continue
            trainer = TRAINERS[spec.trainer_id]
            refit_predictions = []
            print(f"\n{model_id} [{signature}]")

            n_refits = len(schedule)
            for refit_number, row in enumerate(schedule.itertuples(index=False), start=1):
                refit_label = (
                    f"  refit {refit_number:02d}/{n_refits:02d} | "
                    f"test_year={row.test_year}"
                )
                # Completion and dependency checks must be read-only. Create a
                # new refit directory only after every prerequisite is valid.
                paths = self.store.paths(model_id, signature, row.test_year, create=False)
                refit_complete = self.store.is_complete(paths, signature)
                if refit_complete:
                    print(f"{refit_label} | status=loading")
                    refit_predictions.append(pd.read_parquet(paths["predictions"]))
                    continue

                dependency_metadata = {}
                dependency_paths = {}
                for dependency_id in MODEL_DEPENDENCIES.get(model_id, ()):
                    dependency_signature = self._model_signature(dependency_id)
                    dependency_refit_paths = self.store.paths(
                        dependency_id, dependency_signature, row.test_year, create=False
                    )
                    if not self.store.is_complete(dependency_refit_paths, dependency_signature):
                        raise FileNotFoundError(
                            f"{model_id} requires completed {dependency_id} artifacts "
                            f"for test year {row.test_year} and signature "
                            f"{dependency_signature}. Run {dependency_id} first."
                        )
                    if not dependency_refit_paths["model"].exists():
                        raise FileNotFoundError(
                            f"Missing saved model weights: {dependency_refit_paths['model']}"
                        )
                    dependency_paths[dependency_id] = {
                        **dependency_refit_paths,
                        "signature": dependency_signature,
                    }
                    dependency_metadata[dependency_id] = dependency_signature
                paths = self.store.paths(model_id, signature, row.test_year, create=True)
                paths["dependencies"] = dependency_paths

                train = year_slice(panel, row.train_start_year, row.train_end_year)
                validation = year_slice(panel, row.validation_start_year, row.validation_end_year)
                test = year_slice(panel, row.test_start_year, row.test_end_year)
                if train[self.config.target_col].notna().sum() == 0 or validation[self.config.target_col].notna().sum() == 0 or test.empty:
                    raise ValueError(f"Empty usable split for {model_id}, test year {row.test_year}")

                print(
                    f"{refit_label} | status=training | "
                    f"train_n={train[self.config.target_col].notna().sum():,} | "
                    f"validation_n={validation[self.config.target_col].notna().sum():,} | "
                    f"test_n={len(test):,}"
                )

                prediction, fit_details = trainer(
                    train, validation, test, model_features, self.config.target_col,
                    spec.params, paths, self.config.seed, device,
                )
                output = test[["eom", "id", "permno"]].copy()
                output["y_pred"] = np.asarray(prediction, dtype=np.float64)
                output["model_id"] = model_id
                output["test_year"] = int(row.test_year)
                output["refit_id"] = int(row.refit_id)
                output["model_signature"] = signature
                finite_prediction = output["y_pred"].replace([np.inf, -np.inf], np.nan).dropna()
                monthly_unique = output.assign(
                    y_pred=output["y_pred"].replace([np.inf, -np.inf], np.nan)
                ).groupby("eom")["y_pred"].nunique(dropna=True)
                fit_details = {
                    **fit_details,
                    "test_unique_predictions": int(finite_prediction.nunique()),
                    "test_no_signal_months": int((monthly_unique < 2).sum()),
                    "test_signal_months": int((monthly_unique >= 2).sum()),
                    "collapsed_test_refit": bool((monthly_unique < 2).all()),
                }

                metadata = {
                    "model_id": model_id,
                    "model_signature": signature,
                    "test_year": int(row.test_year),
                    "refit_id": int(row.refit_id),
                    "schedule": row._asdict(),
                    "features": list(model_features),
                    "model_spec": {"trainer_id": spec.trainer_id, "data_layout": spec.data_layout, "params": spec.params},
                    "dependencies": dependency_metadata,
                    "fit_details": fit_details,
                }
                write_parquet_atomic(output, paths["predictions"])
                write_json_atomic(metadata, paths["metadata"])
                self.store.mark_complete(paths)
                refit_predictions.append(output)
                print(
                    f"{refit_label} | status=saved | "
                    f"n_predictions={len(output):,}"
                )

            pooled = pd.concat(refit_predictions, ignore_index=True).sort_values(["eom", "permno"])
            # Completed refits may have been migrated from an earlier display
            # ID. Canonicalize only the label; predictions and signatures stay
            # byte-for-byte tied to the original fitted model.
            pooled["model_id"] = model_id
            evaluated = merge_predictions(pooled, universe)
            pooled_columns = [
                "eom", "target_date", "id", "permno", "country", "y_true",
                "y_pred", "me", "size_grp", "target_available", "model_id",
                "model_signature", "test_year", "refit_id",
            ]
            pooled_path = self.config.run_dir / "predictions" / f"{model_id}.parquet"
            write_parquet_atomic(evaluated[pooled_columns], pooled_path)
            r2 = oos_r2(evaluated)
            robust_r2 = robust_oos_r2(evaluated)
            monthly_ic, rank_stats = monthly_rank_ic(
                evaluated, self.config.portfolio.newey_west_lags
            )
            monthly_mechanisms, mechanism_stats = monthly_mechanism_diagnostics(
                evaluated, self.config.portfolio.n_groups
            )
            monthly_signal, signal_stats = monthly_signal_diagnostics(evaluated)
            deciles = form_equal_weight_deciles(evaluated, self.config.portfolio.n_groups)
            stats = performance_stats(
                deciles["long_short_ret"] if "long_short_ret" in deciles else pd.Series(dtype=float),
                self.config.portfolio.newey_west_lags,
            )
            decile_stats = decile_summary(deciles, self.config.portfolio.n_groups)
            variants = form_portfolio_variants(evaluated)
            variant_stats = portfolio_variant_stats(
                variants, self.config.portfolio.newey_west_lags
            )
            write_parquet_atomic(
                monthly_ic,
                self.config.run_dir / "diagnostics" / f"{model_id}_monthly_rank_ic.parquet",
            )
            write_parquet_atomic(
                monthly_mechanisms,
                self.config.run_dir / "diagnostics" / f"{model_id}_monthly_mechanisms.parquet",
            )
            write_parquet_atomic(
                monthly_signal,
                self.config.run_dir / "diagnostics" / f"{model_id}_monthly_signal.parquet",
            )
            write_parquet_atomic(deciles, self.config.run_dir / "portfolios" / f"{model_id}_deciles.parquet")
            write_parquet_atomic(
                variants,
                self.config.run_dir / "portfolios" / f"{model_id}_variants.parquet",
            )
            write_json_atomic(
                {
                    "model_id": model_id, "model_signature": signature,
                    "diagnostics_version": self.DIAGNOSTICS_VERSION,
                    "pooled_oos_r2": r2, "robust_oos_r2": robust_r2,
                    **rank_stats, **mechanism_stats, **signal_stats, **decile_stats,
                    **variant_stats, **stats,
                },
                self.config.run_dir / "metrics" / f"{model_id}.json",
            )
            comparison.append({
                "model_id": model_id, "model_signature": signature,
                "pooled_oos_r2": r2, "robust_oos_r2": robust_r2,
                "n_predictions": len(pooled), **rank_stats, **mechanism_stats, **signal_stats,
                **decile_stats, **variant_stats, **stats,
            })

        return self._cumulative_comparison()
