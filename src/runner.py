from __future__ import annotations

import json
import platform
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from .config import (
    CORE20,
    FEATURES_40_LAG1,
    FEATURES_40_LAG2,
    ExperimentConfig,
)
from .data import (
    build_oos_universe,
    load_and_prepare_panel,
    target_availability_summary,
)
from .evaluation import (
    form_equal_weight_deciles,
    merge_predictions,
    monthly_rank_ic,
    oos_r2,
    performance_stats,
)
from .model_comparison import (
    build_model_comparison_table,
)
from .models import (
    MODEL_FEATURES,
    MODEL_REGISTRY,
    TRAINERS,
    TRAINING_VERSION,
    _finite_target_mask,
    set_seed,
)
from .schedule import (
    make_rolling_schedule,
    year_slice,
)
from .self_checks import (
    run_framework_self_checks,
)


class ExperimentRunner:
    """
    Run the frozen rolling out-of-sample experiment.

    The experiment directory is treated as immutable with respect to its
    research specification. Existing complete model refits and evaluation
    outputs are reused. Missing or incomplete artifacts are recomputed.
    """

    def __init__(
        self,
        config: ExperimentConfig,
    ) -> None:
        config.validate()
        self.config = config

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _prediction_path(
        self,
        model_id: str,
    ) -> Path:
        return (
            self.config.run_dir
            / "predictions"
            / f"{model_id}.parquet"
        )

    def _metric_path(
        self,
        model_id: str,
    ) -> Path:
        return (
            self.config.run_dir
            / "metrics"
            / f"{model_id}.json"
        )

    def _rank_ic_path(
        self,
        model_id: str,
    ) -> Path:
        return (
            self.config.run_dir
            / "diagnostics"
            / f"{model_id}_monthly_rank_ic.parquet"
        )

    def _portfolio_path(
        self,
        model_id: str,
    ) -> Path:
        return (
            self.config.run_dir
            / "portfolios"
            / f"{model_id}_long_short.parquet"
        )

    def _refit_dir(
        self,
        model_id: str,
        test_year: int,
    ) -> Path:
        return (
            self.config.run_dir
            / "models"
            / model_id
            / f"refit_{int(test_year)}"
        )

    def _refit_paths(
        self,
        model_id: str,
        test_year: int,
    ) -> dict[str, Path]:
        directory = self._refit_dir(
            model_id,
            test_year,
        )

        return {
            "dir": directory,
            "model": directory / "model.bin",
            "best": directory / "best.pt",
            "latest": directory / "latest.pt",
            "predictions": (
                directory
                / "predictions.parquet"
            ),
            "metadata": (
                directory
                / "metadata.json"
            ),
        }

    # ------------------------------------------------------------------
    # Runtime / provenance
    # ------------------------------------------------------------------

    def _device(
        self,
    ) -> str:
        if not self.config.use_gpu:
            return "cpu"

        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "GPU execution was requested but "
                "PyTorch is not installed."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU execution was requested but "
                "CUDA is unavailable."
            )

        return "cuda"

    def _validate_or_write_manifest(
        self,
    ) -> None:
        path = (
            self.config.run_dir
            / "experiment_manifest.json"
        )

        expected = {
            **self.config.to_dict(),
            "training_version": TRAINING_VERSION,
            "data_size_bytes": int(
                Path(
                    self.config.data_path
                ).stat().st_size
            ),
        }

        def research_identity(
            payload: dict,
        ) -> dict:
            universe = payload.get(
                "universe",
                {},
            )
            windows = payload.get(
                "windows",
                {},
            )
            preprocessing = payload.get(
                "preprocessing",
                {},
            )
            portfolio = payload.get(
                "portfolio",
                {},
            )

            return {
                "training_version": payload.get(
                    "training_version"
                ),
                "data_size_bytes": payload.get(
                    "data_size_bytes"
                ),
                "feature_set_id": payload.get(
                    "feature_set_id"
                ),
                "target_col": payload.get(
                    "target_col"
                ),
                "seed": payload.get(
                    "seed"
                ),
                "universe": {
                    "country": universe.get(
                        "country"
                    ),
                    "start_year": universe.get(
                        "start_year"
                    ),
                    "end_year": universe.get(
                        "end_year"
                    ),
                    "allowed_size_groups": sorted(
                        universe.get(
                            "allowed_size_groups",
                            [],
                        )
                    ),
                    "security_id_col": universe.get(
                        "security_id_col"
                    ),
                },
                "windows": {
                    "train_years": windows.get(
                        "train_years"
                    ),
                    "validation_years": windows.get(
                        "validation_years"
                    ),
                    "test_years": windows.get(
                        "test_years"
                    ),
                },
                "preprocessing": {
                    "method": preprocessing.get(
                        "method"
                    ),
                    "missing_feature_fill": preprocessing.get(
                        "missing_feature_fill"
                    ),
                },
                "portfolio": {
                    "n_groups": portfolio.get(
                        "n_groups"
                    ),
                    "newey_west_lags": portfolio.get(
                        "newey_west_lags"
                    ),
                },
            }

        expected_identity = (
            research_identity(
                expected
            )
        )

        if path.exists():
            observed = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            observed_identity = (
                research_identity(
                    observed
                )
            )

            if (
                observed_identity
                != expected_identity
            ):
                differences = []

                for key in (
                    expected_identity
                ):
                    if (
                        observed_identity.get(
                            key
                        )
                        != expected_identity.get(
                            key
                        )
                    ):
                        differences.append(
                            key
                        )

                raise RuntimeError(
                    "Existing experiment manifest "
                    "differs in research-defining "
                    "settings: "
                    f"{differences}. "
                    "Use a new experiment_id only "
                    "if these settings were "
                    "intentionally changed."
                )

            return

        write_json_atomic(
            expected,
            path,
        )

    def _validate_or_write_runtime_versions(
        self,
    ) -> None:
        path = (
            self.config.run_dir
            / "runtime_versions.json"
        )

        packages = (
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "lightgbm",
            "xgboost",
            "torch",
        )

        resolved = {
            "python": (
                platform.python_version()
            ),
        }

        for package in packages:
            try:
                resolved[package] = (
                    version(package)
                )
            except PackageNotFoundError:
                resolved[package] = None

        try:
            import torch

            resolved[
                "cuda_available"
            ] = bool(
                torch.cuda.is_available()
            )

            resolved[
                "cuda_device"
            ] = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            )

        except ImportError:
            resolved[
                "cuda_available"
            ] = False
            resolved[
                "cuda_device"
            ] = None

        if path.exists():
            # Colab package patch versions can change between reconnects.
            # Completed artifacts remain authoritative for this frozen run;
            # runtime versions are provenance, not a cache invalidator.
            return

        write_json_atomic(
            resolved,
            path,
        )

    # ------------------------------------------------------------------
    # Model roster
    # ------------------------------------------------------------------

    def _validate_selected_models(
        self,
    ) -> None:
        for model_id in (
            self.config.selected_models
        ):
            if (
                model_id
                not in MODEL_REGISTRY
            ):
                raise ValueError(
                    f"Unknown model: {model_id}"
                )

            spec = MODEL_REGISTRY[
                model_id
            ]

            if (
                spec.feature_set_id
                != self.config.feature_set_id
            ):
                raise ValueError(
                    f"{model_id} expects "
                    f"{spec.feature_set_id}, "
                    f"not "
                    f"{self.config.feature_set_id}."
                )

            if (
                spec.trainer_id
                not in TRAINERS
            ):
                raise ValueError(
                    f"{model_id} references "
                    "unknown trainer "
                    f"{spec.trainer_id}."
                )

            if spec.data_layout not in {
                "flat",
                "monthly_panel",
            }:
                raise NotImplementedError(
                    "Unsupported data layout: "
                    f"{spec.data_layout}"
                )

    # ------------------------------------------------------------------
    # Cache validation
    # ------------------------------------------------------------------

    def _annual_refit_complete(
        self,
        model_id: str,
        test_year: int,
        refit_id: int,
    ) -> bool:
        paths = self._refit_paths(
            model_id,
            test_year,
        )

        required = (
            paths["model"],
            paths["predictions"],
            paths["metadata"],
        )

        if not all(
            path.exists()
            and path.stat().st_size > 0
            for path in required
        ):
            return False

        try:
            metadata = json.loads(
                paths[
                    "metadata"
                ].read_text(
                    encoding="utf-8"
                )
            )

            if (
                metadata.get(
                    "model_id"
                )
                != model_id
                or int(
                    metadata.get(
                        "test_year",
                        -1,
                    )
                )
                != int(test_year)
                or int(
                    metadata.get(
                        "refit_id",
                        -1,
                    )
                )
                != int(refit_id)
                or metadata.get(
                    "training_version"
                )
                != TRAINING_VERSION
                or metadata.get("features")
                != list(MODEL_FEATURES[model_id])
                or metadata.get("model_spec")
                != {
                    "trainer_id": MODEL_REGISTRY[model_id].trainer_id,
                    "data_layout": MODEL_REGISTRY[model_id].data_layout,
                    "params": MODEL_REGISTRY[model_id].params,
                }
            ):
                return False

            prediction = (
                pd.read_parquet(
                    paths[
                        "predictions"
                    ],
                    columns=[
                        "eom",
                        "security_id",
                        "y_pred",
                        "model_id",
                        "test_year",
                        "refit_id",
                    ],
                )
            )

            if prediction.empty:
                return False

            if prediction.duplicated(
                [
                    "eom",
                    "security_id",
                ]
            ).any():
                return False

            if not prediction[
                "model_id"
            ].eq(
                model_id
            ).all():
                return False

            if not prediction[
                "test_year"
            ].eq(
                int(test_year)
            ).all():
                return False

            if not prediction[
                "refit_id"
            ].eq(
                int(refit_id)
            ).all():
                return False

            return True

        except Exception:
            return False

    def _evaluation_exists(
        self,
        model_id: str,
    ) -> bool:
        return all(
            path.exists()
            for path in (
                self._metric_path(
                    model_id
                ),
                self._rank_ic_path(
                    model_id
                ),
                self._portfolio_path(
                    model_id
                ),
            )
        )

    def _evaluation_complete(
        self,
        model_id: str,
    ) -> bool:
        if not self._evaluation_exists(
            model_id
        ):
            return False

        try:
            metric = json.loads(
                self._metric_path(
                    model_id
                ).read_text(
                    encoding="utf-8"
                )
            )

            required_metrics = {
                "model_id",
                "pooled_oos_r2",
                "mean_monthly_rank_ic",
                "rank_ic_newey_west_t_stat",
                "rank_ic_n_months",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "portfolio_newey_west_t_stat",
                "max_drawdown",
            }

            if (
                metric.get(
                    "model_id"
                )
                != model_id
                or not (
                    required_metrics
                    .issubset(metric)
                )
            ):
                return False

            pd.read_parquet(
                self._rank_ic_path(
                    model_id
                ),
                columns=[
                    "eom",
                    "rank_ic",
                ],
            )

            pd.read_parquet(
                self._portfolio_path(
                    model_id
                ),
                columns=[
                    "eom",
                    "long_short_ret",
                ],
            )

            return True

        except Exception:
            return False

    def _pooled_prediction_complete(
        self,
        model_id: str,
    ) -> bool:
        path = self._prediction_path(
            model_id
        )

        if (
            not path.exists()
            or path.stat().st_size <= 0
        ):
            return False

        try:
            prediction = (
                pd.read_parquet(
                    path,
                    columns=[
                        "eom",
                        "target_date",
                        "id",
                        "security_id",
                        "country",
                        "y_true",
                        "y_pred",
                        "me",
                        "size_grp",
                        "target_available",
                        "model_id",
                        "test_year",
                        "refit_id",
                    ],
                )
            )

            if prediction.empty:
                return False

            if prediction.duplicated(
                [
                    "eom",
                    "security_id",
                ]
            ).any():
                return False

            if not prediction[
                "model_id"
            ].eq(
                model_id
            ).all():
                return False

            target_timing = (
                pd.to_datetime(
                    prediction[
                        "target_date"
                    ]
                )
                == (
                    pd.to_datetime(
                        prediction[
                            "eom"
                        ]
                    )
                    + pd.offsets.MonthEnd(
                        1
                    )
                )
            )

            return bool(
                target_timing.all()
            )

        except Exception:
            return False

    def _saved_schedule_complete(
        self,
        schedule: pd.DataFrame,
    ) -> bool:
        required = {
            "refit_id",
            "test_year",
            "train_start_year",
            "train_end_year",
            "validation_start_year",
            "validation_end_year",
            "test_start_year",
            "test_end_year",
        }

        if not required.issubset(
            schedule.columns
        ):
            return False

        expected_years = list(
            range(
                (
                    self.config.universe
                    .start_year
                    + self.config.windows
                    .train_years
                    + self.config.windows
                    .validation_years
                ),
                (
                    self.config.universe
                    .end_year
                    + 1
                ),
            )
        )

        observed_years = (
            schedule[
                "test_year"
            ]
            .astype(int)
            .tolist()
        )

        if (
            observed_years
            != expected_years
        ):
            return False

        train_length = (
            schedule[
                "train_end_year"
            ]
            - schedule[
                "train_start_year"
            ]
            + 1
        )

        validation_length = (
            schedule[
                "validation_end_year"
            ]
            - schedule[
                "validation_start_year"
            ]
            + 1
        )

        test_length = (
            schedule[
                "test_end_year"
            ]
            - schedule[
                "test_start_year"
            ]
            + 1
        )

        return bool(
            train_length.eq(
                self.config.windows
                .train_years
            ).all()
            and validation_length.eq(
                self.config.windows
                .validation_years
            ).all()
            and test_length.eq(
                self.config.windows
                .test_years
            ).all()
            and (
                schedule[
                    "train_end_year"
                ]
                + 1
                == schedule[
                    "validation_start_year"
                ]
            ).all()
            and (
                schedule[
                    "validation_end_year"
                ]
                + 1
                == schedule[
                    "test_start_year"
                ]
            ).all()
        )

    def _cached_model_complete(
        self,
        model_id: str,
        schedule: pd.DataFrame,
    ) -> bool:
        if (
            not self
            ._pooled_prediction_complete(
                model_id
            )
            or not self
            ._evaluation_complete(
                model_id
            )
        ):
            return False

        return all(
            self._annual_refit_complete(
                model_id,
                int(
                    row.test_year
                ),
                int(
                    row.refit_id
                ),
            )
            for row in (
                schedule.itertuples(
                    index=False
                )
            )
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _requested_features(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                feature
                for model_id
                in (
                    self.config
                    .selected_models
                )
                for feature
                in MODEL_FEATURES.get(
                    model_id,
                    CORE20,
                )
            )
        )

    @staticmethod
    def _raw_features(
        requested_features: tuple[
            str,
            ...,
        ],
    ) -> tuple[str, ...]:
        suffixes = (
            "_lag1",
            "_lag2",
            "_velocity1",
            "_lag1_available",
            "_lag2_available",
        )

        return tuple(
            feature
            for feature
            in requested_features
            if not feature.endswith(
                suffixes
            )
        )

    def _prepare_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        requested_features = (
            self._requested_features()
        )

        raw_features = (
            self._raw_features(
                requested_features
            )
        )

        include_lag1 = bool(
            set(
                requested_features
            )
            & set(
                FEATURES_40_LAG1
            )
        )

        include_lag2 = bool(
            set(
                requested_features
            )
            & set(
                FEATURES_40_LAG2
            )
        )

        panel, audit = (
            load_and_prepare_panel(
                self.config,
                raw_features,
                include_feature40_lag1=(
                    include_lag1
                ),
                include_feature40_lag2=(
                    include_lag2
                ),
            )
        )

        write_json_atomic(
            audit,
            (
                self.config.run_dir
                / "data_audit.json"
            ),
        )

        write_csv_atomic(
            target_availability_summary(
                panel
            ),
            (
                self.config.run_dir
                / "target_availability_summary.csv"
            ),
        )

        schedule = (
            make_rolling_schedule(
                panel,
                self.config.windows,
            )
        )

        write_parquet_atomic(
            schedule,
            (
                self.config.run_dir
                / "rolling_schedule.parquet"
            ),
        )

        universe = (
            build_oos_universe(
                panel,
                schedule,
                self.config.target_col,
            )
        )

        write_parquet_atomic(
            universe,
            (
                self.config.run_dir
                / "oos_universe.parquet"
            ),
        )

        return (
            panel,
            schedule,
            universe,
        )

    # ------------------------------------------------------------------
    # Annual model fitting
    # ------------------------------------------------------------------

    def _fit_or_load_refits(
        self,
        model_id: str,
        panel: pd.DataFrame,
        schedule: pd.DataFrame,
        device: str,
    ) -> pd.DataFrame:
        spec = MODEL_REGISTRY[
            model_id
        ]

        model_features = (
            MODEL_FEATURES.get(
                model_id,
                CORE20,
            )
        )

        trainer = TRAINERS[
            spec.trainer_id
        ]

        all_predictions: list[
            pd.DataFrame
        ] = []

        n_refits = len(
            schedule
        )

        for refit_number, row in (
            enumerate(
                schedule.itertuples(
                    index=False
                ),
                start=1,
            )
        ):
            test_year = int(
                row.test_year
            )

            refit_id = int(
                row.refit_id
            )

            label = (
                f"  refit "
                f"{refit_number:02d}/"
                f"{n_refits:02d} | "
                f"test_year={test_year}"
            )

            paths = (
                self._refit_paths(
                    model_id,
                    test_year,
                )
            )

            if (
                self
                ._annual_refit_complete(
                    model_id,
                    test_year,
                    refit_id,
                )
            ):
                print(
                    f"{label} | "
                    "status=loading"
                )

                all_predictions.append(
                    pd.read_parquet(
                        paths[
                            "predictions"
                        ]
                    )
                )

                continue

            paths[
                "dir"
            ].mkdir(
                parents=True,
                exist_ok=True,
            )

            train = year_slice(
                panel,
                int(
                    row.train_start_year
                ),
                int(
                    row.train_end_year
                ),
            )

            validation = year_slice(
                panel,
                int(
                    row.validation_start_year
                ),
                int(
                    row.validation_end_year
                ),
            )

            test = year_slice(
                panel,
                int(
                    row.test_start_year
                ),
                int(
                    row.test_end_year
                ),
            )

            train_n = int(
                _finite_target_mask(
                    train,
                    self.config.target_col,
                ).sum()
            )

            validation_n = int(
                _finite_target_mask(
                    validation,
                    self.config.target_col,
                ).sum()
            )

            if (
                train_n == 0
                or validation_n == 0
                or test.empty
            ):
                raise ValueError(
                    "Empty usable split for "
                    f"{model_id}, "
                    f"test year "
                    f"{test_year}."
                )

            print(
                f"{label} | "
                "status=training | "
                f"train_n="
                f"{train_n:,} | "
                f"validation_n="
                f"{validation_n:,} | "
                f"test_n="
                f"{len(test):,}"
            )

            prediction, fit_details = (
                trainer(
                    train,
                    validation,
                    test,
                    model_features,
                    self.config.target_col,
                    spec.params,
                    paths,
                    self.config.seed,
                    device,
                )
            )

            prediction = np.asarray(
                prediction,
                dtype=np.float64,
            )

            if (
                len(prediction)
                != len(test)
            ):
                raise ValueError(
                    f"{model_id} returned "
                    f"{len(prediction):,} "
                    "predictions for "
                    f"{len(test):,} "
                    "test rows."
                )

            output = test[
                [
                    "eom",
                    "id",
                    "security_id",
                ]
            ].copy()

            output[
                "y_pred"
            ] = prediction

            output[
                "model_id"
            ] = model_id

            output[
                "test_year"
            ] = test_year

            output[
                "refit_id"
            ] = refit_id

            metadata = {
                "model_id": model_id,
                "training_version": (
                    TRAINING_VERSION
                ),
                "test_year": (
                    test_year
                ),
                "refit_id": (
                    refit_id
                ),
                "schedule": (
                    row._asdict()
                ),
                "features": list(
                    model_features
                ),
                "model_spec": {
                    "trainer_id": (
                        spec.trainer_id
                    ),
                    "data_layout": (
                        spec.data_layout
                    ),
                    "params": (
                        spec.params
                    ),
                },
                "fit_details": (
                    fit_details
                ),
            }

            write_parquet_atomic(
                output,
                paths[
                    "predictions"
                ],
            )

            write_json_atomic(
                metadata,
                paths[
                    "metadata"
                ],
            )

            if not paths[
                "model"
            ].exists():
                raise RuntimeError(
                    f"{model_id}, "
                    f"test year "
                    f"{test_year}: trainer "
                    "did not save the fitted "
                    "model artifact."
                )

            all_predictions.append(
                output
            )

            print(
                f"{label} | "
                "status=saved | "
                f"n_predictions="
                f"{len(output):,}"
            )

        pooled = (
            pd.concat(
                all_predictions,
                ignore_index=True,
            )
            .sort_values(
                [
                    "eom",
                    "security_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        pooled[
            "model_id"
        ] = model_id

        return pooled

    # ------------------------------------------------------------------
    # Pooled prediction handling
    # ------------------------------------------------------------------

    def _load_or_build_pooled(
        self,
        model_id: str,
        panel: pd.DataFrame,
        schedule: pd.DataFrame,
        device: str,
    ) -> pd.DataFrame:
        path = self._prediction_path(
            model_id
        )

        annual_refits_complete = all(
            self._annual_refit_complete(
                model_id, int(row.test_year), int(row.refit_id)
            )
            for row in schedule.itertuples(index=False)
        )

        if self._pooled_prediction_complete(model_id) and annual_refits_complete:
            print(
                f"{model_id}: "
                "pooled predictions found"
            )

            return pd.read_parquet(
                path
            )

        if self._pooled_prediction_complete(model_id):
            print(f"{model_id}: repairing missing or incomplete annual refits")

        refit_predictions = (
            self._fit_or_load_refits(
                model_id,
                panel,
                schedule,
                device,
            )
        )

        return refit_predictions

    # ------------------------------------------------------------------
    # Common evaluation
    # ------------------------------------------------------------------

    def _evaluate_model(
        self,
        model_id: str,
        pooled: pd.DataFrame,
        universe: pd.DataFrame,
    ) -> None:
        evaluated = merge_predictions(
            pooled,
            universe,
        )

        pooled_columns = [
            "eom",
            "target_date",
            "id",
            "security_id",
            "country",
            "y_true",
            "y_pred",
            "me",
            "size_grp",
            "target_available",
            "model_id",
            "test_year",
            "refit_id",
        ]

        write_parquet_atomic(
            evaluated[
                pooled_columns
            ],
            self._prediction_path(
                model_id
            ),
        )

        pooled_r2 = oos_r2(
            evaluated
        )

        monthly_ic, rank_stats = (
            monthly_rank_ic(
                evaluated,
                (
                    self.config
                    .portfolio
                    .newey_west_lags
                ),
            )
        )

        portfolio = (
            form_equal_weight_deciles(
                evaluated,
                (
                    self.config
                    .portfolio
                    .n_groups
                ),
            )
        )

        performance = (
            performance_stats(
                portfolio[
                    "long_short_ret"
                ],
                (
                    self.config
                    .portfolio
                    .newey_west_lags
                ),
            )
        )

        write_parquet_atomic(
            monthly_ic,
            self._rank_ic_path(
                model_id
            ),
        )

        write_parquet_atomic(
            portfolio,
            self._portfolio_path(
                model_id
            ),
        )

        metric = {
            "model_id": (
                model_id
            ),
            "pooled_oos_r2": (
                pooled_r2
            ),
            **rank_stats,
            **performance,
        }

        write_json_atomic(
            metric,
            self._metric_path(
                model_id
            ),
        )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def _comparison_table(
        self,
    ) -> pd.DataFrame:
        return (
            build_model_comparison_table(
                self.config.run_dir
            )
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare the frozen panel once for staged execution in one runtime."""
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self._validate_selected_models()
        self._validate_or_write_manifest()
        return self._prepare_data()

    def run(
        self,
        prepared_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        run_framework_self_checks()

        self.config.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._validate_selected_models()
        self._validate_or_write_manifest()

        schedule_path = (
            self.config.run_dir
            / "rolling_schedule.parquet"
        )

        saved_schedule = (
            pd.read_parquet(
                schedule_path
            )
            if schedule_path.exists()
            else None
        )

        everything_done = (
            saved_schedule is not None
            and self._saved_schedule_complete(
                saved_schedule
            )
            and all(
                self._cached_model_complete(
                    model_id,
                    saved_schedule,
                )
                for model_id
                in (
                    self.config
                    .selected_models
                )
            )
        )

        if everything_done:
            for model_id in (
                self.config.selected_models
            ):
                print(
                    f"{model_id}: "
                    "complete saved outputs; "
                    "skipping"
                )

            return (
                self._comparison_table()
            )

        self._validate_or_write_runtime_versions()

        set_seed(
            self.config.seed
        )

        device = self._device()

        print(
            f"Device: {device}"
        )

        panel, schedule, universe = (
            prepared_data if prepared_data is not None else self._prepare_data()
        )

        for model_id in (
            self.config.selected_models
        ):
            if (
                self._cached_model_complete(
                    model_id,
                    schedule,
                )
            ):
                print(
                    f"{model_id}: "
                    "complete saved outputs; "
                    "skipping"
                )

                continue

            pooled = (
                self._load_or_build_pooled(
                    model_id,
                    panel,
                    schedule,
                    device,
                )
            )

            if (
                self._evaluation_complete(
                    model_id
                )
                and (
                    self
                    ._pooled_prediction_complete(
                        model_id
                    )
                )
            ):
                print(
                    f"{model_id}: "
                    "saved evaluation found; "
                    "skipping evaluation"
                )
                continue

            self._evaluate_model(
                model_id,
                pooled,
                universe,
            )

        return (
            self._comparison_table()
        )
