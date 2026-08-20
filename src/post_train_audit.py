from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json_atomic
from .config import ExperimentConfig, FEATURES_40, FINAL_MODEL_ROSTER


REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
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
)

REQUIRED_COMMON_OUTPUTS: tuple[str, ...] = (
    "prediction",
    "metric",
    "rank_ic",
    "long_short",
)


def _record(
    rows: list[dict],
    *,
    area: str,
    check: str,
    passed: bool,
    detail: str,
) -> None:
    rows.append(
        {
            "area": area,
            "check": check,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def _expected_test_years(
    config: ExperimentConfig,
) -> list[int]:
    first_test_year = (
        config.universe.start_year
        + config.windows.train_years
        + config.windows.validation_years
    )

    return list(
        range(
            first_test_year,
            config.universe.end_year + 1,
        )
    )


def _common_paths(
    run_dir: Path,
    model_id: str,
) -> dict[str, Path]:
    return {
        "prediction": (
            run_dir
            / "predictions"
            / f"{model_id}.parquet"
        ),
        "metric": (
            run_dir
            / "metrics"
            / f"{model_id}.json"
        ),
        "rank_ic": (
            run_dir
            / "diagnostics"
            / f"{model_id}_monthly_rank_ic.parquet"
        ),
        "long_short": (
            run_dir
            / "portfolios"
            / f"{model_id}_long_short.parquet"
        ),
    }


def _audit_schedule(
    rows: list[dict],
    config: ExperimentConfig,
) -> None:
    path = (
        config.run_dir
        / "rolling_schedule.parquet"
    )

    if not path.exists():
        _record(
            rows,
            area="schedule",
            check="rolling_schedule_exists",
            passed=False,
            detail=str(path),
        )
        return

    schedule = pd.read_parquet(path)

    required_columns = {
        "refit_id",
        "test_year",
        "train_start_year",
        "train_end_year",
        "validation_start_year",
        "validation_end_year",
        "test_start_year",
        "test_end_year",
    }

    schema_ok = required_columns.issubset(
        schedule.columns
    )

    _record(
        rows,
        area="schedule",
        check="schedule_schema",
        passed=schema_ok,
        detail=(
            "ok"
            if schema_ok
            else (
                "missing="
                f"{sorted(required_columns - set(schedule.columns))}"
            )
        ),
    )

    if not schema_ok:
        return

    expected_years = (
        _expected_test_years(config)
    )

    observed_years = (
        schedule["test_year"]
        .astype(int)
        .tolist()
    )

    _record(
        rows,
        area="schedule",
        check="complete_test_years",
        passed=(
            observed_years
            == expected_years
        ),
        detail=(
            f"observed={observed_years[0] if observed_years else None}"
            f"-{observed_years[-1] if observed_years else None}; "
            f"expected={expected_years[0]}"
            f"-{expected_years[-1]}"
        ),
    )

    train_length = (
        schedule["train_end_year"]
        - schedule["train_start_year"]
        + 1
    )

    validation_length = (
        schedule["validation_end_year"]
        - schedule["validation_start_year"]
        + 1
    )

    test_length = (
        schedule["test_end_year"]
        - schedule["test_start_year"]
        + 1
    )

    exact_windows = bool(
        train_length.eq(
            config.windows.train_years
        ).all()
        and validation_length.eq(
            config.windows.validation_years
        ).all()
        and test_length.eq(
            config.windows.test_years
        ).all()
    )

    _record(
        rows,
        area="schedule",
        check="exact_window_lengths",
        passed=exact_windows,
        detail=(
            f"{config.windows.train_years}/"
            f"{config.windows.validation_years}/"
            f"{config.windows.test_years}"
        ),
    )

    contiguous = bool(
        (
            schedule["train_end_year"]
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

    _record(
        rows,
        area="schedule",
        check="windows_contiguous",
        passed=contiguous,
        detail=(
            "train -> validation -> test"
        ),
    )


def _audit_data_outputs(
    rows: list[dict],
    config: ExperimentConfig,
) -> None:
    required = {
        "data_audit": (
            config.run_dir
            / "data_audit.json"
        ),
        "target_availability": (
            config.run_dir
            / "target_availability_summary.csv"
        ),
        "oos_universe": (
            config.run_dir
            / "oos_universe.parquet"
        ),
    }

    for name, path in required.items():
        _record(
            rows,
            area="data",
            check=f"{name}_exists",
            passed=path.exists(),
            detail=str(path),
        )


def _audit_prediction_file(
    rows: list[dict],
    *,
    config: ExperimentConfig,
    model_id: str,
    prediction_path: Path,
    require_annual_refits: bool,
) -> None:
    area = f"model:{model_id}"

    predictions = pd.read_parquet(
        prediction_path
    )

    columns_ok = set(
        REQUIRED_PREDICTION_COLUMNS
    ).issubset(
        predictions.columns
    )

    _record(
        rows,
        area=area,
        check="prediction_schema",
        passed=columns_ok,
        detail=(
            "ok"
            if columns_ok
            else (
                "missing="
                f"{sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns))}"
            )
        ),
    )

    if not columns_ok:
        return

    duplicate_keys = predictions.duplicated(
        ["eom", "security_id"]
    ).any()

    _record(
        rows,
        area=area,
        check="unique_month_security_id",
        passed=not duplicate_keys,
        detail=f"rows={len(predictions)}",
    )

    expected_years = (
        _expected_test_years(config)
    )

    observed_years = sorted(
        pd.Series(
            predictions["test_year"]
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    _record(
        rows,
        area=area,
        check="complete_test_years",
        passed=(
            observed_years
            == expected_years
        ),
        detail=(
            f"observed={observed_years[0] if observed_years else None}"
            f"-{observed_years[-1] if observed_years else None}"
        ),
    )

    observed_months = int(
        pd.to_datetime(
            predictions["eom"]
        ).nunique()
    )

    expected_months = (
        12 * len(expected_years)
    )

    _record(
        rows,
        area=area,
        check="complete_oos_calendar",
        passed=(
            observed_months
            == expected_months
        ),
        detail=(
            f"months={observed_months}, "
            f"expected={expected_months}"
        ),
    )

    target_timing_ok = bool(
        (
            pd.to_datetime(
                predictions["target_date"]
            )
            == pd.to_datetime(
                predictions["eom"]
            )
            + pd.offsets.MonthEnd(1)
        ).all()
    )

    _record(
        rows,
        area=area,
        check="one_month_target_timing",
        passed=target_timing_ok,
        detail="target_date = eom + MonthEnd(1)",
    )

    finite_target = (
        pd.to_numeric(
            predictions["y_true"],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .notna()
    )

    availability_ok = bool(
        predictions[
            "target_available"
        ]
        .astype(bool)
        .eq(finite_target)
        .all()
    )

    _record(
        rows,
        area=area,
        check="target_availability_consistent",
        passed=availability_ok,
        detail="target_available matches finite y_true",
    )

    if require_annual_refits:
        def valid_refit(year: int) -> bool:
            directory = config.run_dir / "models" / model_id / f"refit_{year}"
            model_path = directory / "model.bin"
            prediction_path = directory / "predictions.parquet"
            metadata_path = directory / "metadata.json"
            if not all(path.exists() and path.stat().st_size > 0 for path in (model_path, prediction_path, metadata_path)):
                return False
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                prediction = pd.read_parquet(
                    prediction_path,
                    columns=["eom", "security_id", "model_id", "test_year", "refit_id"],
                )
            except Exception:
                return False
            return bool(
                metadata.get("model_id") == model_id
                and int(metadata.get("test_year", -1)) == year
                and not prediction.empty
                and prediction["model_id"].eq(model_id).all()
                and prediction["test_year"].eq(year).all()
                and not prediction.duplicated(["eom", "security_id"]).any()
            )

        missing_refits = [
            year
            for year in expected_years
            if not valid_refit(year)
        ]

        _record(
            rows,
            area=area,
            check="all_annual_refits_saved",
            passed=not missing_refits,
            detail=(
                "complete"
                if not missing_refits
                else f"missing={missing_refits}"
            ),
        )


def _audit_common_model(
    rows: list[dict],
    *,
    config: ExperimentConfig,
    model_id: str,
    require_annual_refits: bool,
) -> None:
    area = f"model:{model_id}"
    paths = _common_paths(
        config.run_dir,
        model_id,
    )

    for label in (
        REQUIRED_COMMON_OUTPUTS
    ):
        path = paths[label]

        _record(
            rows,
            area=area,
            check=f"{label}_exists",
            passed=path.exists(),
            detail=str(path),
        )

    if not paths["prediction"].exists():
        return

    _audit_prediction_file(
        rows,
        config=config,
        model_id=model_id,
        prediction_path=paths[
            "prediction"
        ],
        require_annual_refits=(
            require_annual_refits
        ),
    )

    if paths["metric"].exists():
        metric = json.loads(
            paths["metric"].read_text(
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

        metric_ok = (
            required_metrics
            .issubset(metric)
        )

        _record(
            rows,
            area=area,
            check="metric_schema",
            passed=metric_ok,
            detail=(
                "ok"
                if metric_ok
                else (
                    "missing="
                    f"{sorted(required_metrics - set(metric))}"
                )
            ),
        )

        _record(
            rows,
            area=area,
            check="metric_model_id",
            passed=(
                metric.get("model_id")
                == model_id
            ),
            detail=str(
                metric.get("model_id")
            ),
        )


def _audit_chosen_analysis(
    rows: list[dict],
    *,
    config: ExperimentConfig,
    chosen_model_id: str,
) -> None:
    """
    Check only the chosen-model artifacts that are part of the final design.

    This checks the complete artifact set produced by the frozen chosen-model
    analysis and excludes exploratory outputs that are not part of the final design.
    """
    chosen_dir = (
        config.run_dir
        / "chosen_model_analysis"
        / chosen_model_id
    )

    _record(
        rows,
        area="chosen_model",
        check="analysis_directory_exists",
        passed=chosen_dir.exists(),
        detail=str(chosen_dir),
    )

    if not chosen_dir.exists():
        return

    required_files = (
        "regime_stability_summary.csv",
        "aligned_component_formal_comparisons.csv",
        "lgbm_shap_importance.csv",
        "deepset_permutation_importance.csv",
        "component_importance_rank_comparison.csv",
        "portfolio_robustness_summary.csv",
        "portfolio_robustness_monthly.parquet",
        "monthly_regime_results.parquet",
    )

    for filename in required_files:
        path = chosen_dir / filename

        _record(
            rows,
            area="chosen_model",
            check=filename,
            passed=path.exists(),
            detail=str(path),
        )

    def read_csv(filename: str) -> pd.DataFrame | None:
        path = chosen_dir / filename
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            _record(
                rows,
                area="chosen_model",
                check=f"{filename}_readable",
                passed=False,
                detail=str(exc),
            )
            return None

    expected_features = set(FEATURES_40)
    shap = read_csv("lgbm_shap_importance.csv")
    if shap is not None:
        required = {"regime", "feature", "mean_abs_shap", "rank"}
        schema_ok = required.issubset(shap.columns)
        coverage_ok = schema_ok and set(shap["feature"]) == expected_features and set(shap["regime"]) == {"ALL", "HIGH_VOL", "LOW_VOL"}
        _record(rows, area="chosen_model", check="lgbm_shap_schema_and_coverage", passed=coverage_ok, detail=f"rows={len(shap)}")

    deep = read_csv("deepset_permutation_importance.csv")
    if deep is not None:
        required = {"characteristic", "all_importance", "high_vol_importance", "low_vol_importance", "all_rank", "high_vol_rank", "low_vol_rank"}
        coverage_ok = required.issubset(deep.columns) and set(deep["characteristic"]) == expected_features
        _record(rows, area="chosen_model", check="deepset_importance_schema_and_coverage", passed=coverage_ok, detail=f"rows={len(deep)}")

    robustness = read_csv("portfolio_robustness_summary.csv")
    if robustness is not None:
        expected_strategies = {"BASELINE_EQUAL_WEIGHT", "EX_MICRO_EQUAL_WEIGHT", "VALUE_WEIGHTED"}
        coverage_ok = "strategy" in robustness and set(robustness["strategy"]) == expected_strategies
        _record(rows, area="chosen_model", check="portfolio_robustness_coverage", passed=coverage_ok, detail=f"rows={len(robustness)}")


def run_post_train_audit(
    config: ExperimentConfig,
    *,
    standalone_model_ids: tuple[str, ...] = FINAL_MODEL_ROSTER,
    chosen_model_id: str | None = None,
    require_chosen_analysis: bool = True,
) -> pd.DataFrame:
    """
    Audit the final retrain and saved analysis artifacts.

    This function does not train models, rebuild portfolios, generate report
    figures, or freeze a research manifest. It only checks whether the final
    saved experiment is complete and internally consistent.

    Standalone models are expected to have annual refit prediction files.
    A derived chosen model, such as the fixed 50/50 ensemble, is not.
    """
    rows: list[dict] = []

    _audit_schedule(
        rows,
        config,
    )

    _audit_data_outputs(
        rows,
        config,
    )

    for model_id in (
        standalone_model_ids
    ):
        _audit_common_model(
            rows,
            config=config,
            model_id=model_id,
            require_annual_refits=True,
        )

    if chosen_model_id is not None:
        _audit_common_model(
            rows,
            config=config,
            model_id=chosen_model_id,
            require_annual_refits=False,
        )

        if require_chosen_analysis:
            _audit_chosen_analysis(
                rows,
                config=config,
                chosen_model_id=chosen_model_id,
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "area",
            "check",
            "passed",
            "detail",
        ],
    )

    output_dir = (
        config.run_dir
        / "post_train_audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_dir
        / "audit_checks.csv",
        index=False,
    )

    summary = {
        "passed": bool(
            result["passed"].all()
        )
        if not result.empty
        else False,
        "n_checks": int(
            len(result)
        ),
        "n_failed": int(
            (~result["passed"]).sum()
        )
        if not result.empty
        else 0,
    }

    write_json_atomic(
        summary,
        output_dir
        / "audit_summary.json",
    )

    return result
