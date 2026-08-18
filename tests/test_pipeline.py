from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import CORE20, ExperimentConfig, WindowConfig
from src.data import load_and_prepare_panel
from src.evaluation import form_equal_weight_deciles, form_portfolio_variants, oos_r2
from src.models import MODEL_REGISTRY, TRAINERS
from src.schedule import make_rolling_schedule


class PipelineTests(unittest.TestCase):
    def config(self, data_path: Path) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="test",
            project_dir=data_path.parent,
            data_path=data_path,
            output_dir=data_path.parent / "runs",
            selected_models=("LGBM_20",),
            use_gpu=False,
        )

    def test_rank_normalization_does_not_depend_on_target_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.parquet"
            path.touch()
            frame = pd.DataFrame({
                "id": ["a", "b", "c", "d"],
                "eom": pd.to_datetime(["2000-01-31"] * 4),
                "excntry": ["USA"] * 4,
                "permno": [1, 2, 3, 4],
                "size_grp": ["small"] * 4,
                "me": [1.0, 2.0, 3.0, 4.0],
                "ret_exc_lead1m": [0.01, 0.02, 0.03, np.nan],
            })
            for feature in CORE20:
                frame[feature] = [1.0, 2.0, 3.0, 4.0]
            with patch("src.data.pd.read_parquet", return_value=frame):
                panel, _ = load_and_prepare_panel(
                    self.config(path), CORE20, include_core_dynamics=False,
                    include_feature40_lag1=False,
                )
            np.testing.assert_allclose(
                panel[CORE20[0]].to_numpy(), [-1.0, -1 / 3, 1 / 3, 1.0]
            )
            self.assertFalse(bool(panel.loc[panel.permno.eq(4), "target_available"].iloc[0]))

    def test_rolling_schedule_is_exact_15_4_1(self):
        panel = pd.DataFrame({
            "eom": pd.to_datetime([f"{year}-12-31" for year in range(1980, 2025)])
        })
        schedule = make_rolling_schedule(panel, WindowConfig())
        first = schedule.iloc[0]
        self.assertEqual((first.train_start_year, first.train_end_year), (1980, 1994))
        self.assertEqual((first.validation_start_year, first.validation_end_year), (1995, 1998))
        self.assertEqual(first.test_year, 1999)
        self.assertTrue((schedule.train_end_year < schedule.validation_start_year).all())
        self.assertTrue((schedule.validation_end_year < schedule.test_start_year).all())

    def test_gkx_oos_r2_uses_zero_benchmark(self):
        data = pd.DataFrame({"y_true": [1.0, -1.0], "y_pred": [0.5, -0.5]})
        self.assertAlmostEqual(oos_r2(data), 0.75)

    def test_constant_predictions_create_cash_not_tie_break_alpha(self):
        data = self._prediction_month(np.zeros(20), np.linspace(-0.1, 0.1, 20))
        portfolios = form_portfolio_variants(data)
        self.assertTrue((portfolios.long_short_ret == 0.0).all())
        self.assertTrue((~portfolios.signal_available).all())

    def test_portfolios_are_assigned_before_missing_returns(self):
        returns = np.linspace(-0.1, 0.1, 20)
        returns[-1] = np.nan
        data = self._prediction_month(np.arange(20), returns)
        variants = form_portfolio_variants(data)
        tail = variants.loc[variants.strategy.eq("TAIL_10PCT")].iloc[0]
        self.assertEqual(tail.n_long_assigned, 2)
        self.assertEqual(tail.n_long, 1)
        self.assertEqual(tail.long_coverage, 0.5)
        self.assertLessEqual(tail.missing_return_stress_ret, tail.long_short_ret)
        deciles = form_equal_weight_deciles(data)
        self.assertEqual(deciles.R10.iloc[0], 1)
        self.assertEqual(deciles.N10.iloc[0], 2)

    def test_every_registered_model_has_a_trainer(self):
        missing = [
            model_id for model_id, spec in MODEL_REGISTRY.items()
            if spec.trainer_id not in TRAINERS
        ]
        self.assertEqual(missing, [])

    @staticmethod
    def _prediction_month(predictions, returns):
        returns = np.asarray(returns, dtype=float)
        return pd.DataFrame({
            "eom": pd.to_datetime(["2020-01-31"] * len(returns)),
            "id": [str(i) for i in range(len(returns))],
            "permno": np.arange(len(returns)),
            "y_pred": np.asarray(predictions, dtype=float),
            "y_true": returns,
            "target_available": np.isfinite(returns),
        })


if __name__ == "__main__":
    unittest.main()
