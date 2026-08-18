from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import CORE20, ExperimentConfig, UniverseConfig, WindowConfig
from src.data import load_and_prepare_panel
from src.chosen_model_analysis import _hybrid_component_fit_details
from src.evaluation import (
    form_equal_weight_deciles,
    form_portfolio_variants,
    fractional_tail_membership,
    oos_r2,
)
from src.models import (
    MODEL_FEATURES,
    MODEL_REGISTRY,
    TRAINERS,
    _arrays,
    _finite_target_mask,
    train_strict_validation_hybrid,
)
from src.portfolio_robustness import build_robustness_portfolios
from src.runner import ExperimentRunner
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

    def test_default_security_identifier_is_jkp_id(self):
        self.assertEqual(UniverseConfig().security_id_col, "id")

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
                    self.config(path), CORE20,
                    include_feature40_lag1=False,
                )
            np.testing.assert_allclose(
                panel[CORE20[0]].to_numpy(), [-1.0, -1 / 3, 1 / 3, 1.0]
            )
            self.assertFalse(bool(panel.loc[panel.permno.eq(4), "target_available"].iloc[0]))

    def test_infinite_characteristics_are_neutral_after_monthly_ranking(self):
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
                "ret_exc_lead1m": [0.01, 0.02, 0.03, 0.04],
            })
            for feature in CORE20:
                frame[feature] = [1.0, 2.0, 3.0, np.inf]
            with patch("src.data.pd.read_parquet", return_value=frame):
                panel, _ = load_and_prepare_panel(
                    self.config(path), CORE20, include_feature40_lag1=False
                )
            np.testing.assert_allclose(
                panel[CORE20[0]].to_numpy(), [-1.0, 0.0, 1.0, 0.0]
            )

    def test_all_flat_trainers_share_the_finite_target_mask(self):
        frame = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "ret": [0.01, np.nan, np.inf, -np.inf],
        })
        mask = _finite_target_mask(frame, "ret")
        self.assertEqual(mask.tolist(), [True, False, False, False])
        arrays = _arrays(frame, frame, frame, ("x",), "ret")
        self.assertEqual(arrays[0].shape, (1, 1))
        self.assertEqual(arrays[2].shape, (1, 1))
        self.assertTrue(np.isfinite(arrays[1]).all())
        self.assertTrue(np.isfinite(arrays[3]).all())

    def test_hybrid_importance_reads_nested_parent_fit_metadata(self):
        metadata = {"fit_details": {
            "component_a_id": "LGBM_40",
            "component_b_id": "DEEPSET_40_DYNAMIC",
            "component_a_fit": {"selected_iteration": 17},
            "component_b_fit": {"best_epoch": 4},
        }}
        details = _hybrid_component_fit_details(
            metadata, "HYBRID_LGBM40_DEEPSET40_DYNAMIC", "LGBM_40"
        )
        self.assertEqual(details["selected_iteration"], 17)

    def test_international_panel_uses_jkp_id_without_permno(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.parquet"
            path.touch()
            frame = pd.DataFrame({
                "id": ["gb1", "gb2"],
                "eom": pd.to_datetime(["2000-01-31"] * 2),
                "excntry": ["GBR"] * 2,
                "size_grp": ["small", "large"],
                "me": [1.0, 2.0],
                "ret_exc_lead1m": [0.01, 0.02],
            })
            for feature in CORE20:
                frame[feature] = [1.0, 2.0]
            config = ExperimentConfig(
                experiment_id="international_test",
                project_dir=path.parent,
                data_path=path,
                output_dir=path.parent / "runs",
                universe=UniverseConfig(
                    country="GBR", start_year=2000, end_year=2024,
                    security_id_col="id",
                ),
                selected_models=("LGBM_20",),
                use_gpu=False,
            )
            with patch("src.data.pd.read_parquet", return_value=frame):
                panel, _ = load_and_prepare_panel(
                    config, CORE20,
                    include_feature40_lag1=False,
                )
            self.assertEqual(panel.security_id.tolist(), ["gb1", "gb2"])
            self.assertNotIn("permno", panel.columns)

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
        self.assertNotIn("INDUCED_SET_TRANSFORMER_20", MODEL_REGISTRY)

    def test_registry_uses_general_ranked_characteristic_label(self):
        self.assertEqual(ExperimentConfig.feature_set_id, "RANKED_CHARACTERISTICS")
        self.assertTrue(all(
            spec.feature_set_id == "RANKED_CHARACTERISTICS"
            for spec in MODEL_REGISTRY.values()
        ))

    def test_lightgbm_does_not_declare_inactive_row_subsampling(self):
        lightgbm_specs = [
            spec for spec in MODEL_REGISTRY.values()
            if spec.trainer_id == "lightgbm"
        ]
        self.assertTrue(lightgbm_specs)
        self.assertTrue(all("subsample" not in spec.params for spec in lightgbm_specs))

    def test_signatures_include_implementation_fingerprint(self):
        runner = ExperimentRunner(self.config(Path("panel.parquet")))
        signature = runner._model_signature("LGBM_40")
        self.assertEqual(len(signature), 16)
        self.assertEqual(len(runner._implementation_fingerprint()), 64)
        with patch.object(runner, "_implementation_fingerprint", return_value="changed"):
            changed_signature = runner._model_signature("LGBM_40")
        self.assertNotEqual(signature, changed_signature)

    def test_legacy_ranked_feature_label_is_manifest_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root / "panel.parquet")
            config.run_dir.mkdir(parents=True)
            manifest = config.to_dict()
            manifest.pop("selected_models", None)
            manifest["feature_set_id"] = "CORE20_RANKED"
            (config.run_dir / "experiment_manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="utf-8"
            )
            ExperimentRunner(config)._write_manifest()

    def test_nn4_has_four_hidden_layers_and_core20_inputs(self):
        specification = MODEL_REGISTRY["NN4_20"]
        self.assertEqual(specification.params["hidden_dims"], [32, 16, 8, 4])
        self.assertEqual(specification.trainer_id, "feedforward_nn")

    def test_nn2_20_uses_two_hidden_layers_and_core20_inputs(self):
        specification_20 = MODEL_REGISTRY["NN2_20"]
        self.assertEqual(specification_20.params["hidden_dims"], [32, 16])
        self.assertEqual(specification_20.trainer_id, "feedforward_nn")
        self.assertEqual(len(MODEL_FEATURES["NN2_20"]), 20)

    def test_lgbm40_deepset40_hybrids_use_strict_three_plus_one(self):
        hybrid_ids = (
            "HYBRID_MLP40_DEEPSET40",
            "HYBRID_LGBM40_DEEPSET40",
            "HYBRID_LGBM40_DEEPSET40_DYNAMIC",
        )
        for model_id in hybrid_ids:
            specification = MODEL_REGISTRY[model_id]
            self.assertEqual(specification.trainer_id, "strict_validation_hybrid")
            self.assertEqual(specification.params["base_validation_years"], 3)
            self.assertEqual(specification.params["weight_validation_years"], 1)
        static = MODEL_REGISTRY["HYBRID_LGBM40_DEEPSET40"]
        dynamic = MODEL_REGISTRY["HYBRID_LGBM40_DEEPSET40_DYNAMIC"]
        self.assertEqual(static.trainer_id, "strict_validation_hybrid")
        self.assertEqual(dynamic.trainer_id, "strict_validation_hybrid")
        self.assertEqual(static.params["base_validation_years"], 3)
        self.assertEqual(static.params["weight_validation_years"], 1)
        self.assertEqual(static.params["component_a_id"], "LGBM_40")
        self.assertEqual(static.params["component_b_id"], "DEEPSET_40")
        self.assertEqual(dynamic.params["component_a_id"], "LGBM_40")
        self.assertEqual(dynamic.params["component_b_id"], "DEEPSET_40_DYNAMIC")
        self.assertEqual(len(MODEL_FEATURES["HYBRID_LGBM40_DEEPSET40"]), 40)
        self.assertGreater(len(MODEL_FEATURES["HYBRID_LGBM40_DEEPSET40_DYNAMIC"]), 40)

    def test_partial_ties_receive_fractional_tail_membership(self):
        month = self._prediction_month(
            [0, 0, 0, 1, 2, 3, 4, 5, 5, 5], np.arange(10) / 100
        )
        assigned = fractional_tail_membership(month, 0.20)
        self.assertAlmostEqual(assigned["short_membership"].sum(), 2.0)
        self.assertAlmostEqual(assigned["long_membership"].sum(), 2.0)
        self.assertTrue(np.allclose(
            assigned.loc[assigned.y_pred.eq(0), "short_membership"], 2 / 3
        ))
        self.assertTrue(np.allclose(
            assigned.loc[assigned.y_pred.eq(5), "long_membership"], 2 / 3
        ))

    def test_tied_portfolio_is_invariant_to_security_id(self):
        predictions = np.repeat(np.arange(5), 4)
        returns = np.linspace(-0.2, 0.2, 20)
        first = self._prediction_month(predictions, returns)
        second = first.copy()
        second["security_id"] = second["security_id"].sample(
            frac=1.0, random_state=1
        ).to_numpy()
        result_a = form_portfolio_variants(first)
        result_b = form_portfolio_variants(second)
        np.testing.assert_allclose(result_a.long_short_ret, result_b.long_short_ret)
        decile_a = form_equal_weight_deciles(first)
        decile_b = form_equal_weight_deciles(second)
        np.testing.assert_allclose(decile_a.long_short_ret, decile_b.long_short_ret)

    def test_observed_return_outlier_scenarios_are_reported(self):
        data = self._prediction_month(np.arange(20), np.linspace(-0.2, 0.2, 20))
        data["me"] = np.arange(1, 21, dtype=float)
        data["size_grp"] = "small"
        data.loc[data.index[-1], "y_true"] = 20.0
        result = build_robustness_portfolios(data, "FULL", "EQUAL")
        self.assertIn("outlier_winsor_1pct_ret", result)
        self.assertIn("outlier_exclude_abs_gt_10_ret", result)
        self.assertEqual(int(result["n_selected_abs_gt_10"].sum()), 1)
        self.assertLess(
            result["outlier_exclude_abs_gt_10_ret"].iloc[0],
            result["gross_long_short_ret"].iloc[0],
        )

    def test_strict_hybrid_reserves_fourth_validation_year_for_weights(self):
        train = pd.DataFrame({
            "eom": pd.to_datetime(["2000-01-31"]), "ret_exc_lead1m": [0.0]
        })
        validation = pd.DataFrame({
            "eom": pd.to_datetime([
                "2001-01-31", "2002-01-31", "2003-01-31", "2004-01-31"
            ]),
            "id": ["a", "a", "a", "a"],
            "security_id": ["a", "a", "a", "a"],
            "ret_exc_lead1m": [0.0, 0.0, 0.0, 0.75],
        })
        test = pd.DataFrame({
            "eom": pd.to_datetime(["2005-01-31"]), "id": ["a"],
            "security_id": ["a"], "ret_exc_lead1m": [0.0],
        })
        seen_validation_years = []

        def fake_component(component_id, train_frame, validation_frame, prediction_panel, *args):
            del train_frame, args
            seen_validation_years.append(tuple(validation_frame.eom.dt.year.unique()))
            value = 0.0 if component_id == "LGBM_40" else 1.0
            return np.full(len(prediction_panel), value), {"component": component_id}

        params = dict(MODEL_REGISTRY["HYBRID_LGBM40_DEEPSET40"].params)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"dir": root, "model": root / "model.bin"}
            with patch("src.models._fit_hybrid_component", side_effect=fake_component):
                prediction, details = train_strict_validation_hybrid(
                    train, validation, test, (), "ret_exc_lead1m", params,
                    paths, 42, "cpu",
                )
            self.assertEqual(seen_validation_years, [(2001, 2002, 2003)] * 2)
            self.assertEqual(details["weight_validation_years"], [2004])
            self.assertAlmostEqual(details["weight_b"], 0.75)
            self.assertAlmostEqual(float(prediction[0]), 0.75)
            self.assertTrue((root / "aligned_component_predictions.parquet").exists())

    @staticmethod
    def _prediction_month(predictions, returns):
        returns = np.asarray(returns, dtype=float)
        return pd.DataFrame({
            "eom": pd.to_datetime(["2020-01-31"] * len(returns)),
            "id": [str(i) for i in range(len(returns))],
            "security_id": [str(i) for i in range(len(returns))],
            "y_pred": np.asarray(predictions, dtype=float),
            "y_true": returns,
            "target_available": np.isfinite(returns),
        })


if __name__ == "__main__":
    unittest.main()
