from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.chosen_model_analysis import compare_prespecified_families
from src.config import FEATURES_40, FINAL_MODEL_ROSTER, ExperimentConfig, PortfolioConfig, WindowConfig
from src.ensemble import ENSEMBLE_ID
from src.self_checks import run_framework_self_checks


class FinalPipelineTests(unittest.TestCase):
    def test_framework_self_checks(self) -> None:
        run_framework_self_checks()

    def test_final_roster_contains_18_results(self) -> None:
        self.assertEqual(len(FINAL_MODEL_ROSTER), 17)
        self.assertEqual(len(set((*FINAL_MODEL_ROSTER, ENSEMBLE_ID))), 18)
        self.assertEqual(len(FEATURES_40), 40)

    def test_removed_configuration_options_are_absent(self) -> None:
        self.assertFalse(hasattr(WindowConfig(), "refit_month"))
        self.assertFalse(hasattr(PortfolioConfig(), "weighting"))

    def test_four_prespecified_pair_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ExperimentConfig(
                experiment_id="test", project_dir=root,
                data_path=root / "unused.parquet", output_dir=root,
                selected_models=("LGBM_40",), use_gpu=False,
            )
            diagnostics = config.run_dir / "diagnostics"
            portfolios = config.run_dir / "portfolios"
            diagnostics.mkdir(parents=True)
            portfolios.mkdir()
            dates = pd.date_range("2020-01-31", periods=36, freq="ME")
            model_ids = {
                ENSEMBLE_ID, "LGBM_40", "DEEPSET_40_DYNAMIC",
                "DEEPSET_40", "MLP_40", "DEEPSET_40_LAG1",
                "MLP_40_LAG1",
            }
            values = {
                model_id: np.linspace(0.001 * index, 0.03 + 0.001 * index, 36)
                for index, model_id in enumerate(sorted(model_ids))
            }
            for model_id, series in values.items():
                pd.DataFrame({"eom": dates, "rank_ic": series}).to_parquet(
                    diagnostics / f"{model_id}_monthly_rank_ic.parquet"
                )
                pd.DataFrame({"eom": dates, "long_short_ret": series / 2}).to_parquet(
                    portfolios / f"{model_id}_long_short.parquet"
                )
            result = compare_prespecified_families(config, ENSEMBLE_ID)
            self.assertEqual(len(result), 22)
            self.assertEqual(set(result["metric"]), {"rank_ic", "long_short_return"})
            self.assertEqual(
                result["family"].value_counts().to_dict(),
                {
                    "F4_TEMPORAL_DEVELOPMENT": 8,
                    "F2_CLOUD_INFORMATION_DESIGN": 6,
                    "F1_ENSEMBLE_COMPONENTS": 4,
                    "F3_BENCHMARK_PERFORMANCE": 4,
                },
            )
            self.assertTrue(result["formal_interpretation"].str.len().gt(0).all())

    def test_submission_notebooks_are_clean_and_current(self) -> None:
        root = Path(__file__).resolve().parents[1]
        notebook_names = {
            "Model_Selection_Pipeline.ipynb",
            "Chosen_Model_Analysis.ipynb",
            "Developed_Markets_Validation.ipynb",
        }
        self.assertEqual({path.name for path in root.glob("*.ipynb")}, notebook_names)
        for name in notebook_names:
            path = root / name
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("FDS Project", text)
            self.assertNotIn("\ufffd", text)
        chosen = (root / "Chosen_Model_Analysis.ipynb").read_text(encoding="utf-8")
        self.assertIn("compare_prespecified_families", chosen)
        self.assertIn("prespecified_paired_comparisons.csv", chosen)
        self.assertNotIn("exploratory_pairwise_comparisons.csv", chosen)


if __name__ == "__main__":
    unittest.main()
