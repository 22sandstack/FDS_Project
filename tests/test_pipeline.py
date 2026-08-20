from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.chosen_model_analysis import compare_model_pairs
from src.config import FEATURES_40, FINAL_MODEL_ROSTER, ExperimentConfig, PortfolioConfig, WindowConfig
from src.ensemble import ENSEMBLE_ID
from src.self_checks import run_framework_self_checks


class FinalPipelineTests(unittest.TestCase):
    def test_framework_self_checks(self) -> None:
        run_framework_self_checks()

    def test_final_roster_contains_17_results(self) -> None:
        self.assertEqual(len(FINAL_MODEL_ROSTER), 16)
        self.assertEqual(len(set((*FINAL_MODEL_ROSTER, ENSEMBLE_ID))), 17)
        self.assertEqual(len(FEATURES_40), 40)

    def test_removed_configuration_options_are_absent(self) -> None:
        self.assertFalse(hasattr(WindowConfig(), "refit_month"))
        self.assertFalse(hasattr(PortfolioConfig(), "weighting"))

    def test_multiple_exploratory_pairs(self) -> None:
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
            values = {
                "MODEL_A": np.linspace(0.01, 0.05, 36),
                "MODEL_B": np.linspace(0.00, 0.03, 36),
                "MODEL_C": np.linspace(-0.01, 0.02, 36),
            }
            for model_id, series in values.items():
                pd.DataFrame({"eom": dates, "rank_ic": series}).to_parquet(
                    diagnostics / f"{model_id}_monthly_rank_ic.parquet"
                )
                pd.DataFrame({"eom": dates, "long_short_ret": series / 2}).to_parquet(
                    portfolios / f"{model_id}_long_short.parquet"
                )
            result = compare_model_pairs(
                config, [("MODEL_A", "MODEL_B"), ("MODEL_C", "MODEL_A")]
            )
            self.assertEqual(len(result), 4)
            self.assertEqual(set(result["metric"]), {"rank_ic", "long_short_return"})
            self.assertTrue(result["interpretation"].str.len().gt(0).all())

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
        self.assertIn("compare_model_pairs", chosen)
        self.assertIn("exploratory_pairwise_comparisons.csv", chosen)
        self.assertIn("aligned_component_formal_comparisons.csv", chosen)


if __name__ == "__main__":
    unittest.main()
