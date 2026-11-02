"""
Unit tests for evaluation metrics calculation and error analysis.
"""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics


class TestEvaluate(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_perfect_predictions(self):
        """Verify metrics when predictions exactly equal actuals."""
        y_true = pd.Series([5.0, 10.0, 15.0, 20.0])
        y_pred = np.array([5.0, 10.0, 15.0, 20.0])
        metrics = compute_metrics(y_true, y_pred)

        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["R2_Score"], 1.0)
        self.assertEqual(metrics["Median_AE"], 0.0)

    def test_known_metrics_computation(self):
        """Verify metric calculation against known theoretical values."""
        y_true = pd.Series([10.0, 20.0])
        y_pred = np.array([12.0, 16.0])  # errors: +2, -4

        metrics = compute_metrics(y_true, y_pred)
        # MAE = (2 + 4) / 2 = 3.0
        self.assertAlmostEqual(metrics["MAE"], 3.0)
        # MSE = (4 + 16) / 2 = 10.0
        self.assertAlmostEqual(metrics["MSE"], 10.0)
        # RMSE = sqrt(10) ≈ 3.1623
        self.assertAlmostEqual(metrics["RMSE"], 3.1623, places=3)


if __name__ == "__main__":
    unittest.main()
