"""Tests for kNN k-sweep (validation vs hyperparameter k)."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "src"))

from ble_indoor.evaluation.knn_sweep import sweep_k_neighbors
from ble_indoor.settings import ProjectConfig, ProjectLayout
from ble_indoor.simulation.omnet_trace_loader import load_omnet_training_trace


class TestKnnSweep(unittest.TestCase):
    def test_sweep_returns_series(self) -> None:
        layout = ProjectLayout(ROOT)
        cfg = ProjectConfig.load(layout.default_config_path())
        df = load_omnet_training_trace(
            ROOT / "simulations/omnet/examples/minimal_valid_example.csv",
            cfg.environment,
            cfg.spatial_zones,
        )
        train_df, val_df = df.iloc[:8].copy(), df.iloc[8:].copy()
        self.assertGreaterEqual(len(train_df), 3)
        self.assertGreaterEqual(len(val_df), 2)
        out = sweep_k_neighbors(train_df, val_df, cfg.environment, cfg, k_max=5)
        self.assertEqual(out["k"], [1, 2, 3, 4, 5])
        self.assertEqual(len(out["validation_zone_accuracy"]), 5)
        self.assertEqual(len(out["validation_rmse_xy_m"]), 5)


if __name__ == "__main__":
    unittest.main()
