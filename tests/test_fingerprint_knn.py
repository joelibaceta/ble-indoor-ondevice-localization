"""Tests for `FingerprintKnnEstimator`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT / "src"))

from ble_indoor.models.fingerprint_knn import FingerprintKnnEstimator
from ble_indoor.settings import ProjectConfig, ProjectLayout
from ble_indoor.simulation.omnet_trace_loader import load_omnet_training_trace


class TestFingerprintKnn(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = ProjectLayout(ROOT)
        self.cfg = ProjectConfig.load(self.layout.default_config_path())
        self.example = ROOT / "simulations/omnet/examples/minimal_valid_example.csv"

    def test_fit_predict_roundtrip_save_load(self) -> None:
        df = load_omnet_training_trace(self.example, self.cfg.environment, self.cfg.spatial_zones)
        est = FingerprintKnnEstimator.from_config(self.cfg)
        est.fit(df, fit_zone=True)
        xy = est.predict_xy(df.iloc[0][[f"rssi_{g}" for g in self.cfg.environment.gateway_ids]].to_numpy())
        self.assertEqual(xy.shape, (2,))
        z = est.predict_zone_id(df.iloc[0][[f"rssi_{g}" for g in self.cfg.environment.gateway_ids]].to_numpy())
        self.assertIsInstance(z, int)
        m = est.evaluate(df)
        self.assertIn("position", m)
        self.assertIn("rmse_xy_m", m["position"])
        self.assertIn("zone", m)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.joblib"
            est.save(p)
            est2 = FingerprintKnnEstimator.load(p)
            xy2 = est2.predict_xy(df.iloc[0][[f"rssi_{g}" for g in self.cfg.environment.gateway_ids]].to_numpy())
            self.assertTrue((xy == xy2).all())


if __name__ == "__main__":
    unittest.main()
