"""Unit tests for omnet_trace_loader."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

from ble_indoor.settings import ProjectConfig, ProjectLayout
from ble_indoor.simulation.omnet_trace_loader import load_omnet_trace_points_only, load_omnet_training_trace


class TestOmnetTraceLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = ProjectLayout(ROOT)
        self.cfg = ProjectConfig.load(self.layout.default_config_path())
        self.env = self.cfg.environment
        self.example_csv = ROOT / "simulations/omnet/examples/minimal_valid_example.csv"

    def test_load_training_trace_adds_zones(self) -> None:
        df = load_omnet_training_trace(self.example_csv, self.env, self.cfg.spatial_zones)
        self.assertIn("x_m", df.columns)
        self.assertIn("y_m", df.columns)
        self.assertIn("zone_id", df.columns)
        self.assertIn("zone_name", df.columns)
        self.assertTrue((df["zone_id"] >= 0).all())
        self.assertTrue((df["zone_id"] < self.cfg.spatial_zones.n_zones).all())
        for gid in self.env.gateway_ids:
            self.assertIn(f"rssi_{gid}", df.columns)

    def test_stale_zone_columns_recomputed_from_xy(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "x_m,y_m,zone_id,zone_name,rssi_A1,rssi_A2,rssi_A3,rssi_A4\n"
                "1.0,1.0,99,WRONG,-62,-71,-68,-74\n"
            )
            path = Path(f.name)
        try:
            df = load_omnet_training_trace(path, self.env, self.cfg.spatial_zones)
            self.assertNotEqual(int(df.loc[0, "zone_id"]), 99)
            self.assertEqual(df.loc[0, "zone_name"], self.cfg.spatial_zones.zone_name(int(df.loc[0, "zone_id"])))
        finally:
            path.unlink(missing_ok=True)

    def test_load_points_only_subset_columns(self) -> None:
        df = load_omnet_trace_points_only(self.example_csv, self.env)
        want = ["x_m", "y_m"] + [f"rssi_{g}" for g in self.env.gateway_ids]
        self.assertListEqual(list(df.columns), want)

    def test_missing_column_raises(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x_m,y_m,rssi_gw1,rssi_gw2\n0,0,-70,-75\n")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_omnet_trace_points_only(path, self.env)
            self.assertIn("Missing", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
