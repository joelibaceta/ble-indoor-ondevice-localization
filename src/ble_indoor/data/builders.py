"""Synthetic grid and trajectory fingerprint datasets from an `RssiObservationSource`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ble_indoor.domain.environment import Environment
from ble_indoor.domain.zones import SpatialZoneMap
from ble_indoor.settings import FingerprintGridSettings, TrajectoryDatasetSettings
from ble_indoor.simulation.ports import RssiObservationSource


def _train_grid(room_width_m: float, room_height_m: float, spacing_m: float) -> np.ndarray:
    xs = np.arange(spacing_m, room_width_m, spacing_m, dtype=np.float64)
    ys = np.arange(spacing_m, room_height_m, spacing_m, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Empty train grid: check spacing vs room size.")
    return np.array(np.meshgrid(xs, ys, indexing="ij")).reshape(2, -1).T


def _test_grid(
    room_width_m: float,
    room_height_m: float,
    spacing_m: float,
    offset_x_m: float,
    offset_y_m: float,
) -> np.ndarray:
    xs = np.arange(spacing_m + offset_x_m, room_width_m, spacing_m, dtype=np.float64)
    ys = np.arange(spacing_m + offset_y_m, room_height_m, spacing_m, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Empty test grid: check offsets and spacing.")
    return np.array(np.meshgrid(xs, ys, indexing="ij")).reshape(2, -1).T


class GridFingerprintDatasetBuilder:
    """Mean RSSI at regular grid points (multiple noisy draws per vertex)."""

    def __init__(
        self,
        environment: Environment,
        grid: FingerprintGridSettings,
        simulator: RssiObservationSource,
    ) -> None:
        self._env = environment
        self._grid = grid
        self._sim = simulator

    def build_train_dataframe(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._grid.random_seed)
        grid = _train_grid(self._env.room.width_m, self._env.room.height_m, self._grid.train_grid_spacing_m)
        rssi_cols = [f"rssi_{gid}" for gid in self._env.gateway_ids]
        rows: list[list[float]] = []
        for x, y in grid:
            samples = np.stack(
                [
                    self._sim.sample_rssi_dbm(np.array([x, y]), rng)
                    for _ in range(self._grid.train_samples_per_point)
                ],
                axis=0,
            )
            mean_rssi = samples.mean(axis=0)
            rows.append([float(x), float(y), *mean_rssi.tolist()])
        return pd.DataFrame(rows, columns=["x_m", "y_m", *rssi_cols])

    def build_test_positions_dataframe(self) -> pd.DataFrame:
        g = _test_grid(
            self._env.room.width_m,
            self._env.room.height_m,
            self._grid.test_grid_spacing_m,
            self._grid.test_offset_x_m,
            self._grid.test_offset_y_m,
        )
        return pd.DataFrame(g, columns=["x_m", "y_m"])


class TrajectoryFingerprintDatasetBuilder:
    """Random walk trajectory with stochastic gateway visibility."""

    def __init__(
        self,
        environment: Environment,
        trajectory: TrajectoryDatasetSettings,
        spatial_zones: SpatialZoneMap,
        simulator: RssiObservationSource,
    ) -> None:
        self._env = environment
        self._traj = trajectory
        self._zones = spatial_zones
        self._sim = simulator

    @staticmethod
    def _clip_xy(xy: np.ndarray, w: float, h: float) -> np.ndarray:
        return np.clip(xy, [0.0, 0.0], [w, h])

    def _random_start(self, rng: np.random.Generator) -> np.ndarray:
        w, h = self._env.room.width_m, self._env.room.height_m
        m = self._traj.start_margin_m
        lo = np.array([m, m], dtype=np.float64)
        hi = np.array([w - m, h - m], dtype=np.float64)
        if np.any(lo >= hi):
            raise ValueError("start_margin_m too large for room.")
        return rng.uniform(lo, hi)

    def _step(self, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        w, h = self._env.room.width_m, self._env.room.height_m
        dirs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64)
        d = dirs[rng.integers(0, 4)]
        delta = d * self._traj.step_m + rng.normal(0.0, self._traj.brownian_std_m, size=2)
        return self._clip_xy(pos + delta, w, h)

    def build_dataframe(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._traj.seed)
        w, h = self._env.room.width_m, self._env.room.height_m
        rssi_cols = [f"rssi_{gid}" for gid in self._env.gateway_ids]
        rows: list[dict[str, Any]] = []
        for session in range(self._traj.n_sessions):
            pos = self._random_start(rng)
            for _ in range(self._traj.steps_per_session):
                pos = self._step(pos, rng)
                rssi, vis = self._sim.sample_rssi_with_reception(
                    pos,
                    rng,
                    gateway_reception_prob=self._traj.gateway_reception_prob,
                    missing_rssi_dbm=self._traj.missing_rssi_dbm,
                )
                n_vis = int(np.sum(vis))
                if n_vis < self._traj.min_visible_gateways:
                    continue
                zid, zname = self._zones.label_xy(float(pos[0]), float(pos[1]))
                row: dict[str, Any] = {
                    "session_id": session,
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "zone_id": int(zid),
                    "zone_name": zname,
                    "n_visible": n_vis,
                }
                for c, v in zip(rssi_cols, rssi.tolist(), strict=True):
                    row[c] = float(v)
                rows.append(row)
        if not rows:
            raise RuntimeError(
                "No trajectory rows: lower min_visible_gateways or raise gateway_reception_prob / sessions."
            )
        return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
