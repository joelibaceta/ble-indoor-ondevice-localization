"""KD-tree interpolation over a (x,y,RSSI) simulator point cloud; implements `RssiObservationSource`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ble_indoor.domain.environment import Environment
from ble_indoor.simulation.ports import RssiObservationSource


class InterpolatedTraceRssiSource:
    """Interpolate RSSI from trace rows (x_m, y_m, rssi_*) using KD-tree weighted average."""

    def __init__(
        self,
        trace_df: pd.DataFrame,
        environment: Environment,
        *,
        k_neighbors: int = 12,
        micro_noise_floor_db: float = 0.35,
    ) -> None:
        self._env = environment
        self._k = max(1, min(int(k_neighbors), len(trace_df)))
        self._micro_floor = float(micro_noise_floor_db)
        xy = trace_df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
        cols = [f"rssi_{gid}" for gid in environment.gateway_ids]
        self._rssi = trace_df[cols].to_numpy(dtype=np.float64)
        self._tree = cKDTree(xy)

    def _interp(self, position_m: np.ndarray) -> np.ndarray:
        pos = np.asarray(position_m, dtype=np.float64).reshape(2,)
        dists, idx = self._tree.query(pos, k=self._k)
        if np.isscalar(idx):
            idx = np.array([int(idx)], dtype=np.int64)
            dists = np.array([float(dists)], dtype=np.float64)
        else:
            idx = np.asarray(idx, dtype=np.int64)
            dists = np.asarray(dists, dtype=np.float64)
        w = 1.0 / (dists + 1e-6)
        w /= w.sum()
        return (w @ self._rssi[idx]).astype(np.float64)

    def mean_rssi_dbm(self, position_m: np.ndarray) -> np.ndarray:
        return self._interp(position_m)

    def sample_rssi_dbm(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        noise_sigma_db: float | None = None,
    ) -> np.ndarray:
        mean = self._interp(position_m)
        pos = np.asarray(position_m, dtype=np.float64).reshape(2,)
        dists, idx = self._tree.query(pos, k=self._k)
        if np.isscalar(idx):
            neigh = self._rssi[[int(idx)]]
        else:
            neigh = self._rssi[np.asarray(idx, dtype=np.int64)]
        local_std = np.maximum(neigh.std(axis=0), self._micro_floor)
        sigma = self._env.rssi_model.noise_sigma_db if noise_sigma_db is None else float(noise_sigma_db)
        scale = max(sigma * 0.15, self._micro_floor)
        noise = rng.normal(loc=0.0, scale=np.minimum(local_std, scale), size=mean.shape)
        return mean + noise

    def sample_rssi_with_reception(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        gateway_reception_prob: float,
        missing_rssi_dbm: float,
        noise_sigma_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        noisy = self.sample_rssi_dbm(position_m, rng, noise_sigma_db=noise_sigma_db)
        visible = rng.random(size=noisy.shape) < gateway_reception_prob
        out = np.where(visible, noisy, float(missing_rssi_dbm))
        return out, visible
