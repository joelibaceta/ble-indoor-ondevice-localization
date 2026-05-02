"""Analytic log-distance path loss with Gaussian noise and per-gateway reception probability."""

from __future__ import annotations

import numpy as np

from ble_indoor.domain.environment import Environment


class PathLossSimulator:
    """Gateway→badge path loss model; implements `RssiObservationSource`."""

    def __init__(self, environment: Environment) -> None:
        self._env = environment

    @property
    def environment(self) -> Environment:
        return self._env

    def _distances_m(self, position_m: np.ndarray) -> np.ndarray:
        pos = np.asarray(position_m, dtype=np.float64).reshape(2,)
        gw = self._env.gateway_positions_m()
        return np.linalg.norm(gw - pos.reshape(1, 2), axis=1)

    def mean_rssi_dbm(self, position_m: np.ndarray) -> np.ndarray:
        d = self._distances_m(position_m)
        d_eff = np.maximum(d, self._env.rssi_model.min_distance_m)
        n = self._env.rssi_model.path_loss_exponent
        tx = np.array([g.tx_power_dbm for g in self._env.gateways], dtype=np.float64)
        return tx - 10.0 * n * np.log10(d_eff)

    def sample_rssi_dbm(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        noise_sigma_db: float | None = None,
    ) -> np.ndarray:
        sigma = self._env.rssi_model.noise_sigma_db if noise_sigma_db is None else float(noise_sigma_db)
        mean = self.mean_rssi_dbm(position_m)
        noise = rng.normal(loc=0.0, scale=sigma, size=mean.shape)
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
        if not (0.0 <= gateway_reception_prob <= 1.0):
            raise ValueError("gateway_reception_prob must be in [0, 1]")
        noisy = self.sample_rssi_dbm(position_m, rng, noise_sigma_db=noise_sigma_db)
        visible = rng.random(size=noisy.shape) < gateway_reception_prob
        out = np.where(visible, noisy, float(missing_rssi_dbm))
        return out, visible
