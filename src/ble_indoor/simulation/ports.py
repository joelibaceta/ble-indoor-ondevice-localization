"""Protocol for RSSI vectors at 2D positions (analytic path loss or OMNeT-backed sources)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RssiObservationSource(Protocol):
    """RSSI samples at badge position (meters, same frame as `Environment`)."""

    def mean_rssi_dbm(self, position_m: np.ndarray) -> np.ndarray:
        """Mean RSSI per gateway, order `environment.gateway_ids`."""
        ...

    def sample_rssi_dbm(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        noise_sigma_db: float | None = None,
    ) -> np.ndarray:
        """One noisy draw per gateway."""
        ...

    def sample_rssi_with_reception(
        self,
        position_m: np.ndarray,
        rng: np.random.Generator,
        *,
        gateway_reception_prob: float,
        missing_rssi_dbm: float,
        noise_sigma_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rssi_vector, visibility_bool_per_gateway)."""
        ...
