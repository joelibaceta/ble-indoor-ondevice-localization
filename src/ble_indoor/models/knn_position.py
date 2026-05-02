from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from ble_indoor.domain.environment import Environment
from ble_indoor.models.features import position_matrix, rssi_feature_matrix

Weights = Literal["uniform", "distance"]


class KnnFingerprintPositionModel:
    """kNN regressor: RSSI features → (x, y) in meters."""

    def __init__(
        self,
        k_neighbors: int = 3,
        weights: Weights = "distance",
        *,
        standardize_rssi: bool = False,
    ) -> None:
        self._k = k_neighbors
        self._weights = weights
        self._standardize = standardize_rssi
        self._scaler: StandardScaler | None = None
        self._model = KNeighborsRegressor(
            n_neighbors=k_neighbors,
            weights=weights,
            metric="euclidean",
        )

    def fit(self, train_df: pd.DataFrame, env: Environment) -> None:
        X = rssi_feature_matrix(train_df, env)
        y = position_matrix(train_df)
        if self._standardize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            self._scaler = None
        self._model.fit(X, y)

    def _prepare_X(self, rssi_dbm: np.ndarray) -> np.ndarray:
        X = np.asarray(rssi_dbm, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return X

    def predict_xy_m(self, rssi_dbm: np.ndarray) -> np.ndarray:
        xy = self._model.predict(self._prepare_X(rssi_dbm))
        return xy.reshape(2,)

    def predict_xy_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        """rssi_dbm shape (N, G) -> (N, 2)."""
        return self._model.predict(self._prepare_X(rssi_dbm))
