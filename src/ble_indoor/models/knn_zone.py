from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ble_indoor.domain.environment import Environment
from ble_indoor.models.features import rssi_feature_matrix

Weights = Literal["uniform", "distance"]

ZONE_ID_COLUMN = "zone_id"


class KnnZoneClassifier:
    """kNN classifier: RSSI features → spatial zone id (0 .. nx*ny-1)."""

    def __init__(
        self,
        k_neighbors: int = 5,
        weights: Weights = "distance",
        *,
        standardize_rssi: bool = False,
    ) -> None:
        self._standardize = standardize_rssi
        self._scaler: StandardScaler | None = None
        self._model = KNeighborsClassifier(
            n_neighbors=k_neighbors,
            weights=weights,
            metric="euclidean",
        )

    def fit(self, train_df: pd.DataFrame, env: Environment) -> None:
        if ZONE_ID_COLUMN not in train_df.columns:
            raise ValueError(f"Missing column '{ZONE_ID_COLUMN}' in training dataframe.")
        X = rssi_feature_matrix(train_df, env)
        y = train_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
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

    def predict_zone_id(self, rssi_dbm: np.ndarray) -> int:
        return int(self._model.predict(self._prepare_X(rssi_dbm))[0])

    def predict_zone_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        return self._model.predict(self._prepare_X(rssi_dbm)).astype(np.int64)
