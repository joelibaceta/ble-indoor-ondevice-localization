"""kNN fingerprint localizer: RSSI → (x, y) and zone id; hyperparameters from `ProjectConfig`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from ble_indoor.domain.environment import Environment
from ble_indoor.models.knn_position import KnnFingerprintPositionModel
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN, KnnZoneClassifier
from ble_indoor.settings import ProjectConfig

Weights = Literal["uniform", "distance"]

_STATE_VERSION = 2


@dataclass
class FingerprintKnnEstimator:
    """kNN on RSSI vectors; feature order matches `environment.gateway_ids`."""

    environment: Environment
    position_k_neighbors: int
    position_weights: Weights
    zone_k_neighbors: int
    zone_weights: Weights
    standardize_rssi: bool
    _position: KnnFingerprintPositionModel | None = field(default=None, repr=False)
    _zone: KnnZoneClassifier | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> FingerprintKnnEstimator:
        return cls(
            environment=config.environment,
            position_k_neighbors=config.baseline_knn.k_neighbors,
            position_weights=config.baseline_knn.weights,
            zone_k_neighbors=config.zone_knn.k_neighbors,
            zone_weights=config.zone_knn.weights,
            standardize_rssi=config.baseline_knn.standardize_rssi,
        )

    @property
    def position_model(self) -> KnnFingerprintPositionModel | None:
        return self._position

    @property
    def zone_model(self) -> KnnZoneClassifier | None:
        return self._zone

    @property
    def fitted_zone(self) -> bool:
        return self._zone is not None

    def fit(self, train_df: pd.DataFrame, *, fit_zone: bool = True) -> None:
        n = len(train_df)
        if n < 2:
            raise ValueError("Training dataframe must have at least 2 rows.")
        k_pos = min(self.position_k_neighbors, max(1, n - 1))
        self._position = KnnFingerprintPositionModel(
            k_neighbors=k_pos,
            weights=self.position_weights,
            standardize_rssi=self.standardize_rssi,
        )
        self._position.fit(train_df, self.environment)

        self._zone = None
        if fit_zone and ZONE_ID_COLUMN in train_df.columns:
            k_z = min(self.zone_k_neighbors, max(1, n - 1))
            z = KnnZoneClassifier(
                k_neighbors=k_z,
                weights=self.zone_weights,
                standardize_rssi=self.standardize_rssi,
            )
            z.fit(train_df, self.environment)
            self._zone = z

    def predict_xy(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._position is None:
            raise RuntimeError("Call fit() before predict_xy.")
        v = np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1)
        return self._position.predict_xy_batch(v).reshape(2,)

    def predict_xy_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._position is None:
            raise RuntimeError("Call fit() before predict_xy_batch.")
        return self._position.predict_xy_batch(np.asarray(rssi_dbm, dtype=np.float64))

    def predict_zone_id(self, rssi_dbm: np.ndarray) -> int:
        if self._zone is None:
            raise RuntimeError("Zone model was not fitted (missing zone_id column or fit_zone=False).")
        v = np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1)
        return int(self._zone.predict_zone_batch(v)[0])

    def predict_zone_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._zone is None:
            raise RuntimeError("Zone model was not fitted (missing zone_id column or fit_zone=False).")
        return self._zone.predict_zone_batch(np.asarray(rssi_dbm, dtype=np.float64))

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Metrics on a labeled frame: position RMSE/R2 + error quantiles; zone accuracy if fitted."""
        from ble_indoor.evaluation.fingerprint_eval import evaluate_predictions
        from ble_indoor.models.features import rssi_feature_matrix

        if self._position is None:
            raise RuntimeError("Call fit() before evaluate().")
        X = rssi_feature_matrix(df, self.environment)
        pred_xy = self._position.predict_xy_batch(X)
        pred_z = self._zone.predict_zone_batch(X) if self._zone is not None else None
        return evaluate_predictions(df, pred_xy, pred_z)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "knn",
                "version": _STATE_VERSION,
                "environment": self.environment,
                "position_k_neighbors": self.position_k_neighbors,
                "position_weights": self.position_weights,
                "zone_k_neighbors": self.zone_k_neighbors,
                "zone_weights": self.zone_weights,
                "standardize_rssi": self.standardize_rssi,
                "position_model": self._position,
                "zone_model": self._zone,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FingerprintKnnEstimator:
        path = Path(path)
        data = joblib.load(path)
        if not isinstance(data, dict):
            raise ValueError(f"Unsupported or corrupt state file: {path}")
        if data.get("model_type") not in (None, "knn"):
            raise ValueError(f"Not a kNN fingerprint model: {path}")
        ver = int(data.get("version", 1))
        if ver not in (1, 2):
            raise ValueError(f"Unsupported kNN state version: {path}")
        out = cls(
            environment=data["environment"],
            position_k_neighbors=int(data["position_k_neighbors"]),
            position_weights=data["position_weights"],
            zone_k_neighbors=int(data["zone_k_neighbors"]),
            zone_weights=data["zone_weights"],
            standardize_rssi=bool(data.get("standardize_rssi", False)),
        )
        out._position = data.get("position_model")
        out._zone = data.get("zone_model")
        return out
