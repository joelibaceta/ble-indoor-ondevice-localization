"""Random forest: RSSI → (x, y) regression + zone classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ble_indoor.domain.environment import Environment
from ble_indoor.evaluation.fingerprint_eval import evaluate_predictions
from ble_indoor.models.features import position_matrix, rssi_feature_matrix
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN
from ble_indoor.settings import ProjectConfig

_STATE_VERSION = 1


@dataclass
class FingerprintRfEstimator:
    """Two forests on the same scaled RSSI features: position (multi-target) + zone."""

    environment: Environment
    n_estimators: int
    max_depth: int | None
    random_state: int
    standardize_rssi: bool
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _position: RandomForestRegressor | None = field(default=None, repr=False)
    _zone: RandomForestClassifier | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> FingerprintRfEstimator:
        rf = config.fingerprint_rf
        return cls(
            environment=config.environment,
            n_estimators=rf.n_estimators,
            max_depth=rf.max_depth,
            random_state=rf.random_state,
            standardize_rssi=rf.standardize_rssi,
        )

    @property
    def fitted_zone(self) -> bool:
        return self._zone is not None

    def fit(self, train_df: pd.DataFrame, *, fit_zone: bool = True) -> None:
        if len(train_df) < 2:
            raise ValueError("Training dataframe must have at least 2 rows.")
        env = self.environment
        X = rssi_feature_matrix(train_df, env)
        y_xy = position_matrix(train_df)

        if self.standardize_rssi:
            self._scaler = StandardScaler()
            Xf = self._scaler.fit_transform(X)
        else:
            self._scaler = None
            Xf = X

        self._position = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._position.fit(Xf, y_xy)

        self._zone = None
        if fit_zone and ZONE_ID_COLUMN in train_df.columns:
            y_z = train_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
            self._zone = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + 1,
                n_jobs=-1,
            )
            self._zone.fit(Xf, y_z)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._scaler is not None:
            return self._scaler.transform(X)
        return X

    def predict_xy(self, rssi_dbm: np.ndarray) -> np.ndarray:
        return self.predict_xy_batch(np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1)).reshape(2,)

    def predict_xy_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._position is None:
            raise RuntimeError("Call fit() before predict_xy_batch.")
        X = self._transform(np.asarray(rssi_dbm, dtype=np.float64))
        return np.asarray(self._position.predict(X), dtype=np.float64)

    def predict_zone_id(self, rssi_dbm: np.ndarray) -> int:
        return int(self.predict_zone_batch(np.asarray(rssi_dbm, dtype=np.float64).reshape(1, -1))[0])

    def predict_zone_batch(self, rssi_dbm: np.ndarray) -> np.ndarray:
        if self._zone is None:
            raise RuntimeError("Zone model was not fitted (missing zone_id column or fit_zone=False).")
        X = self._transform(np.asarray(rssi_dbm, dtype=np.float64))
        return np.asarray(self._zone.predict(X), dtype=np.int64)

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        if self._position is None:
            raise RuntimeError("Call fit() before evaluate().")
        X = rssi_feature_matrix(df, self.environment)
        pred_xy = self.predict_xy_batch(X)
        pred_z = self.predict_zone_batch(X) if self._zone is not None else None
        return evaluate_predictions(df, pred_xy, pred_z)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_type": "rf",
                "version": _STATE_VERSION,
                "environment": self.environment,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "random_state": self.random_state,
                "standardize_rssi": self.standardize_rssi,
                "scaler": self._scaler,
                "position_model": self._position,
                "zone_model": self._zone,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FingerprintRfEstimator:
        path = Path(path)
        data = joblib.load(path)
        if not isinstance(data, dict) or data.get("model_type") != "rf":
            raise ValueError(f"Not a RandomForest fingerprint model: {path}")
        out = cls(
            environment=data["environment"],
            n_estimators=int(data["n_estimators"]),
            max_depth=data.get("max_depth"),
            random_state=int(data["random_state"]),
            standardize_rssi=bool(data.get("standardize_rssi", True)),
        )
        out._scaler = data.get("scaler")
        out._position = data.get("position_model")
        out._zone = data.get("zone_model")
        return out
