"""kNN validation sweep vs neighbor count k (no training epochs)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from ble_indoor.domain.environment import Environment
from ble_indoor.models.features import position_matrix, rssi_feature_matrix
from ble_indoor.models.knn_position import KnnFingerprintPositionModel
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN, KnnZoneClassifier
from ble_indoor.settings import ProjectConfig


def sweep_k_neighbors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    env: Environment,
    cfg: ProjectConfig,
    *,
    k_max: int | None = None,
) -> dict[str, Any]:
    """For each k in 1..min(k_max, n_train-1), fit position + zone kNN on ``train_df`` and score on ``val_df``."""
    n = len(train_df)
    if n < 3:
        raise ValueError("Need at least 3 training rows for k-sweep.")
    hi = min(int(k_max) if k_max is not None else n - 1, n - 1)
    std = bool(cfg.baseline_knn.standardize_rssi)
    w_pos = cfg.baseline_knn.weights
    w_zone = cfg.zone_knn.weights

    ks: list[int] = []
    zone_acc: list[float] = []
    rmse: list[float] = []

    Xv = rssi_feature_matrix(val_df, env)
    yv = position_matrix(val_df)
    zv = val_df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)

    for k in range(1, hi + 1):
        pos = KnnFingerprintPositionModel(k_neighbors=k, weights=w_pos, standardize_rssi=std)
        pos.fit(train_df, env)
        pred_xy = pos.predict_xy_batch(Xv)
        rmse.append(float(np.sqrt(mean_squared_error(yv, pred_xy))))

        zc = KnnZoneClassifier(k_neighbors=k, weights=w_zone, standardize_rssi=std)
        zc.fit(train_df, env)
        pred_z = zc.predict_zone_batch(Xv)
        zone_acc.append(float(accuracy_score(zv, pred_z)))
        ks.append(k)

    return {
        "k": ks,
        "validation_zone_accuracy": zone_acc,
        "validation_rmse_xy_m": rmse,
        "standardize_rssi": std,
        "position_weights": w_pos,
        "zone_weights": w_zone,
    }
