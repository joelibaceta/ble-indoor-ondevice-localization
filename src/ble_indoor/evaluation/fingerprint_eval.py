"""Shared train/validation metrics for RSSI → position / zone models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from ble_indoor.evaluation.metrics import error_summary, position_errors_m
from ble_indoor.models.features import position_matrix
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN


def evaluate_predictions(
    df: pd.DataFrame,
    pred_xy: np.ndarray,
    pred_zone: np.ndarray | None,
) -> dict[str, Any]:
    """Position RMSE/R²/quantiles; optional zone accuracy when ``pred_zone`` is given."""
    y_true = position_matrix(df)
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred_xy)))
    r2 = float(r2_score(y_true, pred_xy))
    err = position_errors_m(y_true, pred_xy)
    out: dict[str, Any] = {
        "position": {"rmse_xy_m": rmse, "r2": r2, **error_summary(err)},
    }
    if pred_zone is not None and ZONE_ID_COLUMN in df.columns:
        z_true = df[ZONE_ID_COLUMN].to_numpy(dtype=np.int64)
        out["zone"] = {
            "accuracy": float(accuracy_score(z_true, np.asarray(pred_zone, dtype=np.int64))),
            "n": int(len(df)),
        }
    return out
