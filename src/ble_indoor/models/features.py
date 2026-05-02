from __future__ import annotations

import numpy as np
import pandas as pd

from ble_indoor.domain.environment import Environment


def rssi_feature_matrix(df: pd.DataFrame, env: Environment) -> np.ndarray:
    cols = [f"rssi_{gid}" for gid in env.gateway_ids]
    return df[cols].to_numpy(dtype=np.float64)


def position_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[["x_m", "y_m"]].to_numpy(dtype=np.float64)
