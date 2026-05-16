"""Load RSSI/position training traces from CSV (any simulator: path loss, Sionna RT, or compatible)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ble_indoor.domain.environment import Environment
from ble_indoor.domain.zones import SpatialZoneMap
from ble_indoor.models.knn_zone import ZONE_ID_COLUMN


def _required_rssi_columns(env: Environment) -> list[str]:
    return [f"rssi_{gid}" for gid in env.gateway_ids]


def load_training_trace(
    path: str | Path,
    environment: Environment,
    spatial_zones: SpatialZoneMap,
) -> pd.DataFrame:
    """Parse CSV: required x_m, y_m, rssi_<gateway_id>; optional columns kept.

    ``zone_id`` / ``zone_name`` are always recomputed from ``x_m``, ``y_m`` and the given
    ``spatial_zones`` so they stay consistent with ``config/baseline_room.yaml``.
    """
    path = Path(path)
    df = pd.read_csv(path)
    req = ["x_m", "y_m", *_required_rssi_columns(environment)]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    zids, znames = spatial_zones.label_xy_batch(
        out["x_m"].to_numpy(dtype=np.float64),
        out["y_m"].to_numpy(dtype=np.float64),
    )
    out[ZONE_ID_COLUMN] = zids
    out["zone_name"] = znames.astype(str)
    return out


def load_trace_points_only(path: str | Path, environment: Environment) -> pd.DataFrame:
    """Return x_m, y_m and rssi_* columns only (no zone labels)."""
    path = Path(path)
    df = pd.read_csv(path)
    req = ["x_m", "y_m", *_required_rssi_columns(environment)]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for trace point cloud: {missing}")
    return df[req].copy()
