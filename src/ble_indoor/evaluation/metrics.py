from __future__ import annotations

import numpy as np


def position_errors_m(true_xy_m: np.ndarray, est_xy_m: np.ndarray) -> np.ndarray:
    d = est_xy_m - true_xy_m
    return np.linalg.norm(d, axis=1)


def error_summary(errors_m: np.ndarray) -> dict[str, float]:
    e = np.asarray(errors_m, dtype=np.float64).ravel()
    return {
        "mean_m": float(np.mean(e)),
        "median_m": float(np.median(e)),
        "p90_m": float(np.percentile(e, 90)),
        "p95_m": float(np.percentile(e, 95)),
        "max_m": float(np.max(e)),
        "std_m": float(np.std(e, ddof=0)),
        "n": float(e.size),
    }
