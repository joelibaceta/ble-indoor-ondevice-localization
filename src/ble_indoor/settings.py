"""YAML-backed project config and filesystem layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ble_indoor.domain.environment import Environment
from ble_indoor.domain.zones import SpatialZoneMap

Weights = Literal["uniform", "distance"]


@dataclass(frozen=True)
class ProjectLayout:
    """Repository root paths (config, data, results)."""

    repo_root: Path

    def default_config_path(self) -> Path:
        return self.repo_root / "config" / "baseline_room.yaml"

    def data_simulated_dir(self) -> Path:
        return self.repo_root / "data" / "simulated"

    def data_results_dir(self) -> Path:
        return self.repo_root / "data" / "results"

    def model_results_dir(self, model: str) -> Path:
        """Per-model outputs, e.g. ``data/results/knn`` or ``data/results/rf``."""
        return self.data_results_dir() / model

    def resolve_repo_path(self, relative: str | Path) -> Path:
        """Resolve path relative to repo root (absolute paths unchanged)."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()


@dataclass(frozen=True)
class FingerprintGridSettings:
    train_grid_spacing_m: float
    test_grid_spacing_m: float
    test_offset_x_m: float
    test_offset_y_m: float
    train_samples_per_point: int
    random_seed: int


@dataclass(frozen=True)
class TrajectoryDatasetSettings:
    n_sessions: int
    steps_per_session: int
    step_m: float
    brownian_std_m: float
    start_margin_m: float
    gateway_reception_prob: float
    min_visible_gateways: int
    missing_rssi_dbm: float
    seed: int


@dataclass(frozen=True)
class BaselineKnnSettings:
    k_neighbors: int
    weights: Weights
    standardize_rssi: bool


@dataclass(frozen=True)
class ZoneKnnSettings:
    k_neighbors: int
    weights: Weights


@dataclass(frozen=True)
class OmnetSettings:
    """OMNeT++ trace CSV path (relative to repo or absolute)."""

    training_trace_csv: str | None


@dataclass(frozen=True)
class FingerprintRfSettings:
    """RandomForest fingerprint trainer (separate from kNN YAML block)."""

    n_estimators: int
    max_depth: int | None
    random_state: int
    standardize_rssi: bool


@dataclass(frozen=True)
class ProjectConfig:
    environment: Environment
    fingerprint_grid: FingerprintGridSettings
    trajectory: TrajectoryDatasetSettings
    spatial_zones: SpatialZoneMap
    baseline_knn: BaselineKnnSettings
    zone_knn: ZoneKnnSettings
    fingerprint_rf: FingerprintRfSettings
    omnet: OmnetSettings

    @classmethod
    def load(cls, config_path: str | Path) -> ProjectConfig:
        path = Path(config_path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = Environment.from_mapping(raw)

        fp = raw.get("fingerprint", {})
        fingerprint_grid = FingerprintGridSettings(
            train_grid_spacing_m=float(fp.get("train_grid_spacing_m", 1.0)),
            test_grid_spacing_m=float(fp.get("test_grid_spacing_m", 1.0)),
            test_offset_x_m=float(fp.get("test_offset_x_m", 0.5)),
            test_offset_y_m=float(fp.get("test_offset_y_m", 0.5)),
            train_samples_per_point=int(fp.get("train_samples_per_point", 40)),
            random_seed=int(fp.get("random_seed", 0)),
        )

        t = raw.get("trajectory_dataset", {})
        trajectory = TrajectoryDatasetSettings(
            n_sessions=int(t.get("n_sessions", 10)),
            steps_per_session=int(t.get("steps_per_session", 200)),
            step_m=float(t.get("step_m", 0.35)),
            brownian_std_m=float(t.get("brownian_std_m", 0.08)),
            start_margin_m=float(t.get("start_margin_m", 0.4)),
            gateway_reception_prob=float(t.get("gateway_reception_prob", 0.88)),
            min_visible_gateways=int(t.get("min_visible_gateways", 3)),
            missing_rssi_dbm=float(t.get("missing_rssi_dbm", -105.0)),
            seed=int(t.get("seed", 0)),
        )

        sz = raw.get("spatial_zones", {})
        nx = int(sz.get("nx", 3))
        ny = int(sz.get("ny", 3))
        spatial_zones = SpatialZoneMap.from_room_division(nx, ny, env.room)

        bl = raw.get("baseline", {})
        w1 = str(bl.get("weights", "distance"))
        if w1 not in ("uniform", "distance"):
            raise ValueError("baseline.weights must be 'uniform' or 'distance'")
        baseline_knn = BaselineKnnSettings(
            k_neighbors=int(bl.get("k_neighbors", 3)),
            weights=w1,  # type: ignore[arg-type]
            standardize_rssi=bool(bl.get("standardize_rssi", True)),
        )

        zk = raw.get("zone_knn", raw.get("quadrant_knn", {}))
        w2 = str(zk.get("weights", "distance"))
        if w2 not in ("uniform", "distance"):
            raise ValueError("zone_knn.weights must be 'uniform' or 'distance'")
        zone_knn = ZoneKnnSettings(k_neighbors=int(zk.get("k_neighbors", 7)), weights=w2)  # type: ignore[arg-type]

        om = raw.get("omnet", {})
        trace_csv = om.get("training_trace_csv")
        if trace_csv is not None and not isinstance(trace_csv, str):
            raise TypeError("omnet.training_trace_csv must be a string path or null")
        omnet = OmnetSettings(training_trace_csv=str(trace_csv) if trace_csv else None)

        rfo = raw.get("fingerprint_rf", {})
        md = rfo.get("max_depth")
        fingerprint_rf = FingerprintRfSettings(
            n_estimators=int(rfo.get("n_estimators", 150)),
            max_depth=int(md) if md is not None else None,
            random_state=int(rfo.get("random_state", 42)),
            standardize_rssi=bool(rfo.get("standardize_rssi", True)),
        )

        return cls(
            environment=env,
            fingerprint_grid=fingerprint_grid,
            trajectory=trajectory,
            spatial_zones=spatial_zones,
            baseline_knn=baseline_knn,
            zone_knn=zone_knn,
            fingerprint_rf=fingerprint_rf,
            omnet=omnet,
        )
