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
class TrainingDataSettings:
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
class FingerprintMlpSettings:
    """MLP fingerprint trainer. No extra dependencies for training; TFLite export needs tensorflow."""

    hidden_layer_sizes: list[int]
    activation: str
    learning_rate_init: float
    max_iter: int
    random_state: int
    standardize_rssi: bool


@dataclass(frozen=True)
class SionnaRTSettings:
    """Sionna RT ray-tracing simulator config (requires pip install -r requirements-sionna.txt)."""

    carrier_frequency_hz: float
    max_depth: int
    num_samples: int
    grid_resolution_m: float
    gateway_height_m: float
    rx_height_m: float
    wall_material: str
    cache_file: str | None
    wall_conductivity: float | None
    wall_permittivity: float | None


@dataclass(frozen=True)
class ProjectConfig:
    environment: Environment
    fingerprint_grid: FingerprintGridSettings
    trajectory: TrajectoryDatasetSettings
    spatial_zones: SpatialZoneMap
    baseline_knn: BaselineKnnSettings
    zone_knn: ZoneKnnSettings
    fingerprint_rf: FingerprintRfSettings
    fingerprint_mlp: FingerprintMlpSettings
    training_data: TrainingDataSettings
    sionna_rt: SionnaRTSettings

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

        td = raw.get("training_data", raw.get("omnet", {}))
        trace_csv = td.get("training_trace_csv")
        if trace_csv is not None and not isinstance(trace_csv, str):
            raise TypeError("training_data.training_trace_csv must be a string path or null")
        training_data = TrainingDataSettings(training_trace_csv=str(trace_csv) if trace_csv else None)

        rfo = raw.get("fingerprint_rf", {})
        md = rfo.get("max_depth")
        fingerprint_rf = FingerprintRfSettings(
            n_estimators=int(rfo.get("n_estimators", 150)),
            max_depth=int(md) if md is not None else None,
            random_state=int(rfo.get("random_state", 42)),
            standardize_rssi=bool(rfo.get("standardize_rssi", True)),
        )

        mlpo = raw.get("fingerprint_mlp", {})
        fingerprint_mlp = FingerprintMlpSettings(
            hidden_layer_sizes=list(mlpo.get("hidden_layer_sizes", [64, 32])),
            activation=str(mlpo.get("activation", "relu")),
            learning_rate_init=float(mlpo.get("learning_rate_init", 1e-3)),
            max_iter=int(mlpo.get("max_iter", 500)),
            random_state=int(mlpo.get("random_state", 42)),
            standardize_rssi=bool(mlpo.get("standardize_rssi", True)),
        )

        srt = raw.get("sionna_rt", {})
        sionna_rt = SionnaRTSettings(
            carrier_frequency_hz=float(srt.get("carrier_frequency_hz", 2.4e9)),
            max_depth=int(srt.get("max_depth", 5)),
            num_samples=int(srt.get("num_samples", 1_000_000)),
            grid_resolution_m=float(srt.get("grid_resolution_m", 0.25)),
            gateway_height_m=float(srt.get("gateway_height_m", 2.5)),
            rx_height_m=float(srt.get("rx_height_m", 1.0)),
            wall_material=str(srt.get("wall_material", "concrete")),
            cache_file=str(srt["cache_file"]) if srt.get("cache_file") else None,
            wall_conductivity=float(srt["wall_conductivity"]) if "wall_conductivity" in srt else None,
            wall_permittivity=float(srt["wall_permittivity"]) if "wall_permittivity" in srt else None,
        )

        return cls(
            environment=env,
            fingerprint_grid=fingerprint_grid,
            trajectory=trajectory,
            spatial_zones=spatial_zones,
            baseline_knn=baseline_knn,
            zone_knn=zone_knn,
            fingerprint_rf=fingerprint_rf,
            fingerprint_mlp=fingerprint_mlp,
            training_data=training_data,
            sionna_rt=sionna_rt,
        )
