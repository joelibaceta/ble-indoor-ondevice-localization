"""Uniform spatial zone grid over the room rectangle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ble_indoor.domain.environment import Room


@dataclass(frozen=True)
class SpatialZoneMap:
    """Uniform grid on [0, width_m]×[0, height_m]; zone_id = iy*nx+ix, iy=0 bottom row."""

    nx: int
    ny: int
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be >= 1")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("Invalid room dimensions")

    @classmethod
    def from_room_division(cls, nx: int, ny: int, room: Room) -> SpatialZoneMap:
        return cls(nx=nx, ny=ny, width_m=room.width_m, height_m=room.height_m)

    @property
    def n_zones(self) -> int:
        return self.nx * self.ny

    @property
    def x_edges_m(self) -> np.ndarray:
        return np.linspace(0.0, self.width_m, self.nx + 1, dtype=np.float64)

    @property
    def y_edges_m(self) -> np.ndarray:
        return np.linspace(0.0, self.height_m, self.ny + 1, dtype=np.float64)

    def zone_name(self, zone_id: int) -> str:
        iy, ix = divmod(int(zone_id), self.nx)
        return f"Z{iy}_{ix}"

    def zone_labels(self) -> list[str]:
        return [self.zone_name(z) for z in range(self.n_zones)]

    def label_xy(self, x_m: float, y_m: float) -> tuple[int, str]:
        x = float(np.clip(x_m, 0.0, self.width_m - 1e-9))
        y = float(np.clip(y_m, 0.0, self.height_m - 1e-9))
        ix = int(np.searchsorted(self.x_edges_m, x, side="right") - 1)
        iy = int(np.searchsorted(self.y_edges_m, y, side="right") - 1)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        zid = iy * self.nx + ix
        return zid, self.zone_name(zid)

    def label_xy_batch(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorizado: devuelve (zone_ids int64, zone_names object array)."""
        x = np.clip(np.asarray(x_m, dtype=np.float64), 0.0, self.width_m - 1e-9)
        y = np.clip(np.asarray(y_m, dtype=np.float64), 0.0, self.height_m - 1e-9)
        ix = np.searchsorted(self.x_edges_m, x, side="right") - 1
        iy = np.searchsorted(self.y_edges_m, y, side="right") - 1
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)
        zid = (iy * self.nx + ix).astype(np.int64)
        names = np.array([self.zone_name(int(z)) for z in zid], dtype=object)
        return zid, names
