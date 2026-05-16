from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Room:
    width_m: float
    height_m: float


@dataclass(frozen=True)
class Gateway:
    id: str
    x_m: float
    y_m: float
    tx_power_dbm: float


@dataclass(frozen=True)
class RssiModelParams:
    path_loss_exponent: float
    noise_sigma_db: float
    min_distance_m: float


@dataclass(frozen=True)
class Environment:
    room: Room
    gateways: tuple[Gateway, ...]
    rssi_model: RssiModelParams

    @property
    def gateway_ids(self) -> tuple[str, ...]:
        return tuple(g.id for g in self.gateways)

    def gateway_positions_m(self) -> np.ndarray:
        return np.array([[g.x_m, g.y_m] for g in self.gateways], dtype=np.float64)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Environment:
        room_cfg = raw["room"]
        room = Room(width_m=float(room_cfg["width_m"]), height_m=float(room_cfg["height_m"]))
        gw_list = raw["gateways"]
        if len(gw_list) < 2:
            raise ValueError(f"Need at least 2 gateways, got {len(gw_list)}")
        gateways: list[Gateway] = []
        for item in gw_list:
            gateways.append(
                Gateway(
                    id=str(item["id"]),
                    x_m=float(item["x_m"]),
                    y_m=float(item["y_m"]),
                    tx_power_dbm=float(item["tx_power_dbm"]),
                )
            )
        rssi_cfg = raw.get("rssi_model", {})
        rssi = RssiModelParams(
            path_loss_exponent=float(rssi_cfg.get("path_loss_exponent", 2.2)),
            noise_sigma_db=float(rssi_cfg.get("noise_sigma_db", 3.0)),
            min_distance_m=float(rssi_cfg.get("min_distance_m", 0.5)),
        )
        env = cls(room=room, gateways=tuple(gateways), rssi_model=rssi)
        env._validate_in_room()
        return env

    @classmethod
    def from_yaml(cls, path: str | Path) -> Environment:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    def _validate_in_room(self) -> None:
        for g in self.gateways:
            if not (0.0 <= g.x_m <= self.room.width_m and 0.0 <= g.y_m <= self.room.height_m):
                raise ValueError(
                    f"Gateway {g.id} at ({g.x_m}, {g.y_m}) is outside the room "
                    f"[0, {self.room.width_m}] x [0, {self.room.height_m}]"
                )
