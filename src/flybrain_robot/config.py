import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from .telemetry import parse_axes


@dataclass
class Config:
    backend: str = "mock"
    camera: int = 0
    video: str | None = None
    pc_host: str = "0.0.0.0"
    pc_port: int = 9001
    robot_ip: str = "192.168.1.50"
    robot_port: int = 9000
    max_speed: float = 35
    smoothing: float = 0.25
    dead_zone: float = 5
    looming_threshold: float = 0.3
    watchdog_timeout: float = 0.5
    invert_left: bool = False
    invert_right: bool = False
    dataset_path: str | None = "data/malecns-v1.0"
    brain_seed: int = 0
    cruise: float = 0.4
    fast_modulators: bool = False
    wiring: str = "connectome"
    imu_axes: str = "x,y,z"  # robot axes (x forward, y left, z up) in sensor axes
    logging_level: str = "INFO"

    def __post_init__(self):
        bounds = {
            "max_speed": (0, 100),
            "smoothing": (0.001, 1),
            "dead_zone": (0, 100),
            "looming_threshold": (0.001, 1),
            "watchdog_timeout": (0.01, 0.5),
            "cruise": (0, 1),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be between {low} and {high}")
        for name in ("pc_port", "robot_port"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 65535:
                raise ValueError(f"Invalid {name}")
        if type(self.brain_seed) is not int or self.brain_seed < 0:
            raise ValueError("brain_seed must be a non-negative integer")
        for name in ("invert_left", "invert_right", "fast_modulators"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.backend not in ("mock", "malecns"):
            raise ValueError("Unknown backend")
        if self.wiring not in ("connectome", "shuffled"):
            raise ValueError("wiring must be 'connectome' or 'shuffled'")
        parse_axes(self.imu_axes)


def load_config(path=None):
    file = Path(path or "config.yaml")
    if not file.exists():
        if path:
            raise ValueError(f"Config does not exist: {path}")
        return Config()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Config)}:
        raise ValueError("Unknown config fields or invalid mapping")
    return Config(**data)
