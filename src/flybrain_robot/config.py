import math
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


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
    dataset_path: str | None = None
    logging_level: str = "INFO"

    def __post_init__(self):
        bounds = {
            "max_speed": (0, 100),
            "smoothing": (0.001, 1),
            "dead_zone": (0, 100),
            "looming_threshold": (0.001, 1),
            "watchdog_timeout": (0.01, 0.5),
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
        for name in ("invert_left", "invert_right"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.backend not in ("mock", "malecns"):
            raise ValueError("Unknown backend")


def load_config(path=None):
    file = Path(path or "config.yaml")
    if not file.exists():
        if path:
            raise ValueError(f"Config does not exist: {path}")
        return Config()
    data = yaml.safe_load(file.read_text()) or {}
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Config)}:
        raise ValueError("Unknown config fields or invalid mapping")
    return Config(**data)
