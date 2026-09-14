import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Telemetry:
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees per second
    accel: tuple[float, float, float] = (0.0, 0.0, 1.0)  # g


def synthetic_telemetry(t):
    return Telemetry((0.0, 0.0, 8 * math.sin(t)), (0.05 * math.sin(t), 0.0, 1.0))
