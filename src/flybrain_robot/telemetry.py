"""Robot IMU telemetry and the body features derived from it.

Robot axes: x forward, y left, z up. At rest the accelerometer reads +1 g along the
axis that points up, so pitching nose-down by an angle a gives accel.x = -sin(a) and
rolling onto the left side gives accel.y = -sin(a). If the board is mounted
differently, set ``imu_axes`` in the config instead of reflashing.
"""

import math
from dataclasses import dataclass

TILT_FLOOR_DEG = 3.0  # below this a walking robot's sway is treated as level
TILT_FULL_DEG = 20.0  # tilt that maps to feature value 1.0
BODY_FEATURES = ("tilt_forward", "tilt_back", "tilt_left", "tilt_right")


@dataclass(frozen=True)
class Telemetry:
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees per second
    accel: tuple[float, float, float] = (0.0, 0.0, 1.0)  # g

    def remapped(self, axes):
        """Reorder and flip axes, e.g. ``"-y,x,z"`` means robot x = -sensor y."""
        if axes in (None, "x,y,z"):
            return self
        return Telemetry(remap(self.gyro, axes), remap(self.accel, axes))


def parse_axes(axes):
    parts = [p.strip() for p in str(axes).split(",")]
    if len(parts) != 3 or sorted(p.lstrip("-") for p in parts) != ["x", "y", "z"]:
        raise ValueError("imu_axes must name x, y and z once each, e.g. '-y,x,z'")
    return [("xyz".index(p.lstrip("-")), -1.0 if p.startswith("-") else 1.0) for p in parts]


def remap(vector, axes):
    return tuple(sign * vector[index] for index, sign in parse_axes(axes))


def tilt_angles(accel):
    """(nose-down pitch, left-down roll) in degrees; (0, 0) if the reading is unusable."""
    ax, ay, az = accel
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if not math.isfinite(norm) or norm < 0.2:  # free fall or a broken sample
        return 0.0, 0.0

    def angle(component):
        return math.degrees(math.asin(max(-1.0, min(1.0, -component / norm))))

    return angle(ax), angle(ay)


def tilt_features(telemetry):
    """Tilt in each direction scaled to 0..1 between TILT_FLOOR_DEG and TILT_FULL_DEG."""
    pitch, roll = tilt_angles(telemetry.accel)

    def unit(degrees):
        return min(max((degrees - TILT_FLOOR_DEG) / (TILT_FULL_DEG - TILT_FLOOR_DEG), 0.0), 1.0)

    return dict(
        tilt_forward=unit(pitch), tilt_back=unit(-pitch), tilt_left=unit(roll), tilt_right=unit(-roll)
    )


def tilted_telemetry(roll_left_deg=0.0, pitch_down_deg=0.0):
    """Static accelerometer reading of a robot tilted by the given angles."""
    roll, pitch = math.radians(roll_left_deg), math.radians(pitch_down_deg)
    ax = -math.sin(pitch)
    ay = -math.sin(roll) * math.cos(pitch)
    az = math.cos(roll) * math.cos(pitch)
    return Telemetry(accel=(ax, ay, az))


def synthetic_telemetry(t):
    return Telemetry((0.0, 0.0, 8 * math.sin(t)), (0.05 * math.sin(t), 0.0, 1.0))
