"""Eight hand-designed leaky populations, not a connectome simulation."""

import math

import numpy as np

from .base import BrainBackend, MotorActivity


class MockBrain(BrainBackend):
    names = (
        "left_motion",
        "right_motion",
        "looming",
        "balance_left",
        "balance_right",
        "left_motor",
        "right_motor",
        "escape",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros(8)

    def step(self, sensory_input, telemetry, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        s = sensory_input
        balance = float(np.clip(telemetry.gyro[2] / 90, -1, 1))
        drive = np.array(
            [
                s.left_motion,
                s.right_motion,
                s.looming,
                max(balance, 0),
                max(-balance, 0),
                0.15 + s.right_motion * 0.7 - max(balance, 0),
                0.15 + s.left_motion * 0.7 - max(-balance, 0),
                s.looming,
            ]
        )
        self.state += (np.clip(drive, 0, 1) - self.state) * (1 - math.exp(-dt / 0.12))
        return self.get_motor_activity()

    def get_motor_activity(self):
        return MotorActivity(*(float(v) for v in self.state[5:8]))
