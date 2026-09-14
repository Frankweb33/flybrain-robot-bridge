import math
import time


class MotorDecoder:
    def __init__(
        self,
        max_speed=35,
        smoothing=0.25,
        dead_zone=5,
        looming_threshold=0.3,
        watchdog_timeout=0.5,
        invert_left=False,
        invert_right=False,
    ):
        self.maximum = max_speed
        self.alpha = smoothing
        self.dead_zone = dead_zone
        self.threshold = looming_threshold
        self.timeout = watchdog_timeout
        self.signs = (-1 if invert_left else 1, -1 if invert_right else 1)
        self.speeds = (0.0, 0.0)
        self.last_update = None

    def stop(self):
        self.speeds = (0.0, 0.0)
        self.last_update = None
        return (0, 0)

    def expired(self, now=None):
        now = time.monotonic() if now is None else now
        return self.last_update is None or now - self.last_update >= self.timeout

    def read(self, now=None):
        if self.expired(now):
            return self.stop()
        return tuple(round(v) for v in self.speeds)

    def update(self, activity, now=None, emergency_stop=False):
        if emergency_stop:
            return self.stop()
        if not all(math.isfinite(v) for v in (activity.left, activity.right, activity.escape)):
            return self.stop()
        target = (
            (-self.maximum, -self.maximum)
            if activity.escape >= self.threshold
            else (activity.left * self.maximum, activity.right * self.maximum)
        )
        values = []
        for old, value, sign in zip(self.speeds, target, self.signs):
            value = max(-self.maximum, min(self.maximum, value)) * sign
            value = 0 if abs(value) < self.dead_zone else value
            values.append(old + self.alpha * (value - old))
        self.speeds = tuple(values)
        self.last_update = time.monotonic() if now is None else now
        return self.read(self.last_update)
