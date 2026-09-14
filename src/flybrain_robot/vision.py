"""Camera adapter; motion magnitude and center-relative expansion are heuristics."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SensoryInput:
    left_motion: float = 0.0
    right_motion: float = 0.0
    looming: float = 0.0


class VisionEncoder:
    def __init__(self):
        self.previous = None

    def encode(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = gray
            return SensoryInput()
        flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.previous = gray
        magnitude = np.linalg.norm(flow, axis=2)
        y, x = np.mgrid[:120, :160]
        radial = np.stack((x - 79.5, y - 59.5), axis=-1)
        radius = np.maximum(np.linalg.norm(radial, axis=2), 8)
        expansion = (flow * radial).sum(axis=2) / radius
        return SensoryInput(
            float(np.clip(magnitude[:, :80].mean() / 3, 0, 1)),
            float(np.clip(magnitude[:, 80:].mean() / 3, 0, 1)),
            float(np.clip(expansion.mean() / 2, 0, 1)),
        )


def synthetic_frame(index):
    """Deterministic textured expanding target, no camera or data downloads."""
    y, x = np.mgrid[:240, :320]
    radius = 20 + (index % 90) * 1.4
    cx = 160 + 35 * np.sin(index / 20)
    mask = (x - cx) ** 2 + (y - 120) ** 2 < radius**2
    texture = (((x - cx) / radius * 10).astype(int) + ((y - 120) / radius * 10).astype(int)) % 2
    gray = np.where(mask, 80 + texture * 150, 15).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
