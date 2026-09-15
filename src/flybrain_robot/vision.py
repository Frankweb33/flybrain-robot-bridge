"""Camera adapter: dense optical flow reduced to a few per-hemifield visual features.

These are engineering estimates, not a model of the fly retina:

* ``left_motion`` / ``right_motion``: mean flow magnitude in each image half.
* ``looming_left`` / ``looming_right``: net flow divergence (expansion) in each half.
  Uniform translation and rotation have zero divergence, so a pan or a sideways
  pass does not register as looming; an object growing in the image does.
* ``object_left`` / ``object_right``: flow that differs from the half's median flow,
  i.e. something moving relative to the background rather than the whole view.
* ``yaw_flow``: median horizontal flow, positive when the scene moves right
  (as it does when the camera turns left).
"""

from dataclasses import dataclass

import cv2
import numpy as np

WIDTH, HEIGHT = 160, 120
LOOMING_GAIN = 60.0
MOVING_PX = 0.15
EDGE_PX = 4
OBJECT_GAIN = 2.0
MOTION_GAIN = 1 / 3


@dataclass(frozen=True)
class SensoryInput:
    left_motion: float = 0.0
    right_motion: float = 0.0
    looming: float = 0.0
    looming_left: float = 0.0
    looming_right: float = 0.0
    object_left: float = 0.0
    object_right: float = 0.0
    yaw_flow: float = 0.0


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


class VisionEncoder:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def encode(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = gray
            return SensoryInput()
        flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.previous = gray
        return features(flow)


def features(flow):
    u = cv2.GaussianBlur(flow[..., 0], (0, 0), 1)
    v = cv2.GaussianBlur(flow[..., 1], (0, 0), 1)
    divergence = np.gradient(u, axis=1) + np.gradient(v, axis=0)
    magnitude = np.hypot(u, v)
    # Summed over a region, divergence only measures flow across its border, so an
    # object on a still background always nets zero. Measure it inside moving areas,
    # away from motion boundaries, where expansion and translation differ.
    moving = (magnitude > MOVING_PX).astype(np.uint8)
    interior = cv2.erode(moving, np.ones((2 * EDGE_PX + 1,) * 2, np.uint8)).astype(bool)
    expansion = np.where(interior, np.maximum(divergence, 0), 0)
    half = WIDTH // 2
    values = {}
    for side, cols in (("left", slice(0, half)), ("right", slice(half, WIDTH))):
        su, sv = u[:, cols], v[:, cols]
        residual = np.hypot(su - np.median(su), sv - np.median(sv))
        values[f"{side}_motion"] = _clip01(magnitude[:, cols].mean() * MOTION_GAIN)
        values[f"looming_{side}"] = _clip01(expansion[:, cols].mean() * LOOMING_GAIN)
        # Mean of the strongest 10 % of residual flow: small objects count, noise does not.
        top = np.partition(residual.ravel(), -residual.size // 10)[-residual.size // 10 :]
        values[f"object_{side}"] = _clip01(top.mean() * OBJECT_GAIN)
    values["looming"] = max(values["looming_left"], values["looming_right"])
    values["yaw_flow"] = float(np.clip(np.median(u) / 4, -1, 1))
    return SensoryInput(**values)


def _texture(height, width, seed, cell=6):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 2, (height // cell + 2, width // cell + 2)).astype(np.uint8)
    return cv2.resize(small * 255, None, fx=cell, fy=cell, interpolation=cv2.INTER_NEAREST)


BACKGROUND = _texture(240, 640, seed=7)
OBJECT_TEXTURE = _texture(240, 320, seed=11, cell=10)


def render_scene(cx, radius, pan=0.0, cy=120.0):
    """Textured background (shifted by ``pan`` px) with a textured disk in front."""
    shift = int(round(pan)) % BACKGROUND.shape[1]
    background = np.roll(BACKGROUND, -shift, axis=1)[:240, :320]
    gray = (40 + background * 0.25).astype(np.uint8)
    y, x = np.mgrid[:240, :320]
    mask = (x - cx) ** 2 + (y - cy) ** 2 < radius**2
    scale = 20 / max(radius, 1)
    tx = ((x - cx) * scale + 160).astype(int) % 320
    ty = ((y - cy) * scale + 120).astype(int) % 240
    gray = np.where(mask, 70 + OBJECT_TEXTURE[ty, tx] * 0.7, gray).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


SCENES = ("static", "pan", "pass_left", "approach_center", "approach_left", "approach_right")
TOUR_SEGMENT = 90


def scene_frame(name, index):
    """Deterministic 320x240 test scenes; ``tour`` cycles through all of them."""
    if name == "tour":
        name = SCENES[(index // TOUR_SEGMENT) % len(SCENES)]
        index %= TOUR_SEGMENT
    t = index % TOUR_SEGMENT
    if name == "static":
        return render_scene(160, 0)
    if name == "pan":
        return render_scene(160, 0, pan=3 * index)
    if name == "pass_left":
        return render_scene(40 + 0.9 * t, 18, cy=110)
    looming = 12 + 90 * (t / TOUR_SEGMENT) ** 3
    cx = {"approach_center": 160, "approach_left": 70, "approach_right": 250}.get(name)
    if cx is None:
        raise ValueError(f"Unknown scene {name!r}; choose from tour, {', '.join(SCENES)}")
    return render_scene(cx, looming)


def synthetic_frame(index):
    """Default synthetic input: the scripted ``tour`` through all test scenes."""
    return scene_frame("tour", index)
