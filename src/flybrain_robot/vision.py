"""Camera adapters.

``VisionEncoder`` (vision "flow") reduces dense optical flow to a few per-hemifield
features. These are engineering estimates, not a model of the fly retina:

* ``left_motion`` / ``right_motion``: mean flow magnitude in each image half.
* ``looming_left`` / ``looming_right``: positive flow divergence inside moving regions,
  away from motion boundaries. Uniform translation and rotation have zero divergence,
  so a pan or a sideways pass does not register as looming; a growing object does.
* ``object_left`` / ``object_right``: flow that differs from the half's median flow,
  i.e. something moving relative to the background rather than the whole view.
* ``yaw_flow``: median horizontal flow, positive when the scene moves right
  (as it does when the camera turns left).

``RetinaEncoder`` (vision "retina") computes no motion at all: it reports how much
darker each optic-lobe column's pixel became since the previous frame, and the
connectome has to compute motion from that (``connectome/retinotopy.py``).
"""

from dataclasses import dataclass, field

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
    # Retina mode: darkening per optic-lobe column, 0..1, in RetinaEncoder column order.
    retina: np.ndarray | None = field(default=None, compare=False)


class RetinaEncoder:
    """Per-column temporal darkening at the pixels given by ``x``, ``y`` (160x120 image)."""

    def __init__(self, x, y):
        self.xi = np.clip(np.round(np.asarray(x)).astype(int), 0, WIDTH - 1)
        self.yi = np.clip(np.round(np.asarray(y)).astype(int), 0, HEIGHT - 1)
        self.previous = None

    def reset(self):
        self.previous = None

    def encode(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2GRAY)
        values = cv2.blur(gray, (3, 3)).astype(np.float64)[self.yi, self.xi]
        if self.previous is None:
            self.previous = values
            return SensoryInput(retina=np.zeros(len(values)))
        darkening = np.clip((self.previous - values) / np.maximum(self.previous, 8.0), 0.0, 1.0)
        self.previous = values
        return SensoryInput(retina=darkening)


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
# Motion-vision test patterns, not part of the tour: a square-wave grating of vertical
# bars drifting left or right, and the same grating reversing contrast in place
# ("flicker"), which darkens each column as often as the drifting grating but carries
# no coherent motion.
PATTERNS = ("grating_left", "grating_right", "flicker")
GRATING_PERIOD_PX = 40
GRATING_SPEED_PX = 3
# Contrast reversal every 4 frames gives the same mean column darkening per frame as the
# drifting grating through RetinaEncoder (0.088 vs 0.086, measured 2026-09-16 before any
# neuron was simulated), so neither pattern delivers more input than the other.
FLICKER_FRAMES = 4
TOUR_SEGMENT = 90


def grating(offset, inverted=False):
    x = np.arange(320)
    dark = ((x + offset) % GRATING_PERIOD_PX) < GRATING_PERIOD_PX // 2
    if inverted:
        dark = ~dark
    row = np.where(dark, 40, 200).astype(np.uint8)
    return cv2.cvtColor(np.tile(row, (240, 1)), cv2.COLOR_GRAY2BGR)


def scene_frame(name, index):
    """Deterministic 320x240 test scenes; ``tour`` cycles through the SCENES."""
    if name == "tour":
        name = SCENES[(index // TOUR_SEGMENT) % len(SCENES)]
        index %= TOUR_SEGMENT
    if name == "grating_left":  # bars move toward the left edge
        return grating(GRATING_SPEED_PX * index)
    if name == "grating_right":
        return grating(-GRATING_SPEED_PX * index)
    if name == "flicker":
        return grating(0, inverted=bool((index // FLICKER_FRAMES) % 2))
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
        raise ValueError(
            f"Unknown scene {name!r}; choose from tour, {', '.join(SCENES + PATTERNS)}"
        )
    return render_scene(cx, looming)


def synthetic_frame(index):
    """Default synthetic input: the scripted ``tour`` through all test scenes."""
    return scene_frame("tour", index)
