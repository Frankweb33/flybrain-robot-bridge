"""Record a run so that the recording itself is the evidence, and plot it afterwards.

    python -m flybrain_robot.main --backend malecns --camera 0 --config config.yaml --send --record runs
    python -m flybrain_robot.recording runs/20260916-101500Z

``--record DIR`` writes ``DIR/<UTC start time>/`` with

- ``video.avi``: every camera frame the brain received, with frame number, run time and
  motor command burned in;
- ``log.jsonl``: one line per frame with the same frame number and time: encoder
  features, IMU telemetry and its state, tilt, named group rates, motor activity, the
  command sent, the stop flag and the simulation speed;
- ``meta.json``: commit (``+dirty`` with uncommitted changes), command line,
  configuration file values, the effective run settings after command-line
  overrides, and package versions.

All times come from one monotonic clock started at the first frame. To line up an
external video, film the laptop screen: the status line prints ``frame=`` every ten
frames, and the same numbers are in the video and the log.
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

from .telemetry import tilt_angles

PACKAGE_DIR = Path(__file__).resolve().parent


def commit():
    """Short hash of the checked-out code, ``+dirty`` if it has uncommitted changes."""
    try:
        run = dict(capture_output=True, text=True, check=True, cwd=PACKAGE_DIR)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **run).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], **run)
        return head + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions():
    found = dict(python=platform.python_version(), platform=platform.platform())
    for name in ("flybrain-robot-bridge", "numpy", "opencv-python-headless", "numba",
                 "numba-cuda-mlir"):  # fmt: skip
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return found


def _round(values, digits=4):
    return [round(float(v), digits) for v in values]


def nice(value):
    """Smallest 1, 2 or 5 times a power of ten that is at least ``value``."""
    exponent = 10 ** np.floor(np.log10(value))
    return float(next(m * exponent for m in (1, 2, 5, 10) if m * exponent >= value))


class Recorder:
    def __init__(self, root, config, settings, argv, frame_period):
        started = datetime.now(timezone.utc)
        self.directory = Path(root) / started.strftime("%Y%m%d-%H%M%SZ")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frame_period = frame_period
        self.video = None
        meta = dict(
            created=started.isoformat(timespec="seconds"), commit=commit(), argv=list(argv),
            config=asdict(config), run=settings, frame_period_s=frame_period,
            versions=versions(),
        )  # fmt: skip
        (self.directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.log = (self.directory / "log.jsonl").open("w", encoding="utf-8")

    def write(self, index, t, frame, sensory, telemetry, imu, activity, command, stopped,
              sequence, rates, realtime):  # fmt: skip
        pitch, roll = tilt_angles(telemetry.accel)
        line = dict(
            frame=index, t=round(t, 4), sequence=sequence,
            looming=_round([sensory.looming_left, sensory.looming_right]),
            object=_round([sensory.object_left, sensory.object_right]),
            yaw_flow=round(float(sensory.yaw_flow), 4),
            imu=imu, gyro=_round(telemetry.gyro), accel=_round(telemetry.accel),
            roll_left_deg=round(roll, 2), pitch_down_deg=round(pitch, 2),
            activity=_round([activity.left, activity.right, activity.escape]),
            command=[int(command[0]), int(command[1])], stopped=bool(stopped),
            rates={k: round(float(v), 1) for k, v in rates.items()},
            brain_x_realtime=round(realtime, 3),
        )  # fmt: skip
        self.log.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.video_frame(index, t, frame, command, stopped)

    def video_frame(self, index, t, frame, command, stopped):
        image = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        image = np.ascontiguousarray(image, dtype=np.uint8).copy()
        height, width = image.shape[:2]
        if self.video is None:
            path = str(self.directory / "video.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video = cv2.VideoWriter(path, fourcc, 1 / self.frame_period, (width, height))
            if not self.video.isOpened():
                raise OSError(f"Cannot write {path}")
        scale = max(width / 640, 0.35)
        band = int(30 * scale)
        cv2.rectangle(image, (0, 0), (width, band), (0, 0, 0), -1)
        text = f"frame {index:05d}  t {t:8.3f} s  command {command[0]:+d}/{command[1]:+d}"
        if stopped:
            text += "  STOP"
        cv2.putText(image, text, (int(8 * scale), int(21 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), max(1, round(scale)), cv2.LINE_AA)  # fmt: skip
        self.video.write(image)

    def close(self):
        self.log.close()
        if self.video is not None:
            self.video.release()


def read_log(directory):
    path = Path(directory) / "log.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"{path} has no frames")
    return rows


def plot(directory, out=None):
    """Giant Fiber rates, motor command and yaw rate against run time (needs the demo extra)."""
    from PIL import Image, ImageDraw, ImageFont

    directory = Path(directory)
    rows = read_log(directory)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    out = Path(out) if out else directory / "plot.png"

    def font(size):
        for name in ("C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):  # fmt: skip
            if Path(name).exists():
                return ImageFont.truetype(name, size)
        return ImageFont.load_default(size=size)

    bg, panel, grid, ink, muted = "#08121d", "#122331", "#27405a", "#eaf5fc", "#8fa9ba"
    left_color, right_color, stop_color = "#2aa87f", "#8a7fc4", "#d9822b"
    width, height, margin = 1200, 820, 32
    t = np.array([r["t"] for r in rows])
    rates = [r.get("rates", {}) for r in rows]
    series = [
        ("Giant Fiber (DNp01)", "Hz", [x.get("DNp01:L", 0) for x in rates],
         [x.get("DNp01:R", 0) for x in rates], ("left", "right")),
        ("Motor command", "speed, −100 … 100", [r["command"][0] for r in rows],
         [r["command"][1] for r in rows], ("left wheel", "right wheel")),
        ("IMU", "yaw rate, °/s (positive = turning left)", [r["gyro"][2] for r in rows], None,
         ("gyro z", None)),
    ]  # fmt: skip
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title = f"Recorded run {directory.name}"
    draw.text((margin, 20), title, font=font(24), fill=ink)
    subtitle = (f"commit {meta.get('commit', '?')} · {meta.get('run', {}).get('backend', '?')} "
                f"backend · {len(rows)} frames · {t[-1]:.1f} s")  # fmt: skip
    draw.text((margin, 54), subtitle, font=font(13), fill=muted)
    top, gap = 90, 14
    panel_h = (height - top - margin - 2 * gap) / 3
    span = max(t[-1] - t[0], 1e-6)
    for k, (name, unit, a, b, labels) in enumerate(series):
        y0 = top + k * (panel_h + gap)
        box = (margin, y0, width - margin, y0 + panel_h)
        draw.rounded_rectangle(box, 12, fill=panel)
        draw.text((margin + 18, y0 + 12), name, font=font(15), fill=ink)
        draw.text((margin + 18, y0 + 32), unit, font=font(12), fill=muted)
        plot_l, plot_r = margin + 70, width - margin - 20
        plot_t, plot_b = y0 + 56, y0 + panel_h - 26
        values = np.array(a + (b or []), float)
        low, high = min(values.min(), 0.0), nice(max(values.max(), 1.0))
        if k == 1:
            low, high = min(low, -40.0), max(high, 40.0)
        if k == 2:
            bound = max(abs(low), abs(high), 10.0)
            low, high = -bound, bound

        def y(v, low=low, high=high, plot_t=plot_t, plot_b=plot_b):
            return plot_b - (v - low) / (high - low) * (plot_b - plot_t)

        for tick in sorted({low, 0.0, high}):
            draw.line((plot_l, y(tick), plot_r, y(tick)), fill=muted if tick == 0 else grid)
            draw.text((plot_l - 8, y(tick)), f"{tick:g}", font=font(11), fill=muted, anchor="rm")
        for start, row in enumerate(rows):
            if row["stopped"]:
                x = plot_l + (t[start] - t[0]) / span * (plot_r - plot_l)
                draw.line((x, plot_t, x, plot_b), fill=stop_color)
        for values_, color, label, offset in ((a, left_color, labels[0], 0),
                                              (b, right_color, labels[1], 110)):  # fmt: skip
            if values_ is None:
                continue
            points = [(plot_l + (ti - t[0]) / span * (plot_r - plot_l), y(v))
                      for ti, v in zip(t, values_)]  # fmt: skip
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            lx = plot_r - 220 + offset
            draw.rounded_rectangle((lx, y0 + 16, lx + 14, y0 + 30), 3, fill=color)
            draw.text((lx + 22, y0 + 23), label, font=font(12), fill=ink, anchor="lm")
        if k == 2:
            draw.text((plot_l, plot_b + 14), "0 s", font=font(11), fill=muted, anchor="lm")
            draw.text((plot_r, plot_b + 14), f"{t[-1]:.1f} s", font=font(11), fill=muted,
                      anchor="rm")  # fmt: skip
            draw.text((plot_l + 60, plot_b + 14), "orange lines: motors stopped by the PC watchdog",
                      font=font(11), fill=muted, anchor="lm")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a run recorded with --record")
    parser.add_argument("run", help="Directory written by --record")
    parser.add_argument("--out", help="PNG path (default: <run>/plot.png)")
    args = parser.parse_args(argv)
    print(f"Wrote {plot(args.run, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
