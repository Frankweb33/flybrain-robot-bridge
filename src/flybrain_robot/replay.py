"""Export a closed-loop run for the browser replay (web/replay.html).

    python -m flybrain_robot.replay --data data/malecns-v1.0 --scene tour --out web/runs/tour.json

Every frame stores what the page shows and nothing it does not: a 160x120 JPEG of the
camera frame, the encoder features, smoothed firing rates of the named neuron groups,
the most active neurons of the whole brain, and the motor commands.
"""

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .brain.malecns import MaleCNSBackend
from .motor_decoder import MotorDecoder
from .telemetry import Telemetry
from .vision import PATTERNS, SCENES, TOUR_SEGMENT, scene_frame

TOP_NEURONS = 30


def thumbnail(frame):
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def export(data, scene="tour", frames=None, wiring="connectome", seed=0, device="auto",
           image_every=2):
    backend = MaleCNSBackend(data, seed=seed, wiring=wiring, device=device)
    graph = backend.graph
    frames = frames or (TOUR_SEGMENT * len(SCENES) if scene == "tour" else TOUR_SEGMENT)
    encoder, decoder = backend.make_encoder(), MotorDecoder()
    driven = np.zeros(graph.size, bool)
    for cells in backend.populations.values():
        driven[cells] = True
    neurons, neuron_row, out = [], {}, []
    for i in range(frames):
        name = SCENES[(i // TOUR_SEGMENT) % len(SCENES)] if scene == "tour" else scene
        if scene == "tour" and i % TOUR_SEGMENT == 0:
            backend.reset()
            encoder.reset()
        frame = scene_frame(scene, i)
        sensory = encoder.encode(frame)
        activity = backend.step(sensory, Telemetry(), 1 / 30)
        left, right = decoder.update(activity, now=i / 30)
        top = np.argsort(backend.rates)[::-1][:TOP_NEURONS]
        active = []
        for index in top:
            if backend.rates[index] < 1:
                break
            if int(index) not in neuron_row:
                neuron_row[int(index)] = len(neurons)
                neurons.append(dict(
                    body_id=int(graph.body_id[index]), type=str(graph.type[index]),
                    instance=str(graph.instance[index]), side=str(graph.side[index]),
                    superclass=str(graph.superclass[index]), driven=bool(driven[index]),
                ))  # fmt: skip
            active.append([neuron_row[int(index)], round(float(backend.rates[index]), 1)])
        out.append(dict(
            t=round(i / 30, 3), scene=name, image=thumbnail(frame) if i % image_every == 0 else "",
            features=dict(looming=[round(sensory.looming_left, 3), round(sensory.looming_right, 3)],
                          object=[round(sensory.object_left, 3), round(sensory.object_right, 3)]),
            groups={k: round(v, 1) for k, v in backend.named_rates().items()},
            active=active, spikes=int(backend.last_counts.sum()),
            motor=dict(left=round(activity.left, 3), right=round(activity.right, 3),
                       escape=round(activity.escape, 3), command=[int(left), int(right)]),
        ))  # fmt: skip
    return dict(
        meta=dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  commit=commit(), scene=scene, wiring=wiring, seed=seed, device=backend.device,
                  neurons=int(graph.size), frame_period_s=1 / 30,
                  dataset=graph.manifest.get("dataset"), license=graph.manifest.get("license"),
                  attribution=graph.manifest.get("attribution")),
        neurons=neurons, frames=out,
    )  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/malecns-v1.0")
    parser.add_argument("--scene", choices=("tour",) + SCENES + PATTERNS, default="tour")
    parser.add_argument("--frames", type=int, help="Default: 540 for the tour, else 90")
    parser.add_argument("--wiring", choices=["connectome", "shuffled"], default="connectome")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", default="web/runs/tour.json")
    parser.add_argument("--image-every", type=int, default=2, help="Camera thumbnail every Nth frame")
    args = parser.parse_args(argv)
    run = export(args.data, args.scene, args.frames, args.wiring, device=args.device,
                 image_every=args.image_every)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size // 1024} KiB, {len(run['frames'])} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
