"""Headless CLI. Network transmission requires --send; dry-run is the default."""

import argparse
import logging
import time

import cv2
import yaml

from .brain.mock import MockBrain
from .config import load_config
from .controller import UDPBridge
from .motor_decoder import MotorDecoder
from .protocol import command
from .telemetry import synthetic_telemetry
from .vision import SCENES, VisionEncoder, scene_frame

FRAME_PERIOD = 1 / 30


def build_brain(backend, cfg, args):
    if backend == "mock":
        return MockBrain()
    from .brain.malecns import MaleCNSBackend

    return MaleCNSBackend(
        args.data or cfg.dataset_path,
        seed=cfg.brain_seed,
        cruise=cfg.cruise,
        fast_modulators=cfg.fast_modulators,
        wiring=args.wiring or cfg.wiring,
    )


def status_line(index, sensory, activity, brain, left, right, imu, stopped, realtime):
    line = (
        f"frame={index:04d} looming={sensory.looming_left:.2f}/{sensory.looming_right:.2f} "
        f"object={sensory.object_left:.2f}/{sensory.object_right:.2f} IMU={imu} "
        f"motor={activity.left:.2f}/{activity.right:.2f} escape={activity.escape:.2f} "
        f"command={left:+d}/{right:+d} watchdog={'STOP' if stopped else 'ready'}"
    )
    if hasattr(brain, "named_rates"):
        r = brain.named_rates()
        line += (
            f" | GF={r['DNp01:L']:.0f}/{r['DNp01:R']:.0f}Hz "
            f"DNa02={r['DNa02:L']:.0f}/{r['DNa02:R']:.0f}Hz brain_x_realtime={realtime:.2f}"
        )
    return line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "malecns"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int)
    source.add_argument("--video")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--scene", choices=("tour",) + SCENES, default="tour",
                        help="Synthetic scene (default: tour through all of them)")  # fmt: skip
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--send", action="store_true", help="Enable physical UDP motor commands")
    parser.add_argument("--config")
    parser.add_argument("--data", help="MaleCNS dataset directory (overrides dataset_path)")
    parser.add_argument("--wiring", choices=["connectome", "shuffled"],
                        help="shuffled = control with permuted synapse targets")  # fmt: skip
    parser.add_argument("--steps", type=int, default=300, help="Finite frame count (default: 300)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    bridge, capture, decoder, sequence = None, None, None, 0
    try:
        cfg = load_config(args.config)
        logging.basicConfig(level=cfg.logging_level)
        backend = args.backend or cfg.backend
        brain = build_brain(backend, cfg, args)
        decoder = MotorDecoder(
            **{
                name: getattr(cfg, name)
                for name in (
                    "max_speed",
                    "smoothing",
                    "dead_zone",
                    "looming_threshold",
                    "watchdog_timeout",
                    "invert_left",
                    "invert_right",
                )
            }
        )
        encoder = VisionEncoder()
        video = args.video or (cfg.video if args.camera is None else None)
        origin = (
            f"synthetic:{args.scene}"
            if args.synthetic
            else (video if video else (args.camera if args.camera is not None else cfg.camera))
        )
        if not args.synthetic:
            capture = cv2.VideoCapture(origin)
            if not capture.isOpened():
                raise ValueError(f"Cannot open image source: {origin}")
        # The connectome advances a fixed simulated time per frame (the frame period),
        # even when it computes slower than real time; the status line reports the ratio.
        fixed_step = args.synthetic or backend == "malecns"
        period = FRAME_PERIOD
        if video and capture is not None and capture.get(cv2.CAP_PROP_FPS) > 0:
            period = 1 / capture.get(cv2.CAP_PROP_FPS)
        if args.send:
            bridge = UDPBridge(cfg)
        neurons = f" neurons={brain.neuron_count}" if hasattr(brain, "neuron_count") else ""
        print(
            f"source={origin} backend={backend}{neurons} "
            f"UDP={'enabled' if bridge else 'off (dry-run)'}"
        )
        previous = time.monotonic()
        for index in range(args.steps):
            started = time.monotonic()
            if capture is None:
                frame = scene_frame(args.scene, index)
            else:
                ok, frame = capture.read()
                if not ok:
                    break
            sensory = encoder.encode(frame)
            telemetry = bridge.receive() if bridge else synthetic_telemetry(index * period)
            now = time.monotonic()
            dt = period if fixed_step else max(0.001, min(now - previous, 0.5))
            previous = now
            brain_started = time.perf_counter()
            activity = brain.step(sensory, telemetry, dt)
            realtime = dt / max(time.perf_counter() - brain_started, 1e-9)
            stopped = bridge is not None and not bridge.fresh()
            left, right = decoder.update(activity, emergency_stop=stopped)
            sequence += 1
            if bridge:
                bridge.send(command(sequence, left, right, stopped))
            if index % 10 == 0:
                imu = "fresh" if bridge and bridge.fresh() else ("stale" if bridge else "synthetic")
                print(status_line(index, sensory, activity, brain, left, right, imu, stopped,
                                  realtime))  # fmt: skip
            time.sleep(max(0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Stopped by user.")
    except (ValueError, OSError, NotImplementedError, cv2.error, yaml.YAMLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if decoder:
            decoder.stop()
        if bridge:
            try:
                bridge.send(command(sequence + 1, emergency_stop=True))
            except OSError:
                logging.warning("Final stop datagram failed; robot watchdog must stop motors")
            finally:
                bridge.close()
        if capture:
            capture.release()


if __name__ == "__main__":
    main()
