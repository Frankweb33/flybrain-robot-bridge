"""Headless CLI. Network transmission requires --send; dry-run is the default."""

import argparse
import logging
import time

import cv2
import yaml

from .brain.malecns import MaleCNSBackend
from .brain.mock import MockBrain
from .config import load_config
from .controller import UDPBridge
from .motor_decoder import MotorDecoder
from .protocol import command
from .telemetry import synthetic_telemetry
from .vision import VisionEncoder, synthetic_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "malecns"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int)
    source.add_argument("--video")
    source.add_argument("--synthetic", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--send", action="store_true", help="Enable physical UDP motor commands")
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int, default=300, help="Finite frame count (default: 300)")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    bridge, capture, decoder, sequence = None, None, None, 0
    try:
        cfg = load_config(args.config)
        logging.basicConfig(level=cfg.logging_level)
        backend = args.backend or cfg.backend
        brain = MockBrain() if backend == "mock" else MaleCNSBackend(cfg.dataset_path)
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
            "synthetic"
            if args.synthetic
            else (video if video else (args.camera if args.camera is not None else cfg.camera))
        )
        if not args.synthetic:
            capture = cv2.VideoCapture(origin)
            if not capture.isOpened():
                raise ValueError(f"Cannot open image source: {origin}")
        if args.send:
            bridge = UDPBridge(cfg)
        print(f"source={origin} backend={backend} UDP={'enabled' if bridge else 'off (dry-run)'}")
        previous = time.monotonic()
        for index in range(args.steps):
            started = time.monotonic()
            if capture is None:
                frame = synthetic_frame(index)
            else:
                ok, frame = capture.read()
                if not ok:
                    break
            sensory = encoder.encode(frame)
            telemetry = bridge.receive() if bridge else synthetic_telemetry(index / 30)
            now = time.monotonic()
            dt = 1 / 30 if args.synthetic else max(0.001, min(now - previous, 0.5))
            previous = now
            activity = brain.step(sensory, telemetry, dt)
            stopped = bridge is not None and not bridge.fresh()
            left, right = decoder.update(activity, emergency_stop=stopped)
            sequence += 1
            if bridge:
                bridge.send(command(sequence, left, right, stopped))
            if index % 10 == 0:
                status = (
                    "fresh" if bridge and bridge.fresh() else ("stale" if bridge else "synthetic")
                )
                print(
                    f"frame={index:04d} Hz={1 / dt:.1f} motion={sensory.left_motion:.2f}/"
                    f"{sensory.right_motion:.2f} looming={sensory.looming:.2f} IMU={status} "
                    f"motor={activity.left:.2f}/{activity.right:.2f} "
                    f"command={left:+d}/{right:+d} watchdog={'STOP' if stopped else 'ready'}"
                )
            time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
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
