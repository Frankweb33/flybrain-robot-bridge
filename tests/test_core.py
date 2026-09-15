import importlib
import json
import pkgutil
import socket
import subprocess
import sys

import numpy as np
import pytest

import flybrain_robot
from flybrain_robot.brain.base import MotorActivity
from flybrain_robot.brain.mock import MockBrain
from flybrain_robot.config import Config
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.protocol import PacketError, command, decode, encode
from flybrain_robot.telemetry import Telemetry
from flybrain_robot.vision import TOUR_SEGMENT, SensoryInput, VisionEncoder, scene_frame


def test_packet_roundtrip():
    packet = command(42, 35, -28)
    assert decode(encode(packet)) == packet
    telemetry = dict(
        type="telemetry", sequence=1, gyro=[0, 0, 0], accel=[0, 0, 1], left_speed=0, right_speed=0
    )
    assert decode(encode(telemetry)) == telemetry


@pytest.mark.parametrize(
    "payload",
    [
        b"bad json",
        b"[]",
        b"null",
        b"\xff",
        b"{}",
        json.dumps(
            dict(type="motor_command", sequence=True, left=0, right=0, emergency_stop=False)
        ).encode(),
        json.dumps(
            dict(type="motor_command", sequence=1, left=101, right=0, emergency_stop=False)
        ).encode(),
        json.dumps(
            dict(type="motor_command", sequence=1, left=float("nan"), right=0, emergency_stop=False)
        ).encode(),
        json.dumps(
            dict(
                type="telemetry", sequence=1, gyro=[0], accel=[0, 0, 1], left_speed=0, right_speed=0
            )
        ).encode(),
    ],
)
def test_bad_packets(payload):
    with pytest.raises(PacketError):
        decode(payload)


def test_speed_clamp_and_inversion():
    decoder = MotorDecoder(max_speed=30, smoothing=1, invert_left=True)
    assert decoder.update(MotorActivity(10, -10)) == (-30, -30)


def test_dead_zone_and_smoothing():
    decoder = MotorDecoder(max_speed=40, smoothing=0.5, dead_zone=5)
    assert decoder.update(MotorActivity(0.1, 0.1)) == (0, 0)
    assert decoder.update(MotorActivity(1, 1)) == (20, 20)
    assert decoder.update(MotorActivity(1, 1)) == (30, 30)


def test_escape_and_emergency():
    decoder = MotorDecoder(smoothing=1)
    assert decoder.update(MotorActivity(1, 1, 0.9)) == (-35, -35)
    assert decoder.update(MotorActivity(1, 1), emergency_stop=True) == (0, 0)
    assert decoder.read() == (0, 0)


def test_watchdog():
    decoder = MotorDecoder(smoothing=1)
    assert decoder.read(now=0) == (0, 0)
    decoder.update(MotorActivity(1, 1), now=10)
    assert decoder.read(now=10.499) == (35, 35)
    assert decoder.read(now=10.5) == (0, 0)


def test_deterministic_mock_and_feedback():
    a, b = MockBrain(), MockBrain()
    sensory = SensoryInput(0.1, 0.8, 0.2)
    for _ in range(10):
        a.step(sensory, Telemetry(), 1 / 30)
        b.step(sensory, Telemetry(), 1 / 30)
    np.testing.assert_array_equal(a.state, b.state)
    assert a.get_motor_activity().left > a.get_motor_activity().right
    before = a.get_motor_activity()
    a.step(sensory, Telemetry(gyro=(0, 0, 90)), 0.5)  # robot yawing left
    after = a.get_motor_activity()
    assert after.right < before.right  # counter-steer right...
    assert after.left >= before.left - 1e-9  # ...without slowing the left side
    a.reset()
    assert not a.state.any()


def scene_features(name):
    encoder = VisionEncoder()
    assert encoder.encode(scene_frame(name, 0)) == SensoryInput()
    return [encoder.encode(scene_frame(name, i)) for i in range(1, TOUR_SEGMENT)]


def peak(rows, field):
    return max(getattr(r, field) for r in rows)


def test_vision_separates_looming_from_translation_and_rotation():
    # The original encoder reported sideways motion as looming and missed approaches.
    for name in ("static", "pan", "pass_left"):
        rows = scene_features(name)
        assert peak(rows, "looming_left") < 0.1 and peak(rows, "looming_right") < 0.1, name
    assert peak(scene_features("pan"), "object_left") < 0.1  # whole-view motion is no object
    assert scene_features("pan")[-1].yaw_flow < -0.2
    passing = scene_features("pass_left")
    assert peak(passing, "object_left") > 0.4 and peak(passing, "object_right") < 0.1
    left, right = scene_features("approach_left"), scene_features("approach_right")
    assert peak(left, "looming_left") > 0.5 and peak(left, "looming_right") < 0.1
    assert peak(right, "looming_right") > 0.5 and peak(right, "looming_left") < 0.1
    center = scene_features("approach_center")
    assert min(peak(center, "looming_left"), peak(center, "looming_right")) > 0.3


def test_no_camera_or_socket_in_synthetic(monkeypatch, capsys):
    from flybrain_robot.main import main

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic dry-run touched camera or network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("cv2.VideoCapture", forbidden)
    main(["--synthetic", "--dry-run", "--steps", "2"])
    assert "UDP=off (dry-run)" in capsys.readouterr().out


def test_cli_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "flybrain_robot.main", "--synthetic", "--dry-run", "--steps", "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "motor=" in result.stdout


def test_imports():
    for module in pkgutil.walk_packages(flybrain_robot.__path__, flybrain_robot.__name__ + "."):
        importlib.import_module(module.name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_speed": 101},
        {"watchdog_timeout": 1},
        {"smoothing": float("nan")},
        {"invert_left": "false"},
    ],
)
def test_config_bounds(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_extreme_numeric_packet():
    packet = command(1)
    packet["left"] = 10**400
    with pytest.raises(PacketError):
        decode(json.dumps(packet).encode())


def test_udp_ignores_invalid_and_reordered(monkeypatch):
    from flybrain_robot.controller import UDPBridge

    packet = dict(
        type="telemetry", sequence=2, gyro=[1, 2, 3], accel=[0, 0, 1], left_speed=0, right_speed=0
    )
    old_packet = dict(packet, sequence=1, gyro=[9, 9, 9])
    queue = [b"broken", encode(packet), encode(old_packet)]

    class FakeSocket:
        def bind(self, address):
            pass

        def setblocking(self, flag):
            pass

        def recvfrom(self, size):
            if queue:
                return queue.pop(0), ("127.0.0.1", 9000)
            raise BlockingIOError()

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    bridge = UDPBridge(Config(robot_ip="127.0.0.1"))
    assert bridge.receive().gyro == (1, 2, 3)
    assert bridge.invalid_packets == 1
    assert bridge.fresh()
    bridge.last_seen -= 1
    assert not bridge.fresh()
    bridge.close()
