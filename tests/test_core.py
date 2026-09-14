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
from flybrain_robot.brain.malecns import MaleCNSBackend
from flybrain_robot.brain.mock import MockBrain
from flybrain_robot.config import Config
from flybrain_robot.motor_decoder import MotorDecoder
from flybrain_robot.protocol import PacketError, command, decode, encode
from flybrain_robot.telemetry import Telemetry
from flybrain_robot.vision import SensoryInput, VisionEncoder, synthetic_frame


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
    old_left = a.get_motor_activity().left
    a.step(sensory, Telemetry(gyro=(0, 0, 90)), 0.5)
    assert a.get_motor_activity().left < old_left
    a.reset()
    assert not a.state.any()


def test_vision_detects_expansion():
    encoder = VisionEncoder()
    assert encoder.encode(synthetic_frame(0)) == SensoryInput()
    results = [encoder.encode(synthetic_frame(i)) for i in range(1, 60)]
    assert max(r.left_motion for r in results) > 0.01
    assert max(r.looming for r in results) > 0.01


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


def test_malecns_is_explicit_stub(tmp_path):
    with pytest.raises(ValueError, match="Obtain data"):
        MaleCNSBackend()
    with pytest.raises(NotImplementedError, match="Not implemented yet"):
        MaleCNSBackend(tmp_path)


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
