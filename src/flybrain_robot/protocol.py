"""Validated JSON datagrams. UDP provides no delivery or authentication guarantee."""

import json
import math


class PacketError(ValueError):
    pass


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate(packet):
    if not isinstance(packet, dict):
        raise PacketError("Expected an object")
    seq = packet.get("sequence")
    if type(seq) is not int or not 0 <= seq <= 2147483647:
        raise PacketError("Invalid sequence")
    kind = packet.get("type")
    if kind == "motor_command":
        keys = ("left", "right")
        if type(packet.get("emergency_stop")) is not bool:
            raise PacketError("Missing boolean emergency_stop")
    elif kind == "telemetry":
        keys = ("left_speed", "right_speed")
        for key in ("gyro", "accel"):
            values = packet.get(key)
            if not isinstance(values, list) or len(values) != 3 or not all(map(number, values)):
                raise PacketError("Invalid IMU vector")
    else:
        raise PacketError("Unknown message type")
    for key in keys:
        value = packet.get(key)
        if not number(value) or not -100 <= value <= 100:
            raise PacketError("Invalid motor speed")
    return packet


def encode(packet):
    return json.dumps(validate(packet), allow_nan=False).encode("utf-8")


def decode(data):
    if len(data) > 2048:
        raise PacketError("Oversized datagram")
    try:
        return validate(json.loads(data))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PacketError(str(exc)) from exc


def command(sequence, left=0, right=0, emergency_stop=False):
    return validate(
        dict(
            type="motor_command",
            sequence=sequence,
            left=left,
            right=right,
            emergency_stop=emergency_stop,
        )
    )
