"""Nonblocking UDP transport; malformed or out-of-order telemetry is ignored."""

import socket
import time

from .protocol import PacketError, decode, encode
from .telemetry import Telemetry


class UDPBridge:
    def __init__(self, config):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((config.pc_host, config.pc_port))
        self.socket.setblocking(False)
        self.target = (socket.gethostbyname(config.robot_ip), config.robot_port)
        self.last_sequence = -1
        self.last_seen = None
        self.telemetry = Telemetry()
        self.invalid_packets = 0

    def send(self, packet):
        self.socket.sendto(encode(packet), self.target)

    def receive(self):
        # Bound work per tick so packet floods cannot starve the control loop.
        for _ in range(32):
            try:
                data, peer = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            if peer != self.target:
                continue
            try:
                packet = decode(data)
                if packet["type"] != "telemetry":
                    continue
                if packet["sequence"] <= self.last_sequence:
                    continue
                self.last_sequence = packet["sequence"]
                self.last_seen = time.monotonic()
                self.telemetry = Telemetry(tuple(packet["gyro"]), tuple(packet["accel"]))
            except PacketError:
                self.invalid_packets += 1
        return self.telemetry

    def fresh(self):
        return self.last_seen is not None and time.monotonic() - self.last_seen < 0.5

    def close(self):
        self.socket.close()
