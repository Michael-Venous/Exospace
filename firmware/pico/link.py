"""Newline-delimited JSON over the Pi UART (PLAN.md §5.1): GP0 = TX, GP1 = RX."""

import json

import board
import busio

BAUD = 115200
MAX_LINE_BYTES = 512


class Link:
    def __init__(self):
        self._uart = busio.UART(board.GP0, board.GP1, baudrate=BAUD, timeout=0)
        self._buffer = bytearray()

    def pending(self) -> bool:
        """True if the Pi has sent anything we have not read yet."""
        return self._uart.in_waiting > 0

    def poll(self) -> list:
        """Return the commands received so far. Garbled lines become {}."""
        data = self._uart.read(self._uart.in_waiting)
        if data:
            self._buffer.extend(data)
        *lines, rest = bytes(self._buffer).split(b"\n")
        self._buffer = bytearray(rest) if len(rest) <= MAX_LINE_BYTES else bytearray()
        return [self._parse(line) for line in lines if line.strip()]

    def send(self, message: dict) -> None:
        self._uart.write(json.dumps(message).encode() + b"\n")

    @staticmethod
    def _parse(line: bytes) -> dict:
        try:
            cmd = json.loads(line)
        except ValueError:
            return {}
        return cmd if isinstance(cmd, dict) else {}
