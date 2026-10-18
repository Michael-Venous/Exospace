"""Newline-delimited JSON command channels (PLAN.md §5.1).

The Pi talks over the UART (GP0 = TX, GP1 = RX). In maintenance mode a computer the Pico is
plugged into can send the same commands over the USB data port, to drive itself for testing."""

import json

import board
import busio
import usb_cdc

BAUD = 115200
MAX_LINE_BYTES = 512


class Link:
    def __init__(self, port):
        self._port = port
        self._buffer = bytearray()

    def pending(self) -> bool:
        """True if the other end has sent anything we have not read yet."""
        return self._port.in_waiting > 0

    def poll(self) -> list:
        """Return the commands received so far. Garbled lines become {}."""
        data = self._port.read(self._port.in_waiting)
        if data:
            self._buffer.extend(data)
        *lines, rest = bytes(self._buffer).split(b"\n")
        self._buffer = bytearray(rest) if len(rest) <= MAX_LINE_BYTES else bytearray()
        return [self._parse(line) for line in lines if line.strip()]

    def send(self, message: dict) -> None:
        self._port.write(json.dumps(message).encode() + b"\n")

    @staticmethod
    def _parse(line: bytes) -> dict:
        try:
            cmd = json.loads(line)
        except ValueError:
            return {}
        return cmd if isinstance(cmd, dict) else {}


def open_links() -> list:
    """The UART, plus the USB data port when boot.py enabled it."""
    links = [Link(busio.UART(board.GP0, board.GP1, baudrate=BAUD, timeout=0))]
    if usb_cdc.data is not None:
        usb_cdc.data.timeout = 0
        usb_cdc.data.write_timeout = 0.1  # nobody reading the port must not stall the UART
        links.append(Link(usb_cdc.data))
    return links
