"""Virtual PC development target: QEMU with OVMF UEFI firmware, driven over QMP.

Stands in for a PC behind the capture card and Pico, so firmware setup and restarts
can be tested without real hardware. Every run starts from a fresh copy of the
firmware settings in a temporary folder, which is deleted on close.
"""
from __future__ import annotations

import json
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from brain.hid import EventCallback, HIDError

# shortcut: Debian/Ubuntu `ovmf` package paths; other distros install OVMF elsewhere.
OVMF_CODE = "/usr/share/OVMF/OVMF_CODE_4M.fd"
OVMF_VARS = "/usr/share/OVMF/OVMF_VARS_4M.fd"
REMOUNT_S = 0.3  # a restarted PC enumerates USB again early in POST
KEY_GAP_S = 0.03  # QEMU drops a repeated key pressed before its release

# QEMU key names that differ from ours lowercased (brain/keys.py and typed ASCII).
_QCODES = {"GUI": "meta_l", "ENTER": "ret", "SPACE": "spc", "PAGEUP": "pgup", "PAGEDOWN": "pgdn",
           " ": "spc", "\n": "ret", "\t": "tab", "-": "minus", "=": "equal", "[": "bracket_left",
           "]": "bracket_right", ";": "semicolon", "'": "apostrophe", "`": "grave_accent",
           "\\": "backslash", ",": "comma", ".": "dot", "/": "slash"}
_SHIFTED = dict(zip('!@#$%^&*()_+{}:"~|<>?', "1234567890-=[];'`\\,./"))


class QemuTarget:
    def __init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="exospace-qemu-"))
        vars_path, sock_path = self._dir / "vars.fd", self._dir / "qmp.sock"
        shutil.copy(OVMF_VARS, vars_path)
        self._proc = subprocess.Popen(
            ["qemu-system-x86_64", "-machine", "q35,accel=kvm:tcg", "-m", "256",
             "-drive", f"if=pflash,format=raw,readonly=on,file={OVMF_CODE}",
             "-drive", f"if=pflash,format=raw,file={vars_path}",
             "-display", "none", "-vga", "std", "-device", "qemu-xhci", "-device", "usb-tablet",
             "-qmp", f"unix:{sock_path},server=on,wait=off", "-net", "none",
             "-serial", "null", "-monitor", "none"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._sock = socket.socket(socket.AF_UNIX)
        self._lock = threading.Lock()
        self._replies: queue.Queue[dict[str, Any]] = queue.Queue()
        self._callbacks: list[EventCallback] = []
        self._mounted = True
        try:
            self._connect(sock_path)
        except BaseException:
            self.close()
            raise

    def grab(self):
        path = self._dir / "screen.png"
        self._qmp("screendump", {"filename": str(path), "format": "png"})
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def signal_present(self) -> bool:
        return self._proc.poll() is None

    def is_mounted(self) -> bool:
        return self._mounted and self._proc.poll() is None

    def on_event(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def send(self, cmd: dict[str, Any]) -> None:
        op = cmd["op"]
        if op in ("ping", "release_all"):
            return
        if not self.is_mounted():
            raise HIDError("usb send failed")  # like the Pico while the PC restarts
        if op == "key":
            self._keys([_QCODES.get(key, key.lower()) for key in cmd["keys"]])
        elif op == "type":
            for char in cmd["text"]:
                base = _SHIFTED.get(char, char.lower())
                self._keys((["shift"] if base != char else []) + [_QCODES.get(base, base)])
                time.sleep(KEY_GAP_S)
        elif op in ("click", "tap", "move"):
            self._pointer(cmd["x"], cmd["y"])
            for _ in range(0 if op == "move" else cmd.get("count", 1)):
                self._button(cmd.get("button", "left"))
        elif op == "scroll":
            for _ in range(abs(cmd["dy"])):
                self._button("wheel-down" if cmd["dy"] > 0 else "wheel-up")
        elif op == "swipe":
            self._pointer(cmd["x1"], cmd["y1"])
            self._qmp("input-send-event", {"events": [_btn("left", True)]})
            self._pointer(cmd["x2"], cmd["y2"])
            self._qmp("input-send-event", {"events": [_btn("left", False)]})
        else:
            raise HIDError(f"Unsupported QEMU input: {op}")

    def close(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._sock.close()
        shutil.rmtree(self._dir, ignore_errors=True)

    def _connect(self, sock_path: Path) -> None:
        deadline = time.monotonic() + 10
        while True:
            try:
                self._sock.connect(str(sock_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline or self._proc.poll() is not None:
                    raise HIDError("QEMU did not start; check qemu-system-x86_64 and KVM access") from None
                time.sleep(0.05)
        stream = self._sock.makefile("rb")
        stream.readline()  # greeting
        threading.Thread(target=self._read_loop, args=(stream,), name="qmp", daemon=True).start()
        self._qmp("qmp_capabilities")

    def _keys(self, qcodes: list[str]) -> None:
        self._qmp("send-key", {"keys": [{"type": "qcode", "data": code} for code in qcodes], "hold-time": 20})

    def _pointer(self, x: int, y: int) -> None:
        # usb-tablet axes span 0-32767, the same range as the HID contract.
        self._qmp("input-send-event", {"events": [{"type": "abs", "data": {"axis": "x", "value": x}},
                                                  {"type": "abs", "data": {"axis": "y", "value": y}}]})

    def _button(self, button: str) -> None:
        self._qmp("input-send-event", {"events": [_btn(button, True), _btn(button, False)]})

    def _qmp(self, execute: str, arguments: dict[str, Any] | None = None) -> Any:
        with self._lock:
            self._sock.sendall(json.dumps({"execute": execute, "arguments": arguments or {}}).encode())
            try:
                reply = self._replies.get(timeout=5)
            except queue.Empty:
                raise HIDError(f"QEMU did not answer {execute}") from None
        if "error" in reply:
            raise HIDError(f"{execute} failed: {reply['error'].get('desc', 'unknown error')}")
        return reply.get("return")

    def _read_loop(self, stream) -> None:
        for line in stream:
            message = json.loads(line)
            if "event" not in message:
                self._replies.put(message)
            elif message["event"] == "RESET":
                # A real restart unmounts the Pico until the firmware enumerates USB again.
                self._set_mounted(False)
                threading.Timer(REMOUNT_S, self._set_mounted, (True,)).start()

    def _set_mounted(self, mounted: bool) -> None:
        for callback in self._callbacks:  # first: a send refused for unmount must find the run cancelled
            callback("mounted" if mounted else "unmounted")
        self._mounted = mounted


def _btn(button: str, down: bool) -> dict[str, Any]:
    return {"type": "btn", "data": {"down": down, "button": button}}
