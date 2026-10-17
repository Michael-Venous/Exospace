"""Typed-goal development interface for the screenshot-driven agent."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from contextlib import ExitStack
from typing import Callable

from brain.agent.executor import Executor
from brain.agent.factory import build_executor
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, ScreenCapture, V4L2Capture, active_rect, is_blank
from brain.config import Config
from brain.controller import Controller
from brain.device import detect_profile
from brain.errors import describe
from brain.display.server import Command, DisplayServer
from brain.events import EventBus
from brain.hid import HID, HIDError, LocalHID, PicoHID
from brain.imaging import prepare
from brain.monitor import POLL_S, ConnectionMonitor
from brain.telemetry import Telemetry
from brain.types import Action, ActionError, DeviceProfile

LIVE_FPS = 4  # capture-card frames pushed to the display between steps
RESUME_S = 10.0  # a blanked screen on the same machine (sleep, mode switch) gets this long to return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "pi", "browser"], default="pi")
    parser.add_argument("--url", help="Browser target URL (isolated profile)")
    parser.add_argument("--headed", action="store_true", help="Show browser target")
    parser.add_argument("--os", choices=["macos", "windows", "linux", "android"])
    parser.add_argument("--task-id", default="interactive")
    parser.add_argument("--config-name", default="jev-flash")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--display", type=int, metavar="PORT",
                        help="Serve the touch display page on this port (e.g. 8770)")
    parser.add_argument("--display-host", default="127.0.0.1")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    cfg = Config.load()
    token = CancelToken()
    status = lambda message: print(message, flush=True)
    if not cfg.gemini_api_key:
        status("Set GEMINI_API_KEY in the project .env before starting (see .env.example).")
        return 1
    try:
        with ExitStack() as cleanup:
            executor = build_executor(cfg, token, cleanup, status)
            if args.backend == "pi":
                capture = V4L2Capture(cfg.capture_device, cfg.capture_width, cfg.capture_height)
                cleanup.callback(capture.close)
                hid = PicoHID(cfg.pico_serial, cfg.pico_baud)
            elif args.backend == "browser":
                from brain.browser import BrowserTarget
                if not args.url:
                    parser.error("--backend browser requires --url")
                capture = hid = BrowserTarget(args.url, headed=args.headed)
            else:
                capture = ScreenCapture()
                cleanup.callback(capture.close)
                hid = LocalHID()
            cleanup.callback(hid.close)
            controller = Controller(hid, capture, token)
            cleanup.callback(controller.release_all)
            monitor = ConnectionMonitor(capture, hid, token.cancel)
            cleanup.callback(monitor.close)
            bus = EventBus() if args.display else None
            commands: queue.Queue[Command] = queue.Queue()
            if bus is not None:
                server = DisplayServer(bus, commands, token.cancel, args.display_host, args.display)
                server.start()
                cleanup.callback(server.close)
                bus.publish("config", min_presence=cfg.jev_min_presence, min_confidence=cfg.jev_min_confidence)
                if args.backend == "pi":
                    _pump_live_frames(capture, bus)
                else:
                    _show_screen(capture, bus)
                threading.Thread(target=_read_stdin, args=(commands,), name="stdin", daemon=True).start()
                status(f"Display: {server.url}  (goals from the page or typed here)")
            agent = Agent(executor, controller, Memory.load(cfg.session_path),
                          Telemetry(cfg.bench_csv, args.config_name), args.max_steps, status, bus,
                          resume=lambda: _await_video(monitor, capture, hid, token))
            profile = None
            generation = -1
            previous_goal = ""
            correction = False
            pointed: tuple[float, float] | None = None

            status(f"Executor: {cfg.executor}; Gemini: {cfg.gemini_model}. Ctrl+C interrupts; idle Ctrl+C exits.")
            while True:
                try:
                    if bus is None:
                        command = Command("goal", text=input("Correction (or /new GOAL)> " if correction else "Goal> "))
                    else:
                        command = _next_command(commands)
                except (EOFError, KeyboardInterrupt):
                    status("Goodbye.")
                    return 0
                if command.kind == "point":
                    pointed = (command.x, command.y)
                    continue
                if command.kind == "click":
                    try:
                        profile, generation = _connected_profile(
                            monitor, capture, executor, args.os, token, profile, generation, status)
                        controller.execute(Action("click", x=command.x, y=command.y), profile)
                        controller.wait_settled()
                        if bus is not None:
                            bus.publish("tap", x=command.x, y=command.y)
                            if args.backend != "pi":
                                _show_screen(capture, bus)
                    except (Cancelled, ActionError, HIDError) as exc:
                        controller.release_all()
                        status(f"Tap not sent ({type(exc).__name__}).")
                    continue
                line = command.text.strip()
                if not line:
                    continue
                if pointed is not None:
                    line += (f"\nThe user tapped the screen to mean 'this': x={pointed[0]:.3f}, y={pointed[1]:.3f} "
                             "(0-1 within the screen, origin top-left).")
                    pointed = None
                if line.startswith("/new "):
                    goal = line[5:].strip()
                elif correction:
                    goal = previous_goal + "\nUser correction: " + line
                else:
                    goal = line
                if not goal:
                    continue
                try:
                    profile, generation = _connected_profile(
                        monitor, capture, executor, args.os, token, profile, generation, status)
                    if args.backend == "local":
                        status("Focus the target window; starting in 2 seconds.")
                        token.sleep(2)
                    result = agent.run(goal, profile, args.task_id)
                    status(f"{result.status}: {result.message}")
                    previous_goal = goal
                    correction = result.status in {"cancelled", "ask_user"}
                except (Cancelled, KeyboardInterrupt):
                    token.cancel("barge-in")
                    controller.release_all()
                    previous_goal, correction = goal, True
                    status("Interrupted. Enter a correction or /new GOAL.")
    except Exception as exc:
        status(f"Setup failed: {describe(exc, 'setup')}")
        return 1


def _next_command(commands: queue.Queue[Command]) -> Command:
    """Block for the next goal or tap; the timeout keeps Ctrl+C responsive."""
    while True:
        try:
            return commands.get(timeout=0.5)
        except queue.Empty:
            continue


def _read_stdin(commands: queue.Queue[Command]) -> None:
    """With a display, typed goals still work; they share the page's queue."""
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return
        commands.put(Command("goal", text=line))


def _show_screen(capture: Capture, bus: EventBus) -> None:
    """Publish one frame. Call from the main thread: browser/mss backends aren't thread-safe."""
    frame = capture.grab()
    if frame is not None:
        bus.set_frame(prepare(frame, active_rect(frame), grid=False))


def _pump_live_frames(capture: Capture, bus: EventBus) -> None:
    """The capture card is read on its own thread, so it can feed the display continuously."""
    def pump() -> None:
        while True:
            _show_screen(capture, bus)
            time.sleep(1 / LIVE_FPS)
    threading.Thread(target=pump, name="display-frames", daemon=True).start()


def _await_video(monitor: ConnectionMonitor, capture: Capture, hid: HID, token: CancelToken,
                 timeout_s: float = RESUME_S) -> None:
    """After the screen blanked mid-run, wait for live video on the same machine. An unplug
    ends the run instead: the next device may be another one, so the user gives its goal."""
    token.reset()  # a stop pressed while waiting still cancels
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not hid.is_mounted():
            raise Cancelled("target disconnected")
        frame = capture.grab()
        if monitor.connected and frame is not None and not is_blank(frame):
            return
        token.sleep(POLL_S)
    raise Cancelled("video did not come back")


def _connected_profile(monitor: ConnectionMonitor, capture: Capture, executor: Executor,
                       os_override: str | None, token: CancelToken, profile: DeviceProfile | None,
                       generation: int, status: Callable[[str], None]) -> tuple[DeviceProfile, int]:
    """Wait for a target and re-detect it if it changed. A disconnect during
    detection retries instead of dropping the user's pending goal."""
    while True:
        if not monitor.connected:
            status("Waiting for target…")
        current = monitor.wait_connected()
        token.reset()
        if profile is not None and current == generation:
            return profile, generation
        try:
            detected = detect_profile(capture, executor, os_override, cancel=token)
        except Cancelled:
            if monitor.connected:
                raise  # a real interruption, not an unplug
            continue
        if monitor.connected and monitor.generation == current:
            status(f"Target: {detected.os} ({detected.input})")
            return detected, current
        status("Target changed during detection; retrying.")



if __name__ == "__main__":
    raise SystemExit(main())
