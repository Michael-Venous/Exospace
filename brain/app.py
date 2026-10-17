"""Typed-goal development interface for the screenshot-driven agent."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import replace
from typing import Callable, cast

from brain.agent.executor import Executor
from brain.agent.factory import build_executor
from brain.agent.jev_only import DEADLINE_S
from brain.agent.loop import RESUMABLE, Agent
from brain.agent.memory import Memory
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, ScreenCapture, V4L2Capture, active_rect
from brain.config import Config
from brain.controller import Controller
from brain.device import detect_profile
from brain.errors import describe
from brain.display.server import Command, DisplayServer
from brain.events import EventBus
from brain.hid import HID, HIDError, LocalHID, PicoHID
from brain.imaging import prepare
from brain.monitor import ConnectionMonitor
from brain.telemetry import Telemetry
from brain.types import OS_NAMES, Action, ActionError, DeviceProfile, OSName

LIVE_FPS = 4  # capture-card frames pushed to the display between steps
RECONNECT_S = 180.0  # restarting a PC into firmware setup can take minutes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "pi", "browser", "qemu", "android"], default="pi")
    parser.add_argument("--url", help="Browser target URL (isolated profile)")
    parser.add_argument("--headed", action="store_true", help="Show browser target")
    parser.add_argument("--os", choices=[os for os in OS_NAMES if os != "unknown"])
    parser.add_argument("--task-id", default="interactive")
    parser.add_argument("--config-name", help="Benchmark label (default: the executor)")
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
    home_os = cast(OSName | None, cfg.target_os or None)
    if home_os not in (None, *OS_NAMES) or home_os in ("unknown", "bios"):
        parser.error("TARGET_OS must be one of windows, macos, linux, chromeos, android, ios")
    if cfg.executor != "jev" and not (cfg.gemini_api_key or cfg.uses_zai):
        status("Set ZAI_API_KEY or GEMINI_API_KEY in the project .env (see .env.example); either one works.")
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
            elif args.backend == "qemu":
                from brain.qemu import QemuTarget
                capture = hid = QemuTarget()
            elif args.backend == "android":
                from brain.android import AndroidTarget
                capture = hid = AndroidTarget()
            else:
                capture = ScreenCapture()
                cleanup.callback(capture.close)
                hid = LocalHID(PicoHID(cfg.local_keyboard, cfg.pico_baud) if cfg.local_keyboard else None)
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
            profile: DeviceProfile | None = None
            generation = -1

            def redetect(new_connection: bool = False) -> DeviceProfile | None:
                """Classify the screen again; None keeps the OS when it cannot tell. The same
                connection keeps its connected_at, which keys executor state (PLAN.md §5.4)."""
                nonlocal profile
                try:
                    found = detect_profile(capture, executor, args.os, home_os=home_os, cancel=token)
                except (RuntimeError, ValueError):  # e.g. an unclassifiable boot logo
                    return None  # the model still sees the new screen
                if profile and not new_connection:
                    found = replace(found, connected_at=profile.connected_at)
                profile = found
                status(f"Target: {profile.os} ({profile.input})")
                return profile

            def reconnect(reason: str) -> DeviceProfile | None:
                """Resume a run once the target is back (PLAN.md §5.6); re-detect a new connection."""
                nonlocal generation
                while True:
                    if not controller.await_target(RECONNECT_S):
                        raise Cancelled("target did not come back")
                    token.reset()
                    current = monitor.generation
                    if current == generation:
                        return None
                    try:
                        found = redetect(new_connection=True)
                    except Cancelled as exc:
                        if exc.reason not in RESUMABLE:
                            raise
                        continue  # it restarted again while coming back
                    generation = current
                    return found

            agent = Agent(executor, controller, Memory.load(cfg.session_path),
                          Telemetry(cfg.bench_csv, args.config_name or cfg.executor), args.max_steps, status,
                          bus, reconnect, redetect, deadline_s=DEADLINE_S if cfg.executor == "jev" else None)
            previous_goal = ""
            correction = False
            pointed: tuple[float, float] | None = None

            status(f"Executor: {cfg.executor}; model: {cfg.jev_model if cfg.executor == 'jev' else cfg.model_name}. Ctrl+C interrupts; idle Ctrl+C exits.")
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
                            monitor, capture, executor, args.os, home_os, token, profile, generation, status)
                        controller.execute(Action("click", x=command.x, y=command.y), profile)
                        controller.wait_settled()
                        if bus is not None:
                            bus.publish("tap", x=command.x, y=command.y)
                            if args.backend != "pi":
                                _show_screen(capture, bus)
                    except (Cancelled, KeyboardInterrupt, ActionError, HIDError) as exc:
                        controller.release_all()
                        status(f"Tap not sent ({type(exc).__name__}).")
                    except Exception as exc:  # noqa: BLE001 - keep serving; report it
                        _report_failure(status, bus, describe(exc, "tap"))
                    continue
                line = command.text.strip()
                if not line:
                    continue
                if pointed is not None:
                    # The note goes first: the user's own words stay the last line, which
                    # is what triage and its irreversible check read.
                    line = (f"The user tapped the screen to mean 'this': x={pointed[0]:.3f}, y={pointed[1]:.3f} "
                            f"(0-1 within the screen, origin top-left).\n{line}")
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
                        monitor, capture, executor, args.os, home_os, token, profile, generation, status)
                    # Goals from the page, the voice layer or a pipe need no window switch first.
                    if args.backend == "local" and bus is None and sys.stdin.isatty():
                        status("Focus the target window; starting in 2 seconds.")
                        token.sleep(2)
                    result = agent.run(goal, profile, args.task_id)
                    status(f"{result.status}: {result.message}")
                    previous_goal = goal
                    # Spoken or page requests after an interruption are new ones (the voice layer interrupts
                    # whenever the user talks); only an answer to the agent's question continues a goal.
                    correction = result.status == "ask_user" or (bus is None and result.status == "cancelled")
                except (Cancelled, KeyboardInterrupt):
                    token.cancel("barge-in")
                    controller.release_all()
                    previous_goal, correction = goal, bus is None
                    status("Interrupted. Enter a correction or /new GOAL.")
                except Exception as exc:  # noqa: BLE001 - one bad request must not stop the device
                    controller.release_all()
                    _report_failure(status, bus, describe(exc, "request"))
    except Exception as exc:
        status(f"Setup failed: {describe(exc, 'setup')}")
        return 1


def _report_failure(status: Callable[[str], None], bus: EventBus | None, message: str) -> None:
    """Tell the console and the display/voice layer (as a failed result) and keep running."""
    status(f"fail: {message}")
    if bus is not None:
        bus.publish("result", status="fail", message=message, steps=0, total_s=0)


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


def _connected_profile(monitor: ConnectionMonitor, capture: Capture, executor: Executor,
                       os_override: OSName | None, home_os: OSName | None, token: CancelToken,
                       profile: DeviceProfile | None,
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
            detected = detect_profile(capture, executor, os_override, home_os=home_os, cancel=token)
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
