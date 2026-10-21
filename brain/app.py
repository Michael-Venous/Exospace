"""Typed-goal development interface for the screenshot-driven agent."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from contextlib import ExitStack
from typing import Callable

from brain.agent.executor import Executor
from brain.agent.factory import build_executor
from brain.agent.loop import RESUMABLE, Agent
from brain.agent.memory import Memory
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, CommandCapture, ScreenCapture, V4L2Capture, active_rect
from brain.config import Config
from brain.controller import Controller
from brain.device import detect_profile, shows_os
from brain.errors import describe
from brain.display.recordings import RECORDINGS, Recorder, RecordingsServer
from brain.display.server import Command, DisplayServer
from brain.events import EventBus
from brain.hid import HID, HIDError, LocalHID, PicoHID
from brain.imaging import prepare
from brain.monitor import ConnectionMonitor
from brain.net import ipv4_only
from brain.telemetry import Telemetry
from brain.types import OS_NAMES, Action, ActionError, DeviceProfile, OSName

LIVE_FPS = 4  # capture-card frames pushed to the display between steps
RECONNECT_S = 180.0  # a restarting computer can take minutes to show video again
FIRST_FRAME_S = 5.0  # for the capture card's first frames after start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "pi", "browser"], default="pi")
    parser.add_argument("--url", help="Browser target URL (isolated profile)")
    parser.add_argument("--headed", action="store_true", help="Show browser target")
    parser.add_argument("--os", choices=[os for os in OS_NAMES if os != "unknown"])
    parser.add_argument("--task-id", default="interactive")
    parser.add_argument("--config-name", help="Benchmark label (default: the executor)")
    parser.add_argument("--max-steps", type=int, default=60)  # deep mode needs room; most requests take < 10
    parser.add_argument("--display", type=int, metavar="PORT",
                        help="Serve the touch display page on this port (e.g. 8770)")
    parser.add_argument("--display-host", default="127.0.0.1")
    parser.add_argument("--recordings", type=int, metavar="PORT",
                        help="Record each request and serve the phone player on this port (e.g. 8790); needs --display")
    parser.add_argument("--recordings-host", default="127.0.0.1", help="e.g. the Pi's Tailscale address")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.recordings and not args.display:
        parser.error("--recordings needs --display (it records the display's events)")
    ipv4_only()
    cfg = Config.load()
    token = CancelToken()
    status = lambda message: print(message, flush=True)
    if cfg.executor != "jev-only" and not (cfg.gemini_api_key or cfg.uses_zai):
        status("Set ZAI_API_KEY or GEMINI_API_KEY in the project .env (see .env.example); either one works.")
        return 1
    try:
        with ExitStack() as cleanup:
            bus = EventBus() if args.display else None

            def notify(message: str) -> None:  # executor news ("switching to deep mode") is spoken too
                status(message)
                if bus is not None:
                    bus.publish("status", message=message)

            executor = build_executor(cfg, token, cleanup, notify)
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
                capture = CommandCapture(cfg.local_capture) if cfg.local_capture else ScreenCapture()
                cleanup.callback(capture.close)
                if cfg.local_pico:  # the Pico plugged into this machine is all of its input
                    hid = PicoHID(cfg.local_pico, cfg.pico_baud)
                else:
                    hid = LocalHID(PicoHID(cfg.local_keyboard, cfg.pico_baud) if cfg.local_keyboard else None)
            cleanup.callback(hid.close)
            controller = Controller(hid, capture, token)
            cleanup.callback(controller.release_all)
            monitor = ConnectionMonitor(capture, hid, token.cancel)
            cleanup.callback(monitor.close)
            commands: queue.Queue[Command] = queue.Queue()
            if bus is not None:
                server = DisplayServer(bus, commands, token.cancel, args.display_host, args.display)
                server.start()
                cleanup.callback(server.close)
                if args.backend == "pi":
                    _pump_live_frames(capture, bus)
                else:
                    _show_screen(capture, bus)
                threading.Thread(target=_read_stdin, args=(commands,), name="stdin", daemon=True).start()
                status(f"Display: {server.url}  (goals from the page or typed here)")
                if args.recordings:
                    Recorder(bus, RECORDINGS).start()
                    player = RecordingsServer(RECORDINGS, args.recordings_host, args.recordings)
                    player.start()
                    cleanup.callback(player.close)
                    status(f"Recordings: {player.url}")
            profile: DeviceProfile | None = None
            generation = -1

            def reconnect(reason: str) -> DeviceProfile | None:
                """Resume a run once the target is back (PLAN.md §5.6); re-detect a new connection."""
                nonlocal profile, generation
                while True:
                    if not controller.await_target(RECONNECT_S):
                        raise Cancelled("target did not come back")
                    token.reset()
                    current = monitor.generation
                    if current == generation:
                        return None
                    try:
                        found = detect_profile(capture, executor, args.os, cancel=token, pointer=controller.hover,
                                               press=controller.hotkey)
                    except Cancelled as exc:
                        if exc.reason not in RESUMABLE:
                            raise
                        continue  # it restarted again while coming back
                    except (RuntimeError, ValueError):  # no usable frame yet: keep the profile
                        found = None
                    generation = current
                    if found is not None:
                        profile = found
                        status(f"Target: {profile.os} ({profile.input}), screen {profile.rect.as_list()} of the frame")
                    return found

            agent = Agent(executor, controller, Memory.load(cfg.session_path),
                          Telemetry(cfg.bench_csv, args.config_name or cfg.executor), args.max_steps, status,
                          bus, reconnect, more_words=lambda: _more_words(commands))
            previous_goal = ""
            correction = False
            pointed: tuple[float, float] | None = None

            status(f"Executor: {cfg.executor}; model: {cfg.model_name}. Ctrl+C interrupts; idle Ctrl+C exits.")
            # Detect now, not on the first request: it took ~8 s on the Pi (OS check, calibration). The capture card's
            # first frames come a moment after start: checked at once, the target was never detected here (2026-10-19).
            if controller.await_target(FIRST_FRAME_S) and monitor.connected:
                try:
                    profile, generation = _connected_profile(
                        monitor, capture, executor, args.os, token, profile, generation, status, controller.hover, controller.hotkey)
                except (Cancelled, RuntimeError, HIDError):
                    pass  # detected again on the first request
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
                            monitor, capture, executor, args.os, token, profile, generation, status, controller.hover, controller.hotkey)
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
                elif correction or (command.continues and previous_goal):
                    goal = previous_goal + "\nUser correction: " + line
                else:
                    goal = line
                if not goal:
                    continue
                try:
                    profile, generation = _connected_profile(
                        monitor, capture, executor, args.os, token, profile, generation, status, controller.hover, controller.hotkey)
                    # Goals from the page, the voice layer or a pipe need no window switch first.
                    if args.backend == "local" and bus is None and sys.stdin.isatty():
                        status("Focus the target window; starting in 2 seconds.")
                        token.sleep(2)
                    result = agent.run(goal, profile, args.task_id)
                    status(f"{result.status}: {result.message}")
                    previous_goal = agent.goal or goal
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


def _more_words(commands: queue.Queue[Command]) -> str | None:
    """The user's next words if they continue the running request (same push-to-talk hold), else None."""
    with commands.mutex:
        if commands.queue and commands.queue[0].kind == "goal" and commands.queue[0].continues:
            return commands.queue.popleft().text
    return None


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
                       os_override: OSName | None, token: CancelToken,
                       profile: DeviceProfile | None,
                       generation: int, status: Callable[[str], None],
                       pointer: Callable[[float, float], None] | None = None,
                       press: Callable[[list[str]], None] | None = None) -> tuple[DeviceProfile, int]:
    """Wait for a target and re-detect it if it changed. A disconnect during
    detection retries instead of dropping the user's pending goal."""
    while True:
        if not monitor.connected:
            status("Waiting for target…")
        current = monitor.wait_connected()
        token.reset()
        # An "unknown" OS is asked again each request: Mint's desktop once read unknown at start, which turned app
        # launches off ("open a terminal" failed) until the agent restarted (2026-10-19).
        if profile is not None and current == generation and (
                profile.os != "unknown" or os_override is not None or not shows_os(capture, executor, profile)):
            return profile, generation
        try:
            detected = detect_profile(capture, executor, os_override, cancel=token, pointer=pointer,
                                      press=None if pointer is None else press)
        except Cancelled:
            if monitor.connected:
                raise  # a real interruption, not an unplug
            continue
        if monitor.connected and monitor.generation == current:
            status(f"Target: {detected.os} ({detected.input}), screen {detected.rect.as_list()} of the frame")
            return detected, current
        status("Target changed during detection; retrying.")


if __name__ == "__main__":
    raise SystemExit(main())
