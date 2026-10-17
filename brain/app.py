"""Typed-goal development interface for the screenshot-driven agent."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from typing import Callable

from brain.agent.executor import Executor
from brain.agent.factory import build_executor
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.cancel import Cancelled, CancelToken
from brain.capture import Capture, ScreenCapture, V4L2Capture
from brain.config import Config
from brain.controller import Controller
from brain.device import detect_profile
from brain.errors import describe
from brain.hid import LocalHID, PicoHID
from brain.monitor import ConnectionMonitor
from brain.telemetry import Telemetry
from brain.types import DeviceProfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "pi", "browser"], default="pi")
    parser.add_argument("--url", help="Browser target URL (isolated profile)")
    parser.add_argument("--headed", action="store_true", help="Show browser target")
    parser.add_argument("--os", choices=["macos", "windows", "linux", "android"])
    parser.add_argument("--task-id", default="interactive")
    parser.add_argument("--config-name", default="jev-flash")
    parser.add_argument("--max-steps", type=int, default=30)
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
            agent = Agent(executor, controller, Memory.load(cfg.session_path),
                          Telemetry(cfg.bench_csv, args.config_name), args.max_steps, status)
            profile = None
            generation = -1
            previous_goal = ""
            correction = False
            status(f"Executor: {cfg.executor}; Gemini: {cfg.gemini_model}. Ctrl+C interrupts; idle Ctrl+C exits.")
            while True:
                try:
                    line = input("Correction (or /new GOAL)> " if correction else "Goal> ").strip()
                except (EOFError, KeyboardInterrupt):
                    status("Goodbye.")
                    return 0
                if not line:
                    continue
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
