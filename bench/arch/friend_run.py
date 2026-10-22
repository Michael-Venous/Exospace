"""Another branch's runtime agent on the test laptop, driven by `laptop_tests.py --agent friend` (bench only).

Run with PYTHONPATH set to that branch's checkout, so `brain` is its runtime (its Agent, executor and controller);
the laptop's capture and HID come from this branch's `laptop.py`, loaded by path. One JSON request per stdin line,
{"goal": "..."}; each reply line is {"status", "message", "steps", "said", "s", "inputs"}: what the agent said, how
long the run took and when each input went out (seconds from the request). EXECUTOR picks its executor
(`jev-only`: Jev's own steps, no language model).
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

from brain.agent.factory import build_executor
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.cancel import CancelToken
from brain.config import Config
from brain.controller import Controller
from brain.telemetry import Telemetry
from brain.types import DeviceProfile, Rect

spec = importlib.util.spec_from_file_location("laptop", Path(__file__).with_name("laptop.py"))
laptop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(laptop)


class Timed:
    """The target, noting when each input is sent (for first-input latency)."""

    def __init__(self, target) -> None:
        self.target, self.inputs = target, []

    def send(self, cmd: dict) -> None:
        if cmd.get("op") not in ("ping", "release_all", "move"):
            self.inputs.append(time.monotonic())
        self.target.send(cmd)

    def __getattr__(self, name):
        return getattr(self.target, name)


def main() -> None:
    from contextlib import ExitStack

    cfg = Config.load()
    target = Timed(laptop.LaptopTarget())
    h, w = target.grab().shape[:2]
    profile = DeviceProfile("linux", Rect(0, 0, w, h), "mouse", time.time())
    said: list[str] = []
    with ExitStack() as cleanup:
        token = CancelToken()
        executor = build_executor(cfg, token, cleanup, said.append)
        controller = Controller(target, target, token)
        agent = Agent(executor, controller, Memory(path=Path("/tmp/exo_friend_memory.json")),
                      Telemetry(Path("/tmp/exo_friend_bench.csv"), cfg.executor), 30, said.append)
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            goal = json.loads(line)["goal"]
            said.clear()
            target.inputs.clear()
            started = time.monotonic()
            result = agent.run(goal, profile)
            print(json.dumps({"status": result.status, "message": result.message, "steps": result.steps,
                              "said": list(said), "s": round(time.monotonic() - started, 2),
                              "inputs": [round(t - started, 2) for t in target.inputs]}), flush=True)
    target.close()


if __name__ == "__main__":
    main()
