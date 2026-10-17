"""Pure-Jev stand-in for race's planner (EXECUTOR=jev): no language model is called.

Race keeps everything Jev does alone: confident clicks, remembered remarks and the
irreversible-action gate. A request that needs a plan ends here: done if Jev sees the
goal met from the executed steps and the screen, else a fail naming the gap.
"""

from __future__ import annotations

from typing import cast

from brain.agent.executor import Plan, StepContext
from brain.agent.jev import JevVision
from brain.agent.jev_flash import COMPLETION_GATE
from brain.agent.prompts import OS_PROMPT
from brain.cancel import CancelToken
from brain.config import Config
from brain.types import Action, OSName

OS_CRITERIA = {
    "windows": "Microsoft Windows desktop or app.",
    "macos": "Apple macOS desktop or app.",
    "linux": "A Linux desktop (GNOME, KDE, Xfce and the like) or app.",
    "chromeos": "A Chromebook (ChromeOS shelf and launcher).",
    "android": "An Android phone or tablet.",
    "ios": "An iPhone or iPad.",
    "bios": "Firmware setup, a boot menu or another pre-OS screen.",
    "unknown": "None of the above can be told from this screen.",
}


class JevOnlyPlanner:
    def __init__(self, cfg: Config, cancel: CancelToken) -> None:
        self.jev = JevVision(cfg.typesafe_api_key, cancel, cfg.model_log, cfg.jev_model,
                             cfg.jev_timeout_s, cfg.jev_min_presence, cfg.jev_min_confidence)

    def acknowledge(self, executed: bool) -> None:
        pass  # no plan cursor to advance

    def next_actions(self, image: bytes, ctx: StepContext) -> Plan:
        score = self.jev.completion(image, ctx.goal, ctx.history, run_id=ctx.run_id, step=ctx.step)
        if score >= COMPLETION_GATE:
            return Plan("Jev verified the completed goal", [Action("done", summary="Done.")], [])
        return Plan("Needs a language model", [Action(
            "fail", reason="Jev alone can only click one visible control, run a quick command or take a note; "
                           "this request needs the language-model planner (EXECUTOR=race).")], [])

    def classify_os(self, image: bytes) -> OSName:
        answers, _, _ = self.jev.evaluate(image, {"os": {"type": "choice", "instructions": OS_PROMPT,
                                                         "criteria": OS_CRITERIA}})
        os = answers["os"].get("choice")
        if os not in OS_CRITERIA:
            raise ValueError("invalid operating system classification")
        return cast(OSName, os)

    def close(self) -> None:
        self.jev.close()
