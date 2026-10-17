"""End-to-end agent checks with synthetic video and recording HID."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from brain.agent.executor import ModelExecutor, Plan, StepContext, context_prompt, detect_os, parse_plan
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.agent.orchestrator import Guidance, Orchestrator
from brain.agent.quick import QuickStart
from brain.cancel import Cancelled, CancelToken
from brain.controller import Controller, SettleConfig
from brain.device import detect_profile
from brain.hid import HIDError
from brain.telemetry import Telemetry
from brain.types import Action, ActionError, DeviceProfile, Rect


class Capture:
    def __init__(self):
        self.frame = np.full((100, 160, 3), 80, dtype=np.uint8)

    def grab(self):
        return self.frame.copy()

    def signal_present(self):
        return True


class HID:
    def __init__(self, capture, change=True):
        self.commands = []
        self.capture, self.change = capture, change

    def send(self, command):
        self.commands.append(command)
        if self.change and command["op"] != "release_all":
            self.capture.frame[:] = 160 if self.capture.frame[0, 0, 0] == 80 else 80

    def is_mounted(self):
        return True


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = Capture()
        self.hid = HID(self.capture)
        self.cancel = CancelToken()
        self.controller = Controller(self.hid, self.capture, self.cancel,
                                     SettleConfig(min_s=0, stable_s=0, timeout_s=.1, poll_s=.001))
        self.memory = Memory(self.root / "nested/session.json")
        self.telemetry = Telemetry(self.root / "results.csv", "test")
        self.profile = DeviceProfile("macos", Rect(0, 0, 160, 100), "mouse", 1)
        self.fast = Mock()
        self.planner = Mock()
        self.planner.plan.return_value = Guidance(["Draft invite", "Verify draft"], "Use saved event facts")
        self.messages = []
        self.executor = Orchestrator(self.fast, self.planner, self.cancel, self.messages.append)

    def agent(self, executor=None):
        return Agent(executor or self.executor, self.controller, self.memory, self.telemetry, max_steps=8)

    def test_simple_goal_never_calls_planner(self):
        self.fast.next_actions.side_effect = [
            Plan("type", [Action("type_text", text="hello")], [], message="Typing hello"),
            Plan("verified", [Action("done", summary="Typed hello")], []),
        ]
        result = self.agent().run("Type hello", self.profile)
        self.assertEqual(result.status, "done")
        self.planner.plan.assert_not_called()
        self.assertEqual(self.hid.commands[0], {"op": "type", "text": "hello"})
        self.assertEqual(self.memory.task["status"], "done")
        with (self.root / "results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]["success"], "1")
        self.assertEqual(len(rows), 3)

    def test_complex_goal_acknowledges_before_planning_and_refreshes_screen(self):
        self.fast.next_actions.side_effect = [
            Plan("complex", [], [], needs_planner=True, message="I'll draft the invite"),
            Plan("done", [Action("done", summary="Draft is ready")], []),
        ]
        def plan(image, ctx):
            self.assertEqual(self.messages[0], "I'll draft the invite")
            self.assertEqual(self.hid.commands, [])
            self.capture.frame[:] = 180
            return Guidance(["Read event", "Draft invite"], "Draft only; do not send")
        self.planner.plan.side_effect = plan
        result = self.agent().run("Read event and draft an invite", self.profile)
        self.assertEqual(result.status, "done")
        calls = self.fast.next_actions.call_args_list
        self.assertNotEqual(calls[0].args[0], calls[1].args[0])
        self.assertIn("Draft only", calls[1].args[1].guidance)
        self.assertEqual(self.memory.task["subtasks"], ["Read event", "Draft invite"])

    def test_cancelled_planner_result_is_discarded(self):
        self.fast.next_actions.return_value = Plan("complex", [], [], needs_planner=True)
        def plan(image, ctx):
            self.cancel.cancel("target disconnected")
            return Guidance(["stale"], "stale")
        self.planner.plan.side_effect = plan
        result = self.agent().run("Complex task", self.profile)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.memory.task["subtasks"], [])
        self.assertEqual(self.hid.commands, [{"op": "release_all"}])

    def test_completion_requires_fresh_frame_after_input(self):
        self.fast.next_actions.side_effect = [
            Plan("type", [Action("type_text", text="hello"), Action("done")], []),
            Plan("verify", [Action("ask_user", question="Is that the right field?")], []),
        ]
        result = self.agent().run("Type", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertEqual(result.steps, 2)
        self.assertIn("fresh screen", self.fast.next_actions.call_args.args[1].note)

    def test_stalled_actions_replan_then_cancel(self):
        self.hid.change = False
        self.fast.next_actions.return_value = Plan("try", [Action("hotkey", keys=("ENTER",))], [])
        result = self.agent().run("Open thing", self.profile)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.message, "3 failed verifications")
        self.planner.plan.assert_called_once()
        self.assertEqual(sum(c["op"] == "key" for c in self.hid.commands), 3)

    def test_rejected_skill_drops_dependent_actions(self):
        self.fast.next_actions.side_effect = [
            Plan("skill", [Action("skill", name="missing"), Action("type_text", text="unsafe tail")], []),
            Plan("ask", [Action("ask_user", question="Which app?")], []),
        ]
        result = self.agent().run("Open app", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertEqual(self.hid.commands, [{"op": "release_all"}])
        self.assertIn("Rejected action", self.fast.next_actions.call_args.args[1].note)

    def test_repeated_delegation_is_bounded(self):
        self.fast.next_actions.return_value = Plan("delegate", [], [], needs_planner=True)
        result = self.agent().run("Complex task", self.profile)
        self.assertEqual(result.status, "fail")
        self.assertEqual(self.planner.plan.call_count, 2)
        self.assertEqual(self.hid.commands, [{"op": "release_all"}])

    def test_new_task_clears_guidance(self):
        self.memory.set_task("old", "running")
        first = StepContext("old", self.profile, self.memory, [], 1, 5, run_id="a")
        self.fast.next_actions.return_value = Plan("delegate", [], [], needs_planner=True)
        self.executor.next_actions(b"image", first)
        self.fast.next_actions.return_value = Plan("done", [Action("done")], [])
        self.executor.next_actions(b"image", StepContext("new", self.profile, self.memory, [], 1, 5, run_id="b"))
        self.assertEqual(self.fast.next_actions.call_args.args[1].guidance, "")

    def test_quick_command_acts_before_any_model_call(self):
        self.fast.next_actions.side_effect = lambda image, ctx: (
            self.assertEqual(self.hid.commands[:2], [{"op": "key", "keys": ["CTRL", "A"]},
                                                     {"op": "type", "text": "hi"}])
            or Plan("verified", [Action("done", summary="ok")], []))
        result = self.agent(QuickStart(self.fast)).run('press ctrl a, then type "hi"', self.profile)
        self.assertEqual(result.status, "done")
        self.assertEqual(self.fast.next_actions.call_args.args[1].step, 2)

    def test_os_detection_and_prompt_hint(self):
        model = Mock()
        for answer in ("bios", "ios", "chromeos"):
            model.generate.return_value = {"os": answer}
            self.assertEqual(detect_os(model, b"jpeg"), answer)
        model.generate.return_value = {"os": "Windows 11"}
        with self.assertRaises(ValueError):
            detect_os(model, b"jpeg")
        prompt = context_prompt(StepContext("copy", self.profile, self.memory, [], 1, 5))
        self.assertIn("Device: macos (mouse). GUI (Command) shortcuts", prompt)

    def test_device_detection_uses_override_without_model(self):
        profile = detect_profile(self.capture, self.fast, "android", cancel=self.cancel)
        self.assertEqual(profile.input, "touch")
        self.fast.classify_os.assert_not_called()

    def test_invalid_batch_is_dropped(self):
        for value in (float("nan"), float("inf"), True, -1, 1001, "5"):
            with self.subTest(value=value):
                result = parse_plan({"actions": [{"type": "click", "x": value, "y": 1},
                                                  {"type": "type_text", "text": "tail"}]})
                self.assertTrue(result.rejected)
                self.assertEqual(result.actions, [])

    def test_oversized_batch_keeps_first_five(self):
        result = parse_plan({"actions": [{"type": "type_text", "text": str(n)} for n in range(7)]})
        self.assertEqual([a.text for a in result.actions], ["0", "1", "2", "3", "4"])

    def test_cancel_does_not_swallow_worker_timeout(self):
        def timeout():
            raise TimeoutError("provider timed out")
        with self.assertRaisesRegex(TimeoutError, "provider timed out"):
            self.cancel.run(timeout)

    def test_cancellation_prevents_model_result_execution(self):
        def next_actions(image, ctx):
            self.cancel.cancel("barge-in")
            return Plan("late", [Action("type_text", text="stale")], [])
        self.fast.next_actions.side_effect = next_actions
        result = self.agent().run("Type something", self.profile)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.hid.commands, [{"op": "release_all"}])

    def test_screen_change_during_inference_discards_action(self):
        def next_actions(image, ctx):
            if ctx.step == 1:
                self.capture.frame[:] = 210
                return Plan("old target", [Action("click", x=.5, y=.5)], [])
            self.assertIn("discarded", ctx.note)
            return Plan("clarify", [Action("ask_user", question="Which window?")], [])
        self.fast.next_actions.side_effect = next_actions
        result = self.agent().run("Click the field", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertEqual(self.hid.commands, [{"op": "release_all"}])

    def test_restart_resumes_on_the_new_screen(self):
        def send(command):
            HID.send(self.hid, command)
            if command == {"op": "key", "keys": ["ENTER"]}:
                self.cancel.cancel("target disconnected")
        self.hid.send = send
        bios = DeviceProfile("bios", Rect(0, 0, 160, 100), "mouse", 2)
        reconnect = Mock(side_effect=lambda reason: self.cancel.reset() or bios)
        self.fast.next_actions.side_effect = [
            Plan("restart", [Action("skill", name="boot_keys", args={"keys": ["F2"]}),
                             Action("hotkey", keys=("ENTER",))], []),
            Plan("in setup", [Action("done", summary="In setup")], []),
        ]
        agent = Agent(self.executor, self.controller, self.memory, self.telemetry, reconnect=reconnect)
        result = agent.run("Open firmware setup", self.profile)
        self.assertEqual(result.status, "done")
        reconnect.assert_called_once_with("target disconnected")
        ctx = self.fast.next_actions.call_args.args[1]
        self.assertEqual(ctx.profile.os, "bios")
        self.assertIn("restarted", ctx.note)
        self.assertEqual(ctx.history[-1], "hotkey(ENTER)")

    def test_waiting_on_a_still_screen_is_not_a_failure(self):
        self.hid.change = False
        self.fast.next_actions.side_effect = [Plan("booting", [Action("skill", name="wait")], [])] * 3 + [
            Plan("booted", [Action("done", summary="Booted")], [])]
        with patch("brain.controller.WAIT_S", 0.01):
            result = self.agent().run("Wait for the boot", self.profile)
        self.assertEqual(result.status, "done")

    def test_firmware_rejects_pointer_input(self):
        bios = DeviceProfile("bios", Rect(0, 0, 160, 100), "mouse", 1)
        for action in (Action("right_click", x=.5, y=.5), Action("scroll", dy=3)):
            with self.subTest(action=action.type), self.assertRaises(ActionError):
                self.controller.execute(action, bios)
        self.assertEqual(self.hid.commands, [])
        with self.assertRaises(ActionError):  # a screen without a menu highlight
            self.controller.execute(Action("click", x=.5, y=.5), bios)
        self.assertEqual(self.hid.commands, [{"op": "key", "keys": ["DOWN"]}])

    def test_boot_keys_reach_skill_args(self):
        for raw in ({"type": "skill", "name": "boot_keys", "keys": ["F2"]}, {"type": "boot_keys", "keys": ["F2"]}):
            result = parse_plan({"actions": [raw]})
            self.assertEqual((result.actions[0].name, result.actions[0].args), ("boot_keys", {"name": "", "keys": ["F2"]}))

    def test_key_press_named_key_becomes_hotkey(self):
        for raw in ({"type": "key", "keys": ["DOWN"]}, {"type": "key", "key": "DOWN"}):
            self.assertEqual(parse_plan({"actions": [raw]}).actions[0].keys, ("DOWN",))

    def test_done_after_input_must_show_on_screen(self):
        model = Mock()
        done = {"reasoning": "", "actions": [{"type": "remember", "key": "k", "text": "v"},
                                             {"type": "done", "message": "Set to 7"}]}
        model.generate.side_effect = [done, {"complete": False, "missing": "time-out shows 0"},
                                      done, {"complete": True, "missing": ""}, done]
        with patch("brain.agent.executor.language_model", return_value=model):
            executor = ModelExecutor(Mock(), self.cancel, routing=False)
        acted = StepContext("Set the time-out to 7", self.profile, self.memory, ["hotkey(F10)"], 3, 8)
        refused = executor.next_actions(b"jpeg", acted)
        self.assertEqual([a.type for a in refused.actions], ["remember"])
        self.assertIn("time-out shows 0", refused.rejected[0])
        self.assertEqual(executor.next_actions(b"jpeg", acted).actions[1].type, "done")
        # A question answered before any input skips the check.
        idle = StepContext("What is the time-out?", self.profile, self.memory, [], 1, 8)
        self.assertEqual(executor.next_actions(b"jpeg", idle).actions[1].type, "done")
        self.assertEqual(model.generate.call_count, 5)

    def test_semantic_executor_checks_covered_result_without_double_counting(self):
        self.hid.change=False
        executor=Mock()
        executor.verifies_stages=True
        executor.next_actions.side_effect=[
            Plan('input',[Action('hotkey',keys=('ENTER',))],[]),
            Plan('reveal',[Action('hotkey',keys=('ESC',))],[]),
            Plan('verify',[Action('hotkey',keys=('TAB',))],[]),
            Plan('observed',[Action('done',summary='Verified')],[])]
        result=self.agent(executor).run('Inspect result',self.profile)
        self.assertEqual(result.status,'done')
        self.assertEqual(executor.acknowledge.call_count,3)


class Menu:
    """Firmware menu video and keyboard: five rows of dense text, the selected row drawn
    inverse on a solid bar, rows 24 px apart from y=40."""

    def __init__(self, selected, wrap):
        self.frame = np.full((240, 320, 3), 152, dtype=np.uint8)
        self.selected, self.wrap, self.commands = selected, wrap, []
        self.draw()

    def draw(self):
        self.frame[:] = 152
        letters = [x for x in range(24, 196) if x % 4 < 2]
        for row in range(5):
            top, chosen = 40 + 24 * row, row == self.selected
            if chosen:
                self.frame[top:top + 20, 20:200] = 0
            self.frame[top + 4:top + 16, letters] = 152 if chosen else 0

    def grab(self):
        return self.frame.copy()

    def signal_present(self):
        return True

    def is_mounted(self):
        return True

    def send(self, command):
        self.commands.append(command)
        if command["op"] == "key" and command["keys"] in (["DOWN"], ["UP"]):
            moved = self.selected + (1 if command["keys"] == ["DOWN"] else -1)
            self.selected = moved % 5 if self.wrap else min(max(moved, 0), 4)
            self.draw()


class FirmwareClickTests(unittest.TestCase):
    def test_click_moves_the_highlight_to_the_nearest_row(self):
        bios = DeviceProfile("bios", Rect(0, 0, 320, 240), "mouse", 1)
        for start, wrap, y, row, keys in (
            (0, True, 122, 3, "DOWN DOWN DOWN"),
            (3, True, 74, 1, "DOWN UP UP UP"),
            (0, True, 86, 2, "DOWN DOWN"),  # between rows 1 and 2, nearer 2
            (1, False, 228, 4, "DOWN DOWN DOWN DOWN"),  # below the menu: stops at its end
            (1, True, 228, 4, "DOWN DOWN DOWN DOWN UP"),  # or wraps round and steps back
        ):
            with self.subTest(start=start, wrap=wrap, y=y):
                menu = Menu(start, wrap)
                controller = Controller(menu, menu, CancelToken(),
                                        SettleConfig(min_s=0, stable_s=0, timeout_s=.1, poll_s=.001))
                controller.execute(Action("click", x=.5, y=y / 240), bios)
                self.assertEqual(menu.selected, row)
                self.assertEqual(" ".join(c["keys"][0] for c in menu.commands), keys)
        controller.execute(Action("double_click", x=.5, y=50 / 240), bios)
        self.assertEqual((menu.selected, menu.commands[-1]), (0, {"op": "key", "keys": ["ENTER"]}))


class AwaitTargetTests(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        self.hid = HID(self.capture, change=False)
        self.controller = Controller(self.hid, self.capture, CancelToken())
        patcher = patch.multiple("brain.controller", BOOT_TAP_S=0.001, BOOT_TAP_AFTER_S=0.02)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_armed_keys_tap_through_one_restart(self):
        def send(command):
            HID.send(self.hid, command)
            if len(self.hid.commands) == 1:
                raise HIDError("usb not up yet")
        self.hid.send = send
        self.controller.arm_boot_keys(["F2"])
        self.assertTrue(self.controller.await_target(1))
        self.assertGreater(len(self.hid.commands), 2)
        self.assertTrue(all(c == {"op": "key", "keys": ["F2"]} for c in self.hid.commands))
        self.hid.commands.clear()
        self.assertTrue(self.controller.await_target(1))
        self.assertEqual(self.hid.commands, [])

    def test_no_video_times_out(self):
        self.capture.frame[:] = 0
        self.assertFalse(self.controller.await_target(0.05))


if __name__ == "__main__":
    unittest.main()
