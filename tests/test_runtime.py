"""End-to-end agent checks with synthetic video and recording HID."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from brain.agent.executor import Plan, StepContext, parse_plan
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.agent.orchestrator import Guidance, Orchestrator
from brain.app import _connected_profile
from brain.cancel import Cancelled, CancelToken
from brain.controller import Controller, SettleConfig
from brain.device import detect_profile
from brain.errors import describe
from brain.hid import HIDError
from brain.telemetry import Telemetry
from brain.types import Action, DeviceProfile, Rect


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

    def test_moving_screen_acts_after_one_discard(self):
        def next_actions(image, ctx):
            if ctx.step <= 2:
                self.capture.frame[:] = 210 if ctx.step == 1 else 40  # never stops moving
                return Plan("target", [Action("click", x=.5, y=.5)], [])
            return Plan("clarify", [Action("ask_user", question="Done?")], [])
        self.fast.next_actions.side_effect = next_actions
        result = self.agent().run("Click the field", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertEqual(sum(c["op"] == "click" for c in self.hid.commands), 1)

    def test_disconnect_during_detection_keeps_pending_goal(self):
        monitor = SimpleNamespace(connected=True, generation=1)
        monitor.wait_connected = lambda: setattr(monitor, "connected", True) or monitor.generation
        def detect(*args, **kwargs):
            if monitor.generation == 1:
                monitor.connected, monitor.generation = False, 2  # unplugged mid-detection
                raise Cancelled("target disconnected")
            return self.profile
        with patch("brain.app.detect_profile", side_effect=detect):
            profile, generation = _connected_profile(
                monitor, self.capture, self.fast, None, self.cancel, None, -1, self.messages.append)
        self.assertEqual((profile, generation), (self.profile, 2))

    def test_errors_show_hardware_detail_but_hide_provider_detail(self):
        with patch("brain.errors.ERROR_LOG", self.root / "errors.log"):
            self.assertEqual(describe(HIDError("Pico did not acknowledge key"), "task"),
                             "HIDError: Pico did not acknowledge key")
            hidden = describe(ValueError("payload with request data"), "task")
            self.assertNotIn("payload", hidden)
            self.assertIn("payload with request data", (self.root / "errors.log").read_text())

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



if __name__ == "__main__":
    unittest.main()
