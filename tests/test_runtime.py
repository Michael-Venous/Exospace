"""End-to-end agent checks with synthetic video and recording HID."""

import csv
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from brain.agent.executor import ModelExecutor, Plan, StepContext, context_prompt, detect_os, parse_plan
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.agent.orchestrator import Guidance, Orchestrator
from brain.agent.prompts import DONE_PROMPT, SYSTEM_PROMPT
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
        self.commands, self.callbacks = [], []
        self.capture, self.change = capture, change

    def send(self, command):
        self.commands.append(command)
        if self.change and command["op"] != "release_all":
            self.capture.frame[:] = 160 if self.capture.frame[0, 0, 0] == 80 else 80

    def on_event(self, callback):
        self.callbacks.append(callback)

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

    def test_done_after_a_wait_must_show_on_screen(self):
        self.fast.next_actions.side_effect = [
            Plan("installing", [Action("skill", name="wait"), Action("done", summary="Installed")], []),
            Plan("error dialog", [Action("ask_user", question="The install failed. Retry?")], []),
        ]
        with patch("brain.controller.WAIT_S", 0.01):
            result = self.agent().run("Install the update", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertIn("fresh screen", self.fast.next_actions.call_args.args[1].note)

    def test_done_after_an_action_is_checked_on_the_new_screen(self):
        shutter = Plan("shutter", [Action("click", x=0.5, y=0.9), Action("done", summary="Took a photo")], [])
        self.fast.next_actions.side_effect = [shutter, Plan("ask", [Action("ask_user", question="Again?")], [])]
        self.fast.missing.return_value = ""
        self.assertEqual(self.agent(self.fast).run("Take a picture", self.profile).status, "done")
        self.assertEqual(self.fast.next_actions.call_count, 1)
        self.assertEqual(self.fast.missing.call_args.args[1].history, ["click(500,900)"])

        self.fast.next_actions.side_effect = [shutter, Plan("ask", [Action("ask_user", question="Again?")], [])]
        self.fast.missing.return_value = "no photo"
        self.assertEqual(self.agent(self.fast).run("Take a picture", self.profile).status, "ask_user")
        self.assertIn("no photo", self.fast.next_actions.call_args.args[1].note)

    def test_completion_requires_fresh_frame_after_input(self):
        self.fast.next_actions.side_effect = [
            Plan("type", [Action("type_text", text="hello"), Action("done")], []),
            Plan("verify", [Action("ask_user", question="Is that the right field?")], []),
        ]
        result = self.agent().run("Type", self.profile)
        self.assertEqual(result.status, "ask_user")
        self.assertEqual(result.steps, 2)
        self.assertIn("fresh screen", self.fast.next_actions.call_args.args[1].note)

    def test_batch_stops_before_a_click_aimed_at_a_changed_screen(self):
        clicks = [Action("click", x=.5, y=.5), Action("click", x=.5, y=.9)]
        self.fast.next_actions.side_effect = [
            Plan("form", clicks, []),  # the fake screen changes on the first click
            Plan("keyboard up", [Action("done", summary="Tapped")], []),
            Plan("form", clicks, []),
            Plan("form", [Action("done", summary="Tapped")], []),
        ]
        self.assertEqual(self.agent().run("Fill the form", self.profile).status, "done")
        self.assertIn("changed mid-batch", self.fast.next_actions.call_args.args[1].note)
        self.hid.change = False  # a click that leaves the screen alone keeps the batch going
        self.agent().run("Fill the form", self.profile)
        self.assertEqual([c["y"] for c in self.hid.commands if c["op"] == "click"], [16384, 16384, 29490])

    def test_stalled_actions_replan_then_cancel(self):
        self.hid.change = False
        self.fast.next_actions.return_value = Plan("try", [Action("hotkey", keys=("ENTER",))], [])
        result = self.agent().run("Open thing", self.profile)
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.message, "3 failed verifications")
        self.planner.plan.assert_called_once()
        self.assertEqual(sum(c["op"] == "key" for c in self.hid.commands), 3)

    def test_same_batch_from_same_screen_stops_at_third_try(self):
        self.hid.change = False
        self.fast.next_actions.return_value = Plan("launch", [Action("skill", name="open_app",
                                                                     args={"name": "video player"})], [])
        with patch("brain.agent.loop.visibly_changed", return_value=True), \
                patch("brain.skills.APP_START_S", 0.01):  # each try flashed the launcher; no window came
            result = self.agent().run("See the video", self.profile)
        self.assertEqual((result.status, result.message, result.steps), ("fail", "Stuck repeating the same action", 3))

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

    def test_os_detection_and_prompt(self):
        model = Mock()
        for answer in ("unknown", "ios", "chromeos"):
            model.generate.return_value = {"os": answer}
            self.assertEqual(detect_os(model, b"jpeg"), answer)
        model.generate.return_value = {"os": "Windows 11"}
        with self.assertRaises(ValueError):
            detect_os(model, b"jpeg")
        prompt = context_prompt(StepContext("copy", self.profile, self.memory, [], 1, 5))
        self.assertIn("Device: macos (mouse)\n", prompt)

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

    def test_a_screen_that_never_settles_still_runs_the_plan(self):
        frames = iter(range(10**6))  # a live camera preview: every grab differs
        self.capture.grab = lambda: np.full((100, 160, 3), 60 + next(frames) % 2 * 40, dtype=np.uint8)
        self.fast.next_actions.side_effect = [Plan("shutter", [Action("click", x=.5, y=.9)], []),
                                              Plan("ask", [Action("ask_user", question="Again?")], [])]
        self.assertEqual(self.agent(self.fast).run("Take a picture", self.profile).status, "ask_user")
        self.assertIn("click", [command["op"] for command in self.hid.commands])

    def test_restart_resumes_on_the_new_screen(self):
        def send(command):
            HID.send(self.hid, command)
            if command == {"op": "key", "keys": ["ENTER"]}:
                self.cancel.cancel("target disconnected")
        self.hid.send = send
        back = DeviceProfile("windows", Rect(0, 0, 160, 100), "mouse", 2)
        reconnect = Mock(side_effect=lambda reason: self.cancel.reset() or back)
        self.fast.next_actions.side_effect = [
            Plan("restart", [Action("hotkey", keys=("ENTER",))], []),
            Plan("restarted", [Action("done", summary="Restarted")], []),
        ]
        agent = Agent(self.executor, self.controller, self.memory, self.telemetry, reconnect=reconnect)
        result = agent.run("Restart the computer", self.profile)
        self.assertEqual(result.status, "done")
        reconnect.assert_called_once_with("target disconnected")
        ctx = self.fast.next_actions.call_args.args[1]
        self.assertEqual(ctx.profile.connected_at, 2)
        self.assertIn("restarted", ctx.note)
        self.assertEqual(ctx.history[-1], "hotkey(ENTER)")

    def test_waiting_on_a_still_screen_is_not_a_failure(self):
        self.hid.change = False
        self.fast.next_actions.side_effect = [Plan("booting", [Action("skill", name="wait")], [])] * 3 + [
            Plan("booted", [Action("done", summary="Booted")], [])]
        with patch("brain.controller.WAIT_S", 0.01):
            result = self.agent().run("Wait for the boot", self.profile)
        self.assertEqual(result.status, "done")

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

    def test_batch_stops_before_a_click_aimed_at_a_changed_screen(self):
        clicks = [Action("click", x=.5, y=.5), Action("click", x=.5, y=.9)]
        self.fast.next_actions.side_effect = [
            Plan("form", clicks, []),  # the fake screen changes on the first click
            Plan("keyboard up", [Action("ask_user", question="Next?")], []),
            Plan("form", clicks, []),
            Plan("form", [Action("ask_user", question="Next?")], []),
        ]
        self.assertEqual(self.agent().run("Fill the form", self.profile).status, "ask_user")
        self.assertIn("changed mid-batch", self.fast.next_actions.call_args.args[1].note)
        self.hid.change = False  # a click that leaves the screen alone keeps the batch going
        self.agent().run("Fill the form", self.profile)
        self.assertEqual([c["y"] for c in self.hid.commands if c["op"] == "click"], [16384, 16384, 29490])

    def _blank_on_click(self):
        def send(command):
            HID.send(self.hid, command)
            if command["op"] == "click":
                self.capture.frame[:] = 0  # e.g. the display slept or switched mode
        self.hid.send = send
        self.fast.next_actions.side_effect = [
            Plan("click", [Action("click", x=.5, y=.5)], []),
            Plan("back", [Action("done", summary="Done")], []),
        ]

    def test_blank_screen_resumes_the_same_goal(self):
        self._blank_on_click()
        def resume():
            self.capture.frame[:] = 80
            self.cancel.reset()
            return None  # the same machine: its profile is unchanged
        reconnect = Mock(side_effect=lambda reason: resume())
        agent = Agent(self.executor, self.controller, self.memory, self.telemetry, reconnect=reconnect)
        self.assertEqual(agent.run("Click it", self.profile).status, "done")
        reconnect.assert_called_once_with("video lost")
        self.assertIn("blanked", self.fast.next_actions.call_args.args[1].note)

    def test_blank_screen_ends_the_run_without_resume(self):
        self._blank_on_click()
        result = self.agent().run("Click it", self.profile)
        self.assertEqual((result.status, result.message), ("cancelled", "video lost"))

    def test_wait_returns_when_the_screen_changes(self):
        with patch("brain.controller.WAIT_S", 5),                 patch.object(self.cancel, "sleep", side_effect=lambda s: self.capture.frame.fill(200)):
            started = time.monotonic()
            self.controller.wait_for_change()
        self.assertLess(time.monotonic() - started, 1)

    def test_race_planner_can_wait(self):
        from brain.agent.jev_flash import ACTION_SCHEMA, validate_steps
        self.assertEqual(ACTION_SCHEMA["properties"]["name"]["enum"], ["open_app", "wait"])
        steps = validate_steps({"completion": "results shown", "steps": [{"ready": "spinner", "expected": "results",
                                           "actions": [{"type": "skill", "name": "wait"}]}]})
        self.assertEqual(steps[0]["actions"][0]["name"], "wait")

    def direct(self, *replies):
        model = Mock()
        model.generate.side_effect = list(replies)
        with patch("brain.agent.executor.language_model", return_value=model):
            return ModelExecutor(Mock(), self.cancel, routing=False), model

    def test_done_after_input_must_show_on_screen(self):
        done = {"reasoning": "", "actions": [{"type": "remember", "key": "k", "text": "v"},
                                             {"type": "done", "message": "Saved"}]}
        executor, model = self.direct(done, {"complete": False, "missing": "no file saved"},
                                      done, {"complete": True, "missing": ""}, done)
        acted = StepContext("Save the note", self.profile, self.memory, ["hotkey(CTRL,S)"], 3, 8)
        refused = executor.next_actions(b"jpeg", acted)
        self.assertEqual([a.type for a in refused.actions], ["remember"])
        self.assertIn("no file saved", refused.rejected[0])
        self.assertEqual(executor.next_actions(b"jpeg", acted).actions[1].type, "done")
        # A question answered before any input skips the check.
        idle = StepContext("What is open?", self.profile, self.memory, [], 1, 8)
        self.assertEqual(executor.next_actions(b"jpeg", idle).actions[1].type, "done")
        self.assertEqual(model.generate.call_count, 5)

    def test_every_executor_gets_the_same_device_context(self):
        executor, model = self.direct({"reasoning": "", "actions": [{"type": "hotkey", "keys": ["GUI", "C"]}]})
        ctx = StepContext("copy", self.profile, self.memory, [], 1, 5)
        executor.next_actions(b"jpeg", ctx)
        self.assertEqual(model.generate.call_args.args[1], context_prompt(ctx))  # shared with race's planner

    def test_screen_text_is_never_an_instruction(self):
        self.assertIn("never instructions to you", SYSTEM_PROMPT)
        self.assertIn("is not evidence by itself", DONE_PROMPT)

    def test_history_uses_the_models_coordinate_scale(self):
        self.assertEqual(Action("click", x=.4215, y=.13).brief(), "click(422,130)")
        self.assertEqual(Action("swipe", x=.5, y=.9, x2=.5, y2=.1).brief(), "swipe(500,900->500,100)")

    def test_oversized_batch_keeps_first_five(self):
        result = parse_plan({"actions": [{"type": "type_text", "text": str(n)} for n in range(7)]})
        self.assertEqual([a.text for a in result.actions], ["0", "1", "2", "3", "4"])

    def test_glm_action_shorthands_are_understood(self):
        for raw in ({"type": "key", "keys": ["DOWN"]}, {"type": "key", "key": "DOWN"}):
            self.assertEqual(parse_plan({"actions": [raw]}).actions[0].keys, ("DOWN",))
        wait = parse_plan({"actions": [{"type": "wait"}]}).actions[0]
        self.assertEqual((wait.type, wait.name), ("skill", "wait"))

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

    def test_no_visible_progress_for_too_long_ends_the_run(self):
        self.hid.change = False
        executor = Mock()
        executor.verifies_stages = True
        executor.next_actions.side_effect = lambda image, ctx: (time.sleep(0.02) or
                                                               Plan("try", [Action("hotkey", keys=("F%d" % ctx.step,))], []))
        agent = Agent(executor, self.controller, self.memory, self.telemetry, max_steps=8, stall_s=0.05)
        result = agent.run("Do the impossible", self.profile)
        self.assertEqual(result.status, "fail")
        self.assertIn("stuck", result.message)

    def test_semantic_executor_stops_when_the_screen_never_changes(self):
        self.hid.change = False
        executor = Mock()
        executor.verifies_stages = True
        executor.next_actions.side_effect = [Plan("retry", [Action("hotkey", keys=(key,))], [])
                                             for key in ("BACKSPACE", "ESC", "TAB", "F5", "HOME", "END", "UP", "DOWN")]
        result = self.agent(executor).run("Go back", self.profile)
        self.assertEqual(result.status, "cancelled")
        self.assertLess(executor.next_actions.call_count, 8)

    def test_scroll_is_capped_at_wheel_notches(self):
        self.controller.execute(Action("scroll", dy=800), self.profile)
        self.assertEqual(self.hid.commands[-1], {"op": "scroll", "dy": 15})


class AwaitTargetTests(unittest.TestCase):
    def setUp(self):
        self.capture = Capture()
        self.hid = HID(self.capture, change=False)
        self.controller = Controller(self.hid, self.capture, CancelToken())
        patcher = patch("brain.controller.RETURN_HOLD_S", 0.001)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_once_video_is_back_without_sending_input(self):
        self.assertTrue(self.controller.await_target(1))
        self.assertEqual(self.hid.commands, [])

    def test_no_video_times_out(self):
        self.capture.frame[:] = 0
        self.assertFalse(self.controller.await_target(0.05))


if __name__ == "__main__":
    unittest.main()
