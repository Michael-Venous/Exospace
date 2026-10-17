"""app.main wiring: which screens get classified, and which connection they belong to."""

import itertools
import unittest
from unittest.mock import Mock, patch

from brain import app
from brain.types import DeviceProfile, Rect


class RedetectTests(unittest.TestCase):
    def test_a_redetect_keeps_the_connection_and_a_reconnect_renews_it(self):
        stamps, monitor, seen = itertools.count(1), Mock(generation=0, connected=True), []
        detect = Mock(side_effect=lambda *a, **k: DeviceProfile("bios", Rect(0, 0, 10, 10), "mouse", next(stamps)))
        monitor.wait_connected.side_effect = lambda: monitor.generation

        class FakeAgent:
            def __init__(self, *args, deadline_s=None):
                self.reconnect, self.redetect = args[-2:]

            def run(self, goal, profile, task_id):
                seen.append((goal, profile.connected_at, detect.call_count))
                if goal == "restart":
                    seen.append(self.redetect().connected_at)  # the same connection
                    monitor.generation += 1
                    seen.append(self.reconnect("target disconnected").connected_at)
                return Mock(status="done", message="ok")

        with patch.object(app.Config, "load", return_value=Mock(executor="direct", target_os="")), \
                patch.object(app, "build_executor"), patch("brain.qemu.QemuTarget"), \
                patch.object(app, "Controller") as controller, \
                patch.object(app, "ConnectionMonitor", return_value=monitor), \
                patch.object(app, "Memory"), patch.object(app, "Telemetry"), \
                patch.object(app, "detect_profile", detect), patch.object(app, "Agent", FakeAgent), \
                patch("builtins.input", side_effect=["first", "second", "restart", "after", EOFError]), \
                patch("builtins.print"):
            controller.return_value.await_target.return_value = True
            self.assertEqual(app.main(["--backend", "qemu"]), 0)
        # No classification at goal start: the loop classifies a bios screen again once it changes a lot.
        self.assertEqual(seen, [("first", 1, 1), ("second", 1, 1), ("restart", 1, 1), 1, 3, ("after", 3, 3)])


class DisplayGoalTests(unittest.TestCase):
    def run_goals(self, statuses):
        goals = []

        class FakeAgent:
            def __init__(self, *args, deadline_s=None):
                pass

            def run(self, goal, profile, task_id):
                goals.append(goal)
                return Mock(status=statuses[len(goals) - 1], message="")

        commands = [app.Command("goal", text=t) for t in ("select guest mode", "search Genghis Khan")]
        monitor = Mock(generation=0, connected=True)
        monitor.wait_connected.return_value = 0
        with patch.object(app.Config, "load", return_value=Mock(executor="direct", target_os="")), \
                patch.object(app, "build_executor"), patch("brain.qemu.QemuTarget"), patch.object(app, "Controller"), \
                patch.object(app, "ConnectionMonitor", return_value=monitor), patch.object(app, "DisplayServer"), \
                patch.object(app, "_show_screen"), patch.object(app, "Memory"), patch.object(app, "Telemetry"), \
                patch.object(app, "detect_profile"), patch.object(app, "Agent", FakeAgent), \
                patch.object(app, "_next_command", side_effect=[*commands, EOFError]), \
                patch("builtins.input", side_effect=EOFError), patch("builtins.print"):
            self.assertEqual(app.main(["--backend", "qemu", "--display", "8770"]), 0)
        return goals

    def test_a_request_after_an_interruption_is_new(self):
        self.assertEqual(self.run_goals(["cancelled", "done"]), ["select guest mode", "search Genghis Khan"])

    def test_an_answer_to_a_question_continues_the_goal(self):
        self.assertEqual(self.run_goals(["ask_user", "done"])[1], "select guest mode\nUser correction: search Genghis Khan")


if __name__ == "__main__":
    unittest.main()
