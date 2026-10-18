"""app.main wiring: when a target is classified, and which connection a goal belongs to."""

import itertools
import unittest
from unittest.mock import Mock, patch

from brain import app
from brain.types import DeviceProfile, Rect


class ReconnectTests(unittest.TestCase):
    def test_a_reconnect_renews_the_connection_once(self):
        stamps, monitor, seen = itertools.count(1), Mock(generation=0, connected=True), []
        detect = Mock(side_effect=lambda *a, **k: DeviceProfile("windows", Rect(0, 0, 10, 10), "mouse", next(stamps)))
        monitor.wait_connected.side_effect = lambda: monitor.generation

        class FakeAgent:
            def __init__(self, *args):
                self.reconnect = args[-1]

            def run(self, goal, profile, task_id):
                seen.append((goal, profile.connected_at, detect.call_count))
                if goal == "restart":
                    monitor.generation += 1
                    seen.append(self.reconnect("target disconnected").connected_at)
                return Mock(status="done", message="ok")

        with patch.object(app.Config, "load", return_value=Mock(executor="direct", target_os="")), \
                patch.object(app, "build_executor"), patch("brain.browser.BrowserTarget"), \
                patch.object(app, "Controller") as controller, \
                patch.object(app, "ConnectionMonitor", return_value=monitor), \
                patch.object(app, "Memory"), patch.object(app, "Telemetry"), \
                patch.object(app, "detect_profile", detect), patch.object(app, "Agent", FakeAgent), \
                patch("builtins.input", side_effect=["first", "second", "restart", "after", EOFError]), \
                patch("builtins.print"):
            controller.return_value.await_target.return_value = True
            self.assertEqual(app.main(["--backend", "browser", "--url", "about:blank"]), 0)
        # Classified once per connection: not at each goal, and once more after the reconnect.
        self.assertEqual(seen, [("first", 1, 1), ("second", 1, 1), ("restart", 1, 1), 2, ("after", 2, 2)])


class DisplayGoalTests(unittest.TestCase):
    def run_goals(self, statuses, continues=False):
        goals = []

        class FakeAgent:
            def __init__(self, *args):
                pass

            def run(self, goal, profile, task_id):
                goals.append(goal)
                return Mock(status=statuses[len(goals) - 1], message="")

        commands = [app.Command("goal", text="select guest mode"),
                    app.Command("goal", text="search Genghis Khan", continues=continues)]
        monitor = Mock(generation=0, connected=True)
        monitor.wait_connected.return_value = 0
        with patch.object(app.Config, "load", return_value=Mock(executor="direct", target_os="")), \
                patch.object(app, "build_executor"), patch("brain.browser.BrowserTarget"), patch.object(app, "Controller"), \
                patch.object(app, "ConnectionMonitor", return_value=monitor), patch.object(app, "DisplayServer"), \
                patch.object(app, "_show_screen"), patch.object(app, "Memory"), patch.object(app, "Telemetry"), \
                patch.object(app, "detect_profile"), patch.object(app, "Agent", FakeAgent), \
                patch.object(app, "_next_command", side_effect=[*commands, EOFError]), \
                patch("builtins.input", side_effect=EOFError), patch("builtins.print"):
            self.assertEqual(app.main(["--backend", "browser", "--url", "about:blank", "--display", "8770"]), 0)
        return goals

    def test_a_request_after_an_interruption_is_new(self):
        self.assertEqual(self.run_goals(["cancelled", "done"]), ["select guest mode", "search Genghis Khan"])

    def test_a_later_segment_of_one_utterance_continues_the_goal(self):
        self.assertEqual(self.run_goals(["cancelled", "done"], continues=True)[1],
                         "select guest mode\nUser correction: search Genghis Khan")

    def test_an_answer_to_a_question_continues_the_goal(self):
        self.assertEqual(self.run_goals(["ask_user", "done"])[1], "select guest mode\nUser correction: search Genghis Khan")


if __name__ == "__main__":
    unittest.main()
