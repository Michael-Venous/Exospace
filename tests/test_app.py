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

        class FakeAgent:
            def __init__(self, *args):
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


if __name__ == "__main__":
    unittest.main()
