"""Display event bus, local display server, and agent event publishing."""

import http.client
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from brain.agent.executor import Plan
from brain.agent.loop import Agent
from brain.agent.memory import Memory
from brain.cancel import CancelToken
from brain.controller import Controller, SettleConfig
from brain.display.server import Command, DisplayServer
from brain.events import CLIENT_BUFFER, EventBus
from brain.telemetry import Telemetry
from brain.types import Action, DeviceProfile, Rect
from test_runtime import HID, Capture


def drain(client):
    events = []
    while not client.empty():
        events.append(client.get_nowait())
    return events


class EventBusTests(unittest.TestCase):
    def test_late_client_gets_history_and_current_frame(self):
        bus = EventBus()
        bus.publish("run", goal="open mail")
        bus.set_frame(b"jpeg")
        events = drain(bus.subscribe())
        self.assertEqual(events[0]["ev"], "run")
        self.assertEqual(events[1:], [{"ev": "frame", "id": 1}])  # current frame only, not every frame
        self.assertEqual(bus.frame(), (1, b"jpeg"))

    def test_slow_client_drops_events_instead_of_blocking(self):
        bus = EventBus()
        client = bus.subscribe()
        for i in range(CLIENT_BUFFER + 10):
            bus.publish("step", i=i)
        self.assertEqual(client.qsize(), CLIENT_BUFFER)


class DisplayServerTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.commands = queue.Queue()
        self.stop = Mock()
        self.server = DisplayServer(self.bus, self.commands, self.stop, port=0)
        self.server.start()
        self.addCleanup(self.server.close)
        self.host, self.port = self.server.url.removeprefix("http://").split(":")

    def request(self, method, path, body=None, **headers):
        conn = http.client.HTTPConnection(self.host, int(self.port), timeout=2)
        self.addCleanup(conn.close)
        conn.request(method, path, json.dumps(body) if body is not None else None,
                     {"Content-Type": "application/json", **headers})
        response = conn.getresponse()
        return response.status, response.read()

    def test_other_websites_cannot_drive_the_agent(self):
        evil = {"Origin": "http://evil.example"}
        self.assertEqual(self.request("POST", "/goal", {"text": "delete everything"}, **evil)[0], 403)
        self.assertEqual(self.request("POST", "/goal", {"text": "x"}, **{"Content-Type": "text/plain"})[0], 400)
        same_page = {"Origin": f"http://localhost:{self.port}"}
        self.assertEqual(self.request("POST", "/goal", {"text": "open mail"}, **same_page)[0], 200)
        self.assertEqual(self.commands.qsize(), 1)

    def test_dns_rebinding_hosts_are_refused(self):
        self.bus.set_frame(b"\xff\xd8jpeg")
        self.assertEqual(self.request("GET", "/screen.jpg", Host=f"evil.example:{self.port}")[0], 403)
        self.assertEqual(self.request("GET", "/events", Host="evil.example")[0], 403)
        self.assertEqual(self.request("POST", "/stop", {}, Host="evil.example")[0], 403)
        self.stop.assert_not_called()

    def test_refuses_to_listen_on_all_interfaces(self):
        with self.assertRaises(ValueError):
            DisplayServer(self.bus, self.commands, self.stop, host="0.0.0.0", port=0)

    def test_page_is_served(self):
        status, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Exospace Display</title>", body)

    def test_goal_and_taps_are_queued_for_the_main_loop(self):
        self.assertEqual(self.request("POST", "/goal", {"text": " open mail "})[0], 200)
        self.assertEqual(self.request("POST", "/tap", {"x": 0.25, "y": 0.75, "mode": "point"})[0], 200)
        self.assertEqual(self.commands.get_nowait(), Command("goal", text="open mail"))
        self.assertEqual(self.commands.get_nowait(), Command("point", x=0.25, y=0.75))

    def test_invalid_requests_are_rejected(self):
        self.assertEqual(self.request("POST", "/tap", {"x": 1.5, "y": 0.5})[0], 400)
        self.assertEqual(self.request("POST", "/tap", {"x": 0.5, "y": 0.5, "mode": "drag"})[0], 400)
        self.assertEqual(self.request("POST", "/goal", {"text": "  "})[0], 400)
        self.assertTrue(self.commands.empty())

    def test_stop_uses_the_cancel_path_and_drops_queued_requests(self):
        client = self.bus.subscribe()
        self.commands.put(Command("goal", text="an older request"))
        self.assertEqual(self.request("POST", "/stop", {})[0], 200)
        self.stop.assert_called_once_with("barge-in")
        self.assertTrue(self.commands.empty())
        self.assertEqual([e["ev"] for e in drain(client)], ["stopped"])

    def test_screen_is_empty_until_a_frame_arrives(self):
        self.assertEqual(self.request("GET", "/screen.jpg")[0], 204)
        self.bus.set_frame(b"\xff\xd8jpeg")
        self.assertEqual(self.request("GET", "/screen.jpg"), (200, b"\xff\xd8jpeg"))

    def test_event_stream_replays_history(self):
        self.bus.publish("run", goal="open mail")
        conn = http.client.HTTPConnection(self.host, int(self.port), timeout=2)
        self.addCleanup(conn.close)
        conn.request("GET", "/events")
        response = conn.getresponse()
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        line = response.fp.readline()
        self.assertEqual(json.loads(line.removeprefix(b"data: "))["goal"], "open mail")


class AgentEventTests(unittest.TestCase):
    def test_run_publishes_plan_actions_frame_and_result(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        capture = Capture()
        cancel = CancelToken()
        controller = Controller(HID(capture), capture, cancel,
                                SettleConfig(min_s=0, react_s=0, load_stable_s=0, load_react_s=0, stable_s=0, timeout_s=.1, poll_s=.001))
        executor = Mock(spec=["next_actions"])
        executor.next_actions.side_effect = [
            Plan("Click Send", [Action("click", x=0.5, y=0.25)], []),
            Plan("Sent", [Action("done", summary="Sent")], []),
        ]
        bus = EventBus()
        client = bus.subscribe()
        agent = Agent(executor, controller, Memory(root / "session.json"),
                      Telemetry(root / "results.csv", "test"), max_steps=4, events=bus)
        result = agent.run("send it", DeviceProfile("macos", Rect(0, 0, 160, 100), "mouse", 1))
        self.assertEqual(result.status, "done")
        events = drain(client)
        names = [e["ev"] for e in events if e["ev"] != "frame"]
        self.assertEqual(names, ["run", "plan", "action", "plan", "result"])
        plan = next(e for e in events if e["ev"] == "plan")
        self.assertEqual(plan["points"], [[0.5, 0.25]])
        self.assertIsNone(plan["scores"])  # this executor has no Jev
        action = next(e for e in events if e["ev"] == "action")
        self.assertEqual((action["point"], action["ok"]), ([0.5, 0.25], True))
        self.assertGreater(bus.frame()[0], 0)
        self.assertTrue(bus.frame()[1].startswith(b"\xff\xd8"))  # JPEG


    def test_batch_skipped_mid_way_reports_only_sent_actions(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        capture = Capture()
        controller = Controller(HID(capture), capture, CancelToken(),
                                SettleConfig(min_s=0, react_s=0, load_stable_s=0, load_react_s=0, stable_s=0, timeout_s=.1, poll_s=.001))
        executor = Mock(spec=["next_actions"])
        executor.next_actions.side_effect = [  # the fake screen changes on the first click
            Plan("Fill", [Action("click", x=0.5, y=0.25), Action("click", x=0.5, y=0.75)], []),
            Plan("Ask", [Action("ask_user", question="Next?")], []),
        ]
        bus = EventBus()
        client = bus.subscribe()
        agent = Agent(executor, controller, Memory(root / "session.json"),
                      Telemetry(root / "results.csv", "test"), max_steps=4, events=bus)
        agent.run("fill it", DeviceProfile("macos", Rect(0, 0, 160, 100), "mouse", 1))
        actions = [e for e in drain(client) if e["ev"] == "action"]
        self.assertEqual([(a["point"], a["ok"]) for a in actions], [([0.5, 0.25], True)])

if __name__ == "__main__":
    unittest.main()
