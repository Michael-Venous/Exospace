"""Voice layer: narration, forwarding to the agent, barge-in, and request guards."""

import json
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock

from brain.display.server import Command, DisplayServer
from brain.events import EventBus
from brain.voice.layer import Agent, narrate, page_server


class NarrationTests(unittest.TestCase):
    def test_questions_are_always_spoken(self):
        line = narrate({"ev": "result", "status": "ask_user", "message": "This can't be undone. Go ahead?"})
        self.assertEqual(line, "This can't be undone. Go ahead?")

    def test_results(self):
        self.assertEqual(narrate({"ev": "result", "status": "done", "message": "Verified: Janet's email is open"}),
                         "Janet's email is open")
        self.assertEqual(narrate({"ev": "result", "status": "done", "message": "x" * 300}), "Done.")
        self.assertEqual(narrate({"ev": "result", "status": "cancelled", "message": "barge-in"}), "Stopped.")
        self.assertEqual(narrate({"ev": "result", "status": "fail", "message": "TimeoutError"}),
                         "Sorry, I couldn't do that.")

    def test_stages_are_brief_and_fast_actions_are_silent(self):
        self.assertEqual(narrate({"ev": "plan", "reasoning": "Stage 1/2: The reply window is open"}),
                         "The reply window is open")
        self.assertIsNone(narrate({"ev": "plan", "reasoning": "Jev direct action"}))
        self.assertIsNone(narrate({"ev": "action", "action": "click(0.1,0.2)"}))


class AgentLinkTests(unittest.TestCase):
    def setUp(self):
        self.bus, self.commands, self.stop = EventBus(), queue.Queue(), Mock()
        self.display = DisplayServer(self.bus, self.commands, self.stop, "127.0.0.1", 0)
        self.display.start()
        self.addCleanup(self.display.close)
        self.agent = Agent(self.display.url)

    def test_segments_become_agent_goals(self):
        self.agent.submit("open chrome")
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="open chrome"))

    def test_talking_interrupts_only_running_work(self):
        self.agent.interrupt()
        self.stop.assert_not_called()
        self.agent.busy = True
        self.agent.interrupt()
        self.stop.assert_called_once_with("barge-in")

    def test_follow_tracks_busy_state_and_narrates(self):
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self.agent.follow, args=(lines,), daemon=True).start()
        time.sleep(0.2)
        self.bus.publish("run", goal="delete the downloads folder")
        deadline = time.monotonic() + 2
        while not self.agent.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.agent.busy)
        self.bus.publish("result", status="ask_user", message="This can't be undone. Should I go ahead?")
        self.assertEqual(lines.get(timeout=2), "This can't be undone. Should I go ahead?")
        self.assertFalse(self.agent.busy)


class PageServerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.server = page_server(self.agent, 0, 8781, queue.Queue())
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def post(self, path, body, origin=True):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = self.base
        request = urllib.request.Request(self.base + path, json.dumps(body).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_page_names_the_relay_port(self):
        with urllib.request.urlopen(self.base + "/", timeout=2) as response:
            self.assertIn(b"ws://${location.hostname}:8781", response.read())

    def test_segment_is_forwarded(self):
        self.assertEqual(self.post("/segment", {"text": "actually go back"}), 200)
        self.agent.submit.assert_called_once_with("actually go back")

    def test_talk_barges_in(self):
        self.assertEqual(self.post("/talk", {}), 200)
        self.agent.interrupt.assert_called_once()

    def test_cross_site_and_empty_requests_are_refused(self):
        self.assertEqual(self.post("/segment", {"text": "send it"}, origin=False), 403)
        self.assertEqual(self.post("/segment", {"text": ""}), 400)
        self.agent.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
