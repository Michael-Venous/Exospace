"""Voice layer: narration, forwarding to the agent, barge-in, and request guards."""

import importlib.util
import json
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, call

import cv2
import numpy as np

from brain.display.server import Command, DisplayServer
from brain.events import EventBus
from brain.voice.layer import Agent, filler, narrate, page_server

HAS_PLAYWRIGHT = importlib.util.find_spec("playwright") is not None


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

    def test_executor_news_is_spoken(self):
        self.assertEqual(narrate({"ev": "status", "message": "This one's harder. I'm thinking it through."}),
                         "This one's harder. I'm thinking it through.")

    def test_a_silent_request_is_acknowledged_then_given_progress(self):
        self.assertIsNone(filler(1.0, spoken=False))  # a fast request finishes before anything is said
        self.assertEqual(filler(3.0, spoken=False), "On it.")
        self.assertIsNone(filler(5.0, spoken=True))
        self.assertEqual(filler(11.0, spoken=True), "Still working on it.")

    def test_a_line_repeating_the_last_one_is_said_once(self):
        agent, lines = Agent("http://127.0.0.1:1"), queue.Queue()
        agent._say(lines, "The calculator is open")
        agent._say(lines, "The calculator is open")
        self.assertEqual(lines.qsize(), 1)

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
        self.agent.submit("and search lofi", continues=True)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="and search lofi", continues=True))

    def test_talking_interrupts_only_running_work(self):
        self.agent.interrupt()
        self.stop.assert_not_called()
        self.agent.busy = True
        self.agent.interrupt()
        self.stop.assert_called_once_with("barge-in")

    def test_follow_tracks_busy_state_and_narrates_only_new_events(self):
        lines: queue.Queue[dict] = queue.Queue()
        self.bus.publish("result", status="fail", message="before the layer connected")
        threading.Thread(target=self.agent.follow, args=(lines,), daemon=True).start()
        time.sleep(0.2)
        self.bus.publish("run", goal="delete the downloads folder")
        deadline = time.monotonic() + 2
        while not self.agent.busy and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.agent.busy)
        self.bus.publish("result", status="ask_user", message="This can't be undone. Should I go ahead?")
        self.assertEqual(lines.get(timeout=2), {"say": "This can't be undone. Should I go ahead?"})
        self.assertFalse(self.agent.busy)

    def test_describe_mode_describes_a_finished_request(self):
        describer = Mock()
        describer.describe.side_effect = lambda jpeg, request, outcome, emit: emit({"pcm": "AAAA", "rate": 24000}) or True
        agent, lines = Agent(self.display.url, describer), queue.Queue()
        agent.describe_on, agent._goal = True, "search youtube for lofi"
        self.bus.set_frame(b"jpeg")
        self.assertTrue(agent.describing({"ev": "result", "status": "done"}))
        self.assertFalse(agent.describing({"ev": "result", "status": "ask_user"}))  # questions are read verbatim
        agent._describe(lines, "Search results are shown")
        self.assertEqual([lines.get(), lines.get()], [{"show": "Search results are shown"}, {"pcm": "AAAA", "rate": 24000}])
        self.assertEqual(describer.describe.call_args.args[:3], (b"jpeg", "search youtube for lofi", "Search results are shown"))

    def test_a_failed_description_still_says_the_result(self):
        describer = Mock()
        describer.describe.return_value = False
        agent, lines = Agent(self.display.url, describer), queue.Queue()
        self.bus.set_frame(b"jpeg")
        agent._describe(lines, "Done.")
        self.assertEqual([lines.get(), lines.get()], [{"show": "Done."}, {"say": "Done."}])

    def test_screen_is_the_latest_frame(self):
        self.assertIsNone(self.agent.screen())
        self.bus.set_frame(b"jpeg")
        self.assertEqual(self.agent.screen(), b"jpeg")
        self.assertIsNone(Agent("http://127.0.0.1:9").screen())

    def test_screen_is_no_frame_when_the_agent_dies_mid_frame(self):
        class Dying(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"cut off")

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Dying)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.assertIsNone(Agent(f"http://127.0.0.1:{server.server_address[1]}").screen())


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

    def test_describe_toggle_needs_a_describer(self):
        self.assertEqual(self.post("/describe", {"on": True}), 200)
        self.assertTrue(self.agent.describe_on)
        self.agent.describer = None
        self.post("/describe", {"on": True})
        self.assertFalse(self.agent.describe_on)

    def test_talking_cuts_off_a_description(self):
        self.post("/talk", {})
        self.agent.describer.cancel.assert_called_once()

    def test_page_names_the_relay_port(self):
        with urllib.request.urlopen(self.base + "/", timeout=2) as response:
            self.assertIn(b"ws://${location.hostname}:8781", response.read())

    def test_segment_is_forwarded(self):
        self.assertEqual(self.post("/segment", {"text": "actually go back"}), 200)
        self.agent.submit.assert_called_once_with("actually go back", False)

    def test_a_later_segment_is_forwarded_as_a_continuation(self):
        self.assertEqual(self.post("/segment", {"text": "and search lofi", "continues": True}), 200)
        self.agent.submit.assert_called_once_with("and search lofi", True)
        self.agent.interrupt.assert_not_called()  # it joins the running request; the step in progress finishes

    def test_talk_barges_in(self):
        self.assertEqual(self.post("/talk", {}), 200)
        self.agent.interrupt.assert_called_once()

    def test_screen_is_relayed(self):
        self.agent.screen.return_value = b"jpeg"
        with urllib.request.urlopen(self.base + "/screen.jpg?t=1", timeout=2) as response:
            self.assertEqual((response.headers["Content-Type"], response.read()), ("image/jpeg", b"jpeg"))
        self.agent.screen.return_value = None
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + "/screen.jpg", timeout=2)
        self.assertEqual(caught.exception.code, 503)

    def test_cross_site_and_empty_requests_are_refused(self):
        self.assertEqual(self.post("/segment", {"text": "send it"}, origin=False), 403)
        self.assertEqual(self.post("/segment", {"text": ""}), 400)
        self.agent.submit.assert_not_called()


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright is only installed with the browser extra")
class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import Error, sync_playwright
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        try:
            cls.browser = playwright.chromium.launch()
        except Error as exc:
            raise unittest.SkipTest(f"no Chromium to drive: {exc}")
        cls.addClassCleanup(cls.browser.close)

    def test_reopening_the_view_at_once_runs_one_frame_loop(self):
        agent = Mock()
        agent.screen.return_value = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
        server = page_server(agent, 0, 8781, queue.Queue())
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        page = self.browser.new_page()
        self.addCleanup(page.close)
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        page.click("#device")
        while agent.screen.call_count < 4:
            time.sleep(0.01)
        page.wait_for_timeout(100)  # the loop is now asleep between two frames
        page.evaluate("() => { dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape'})); device.click(); }")
        reopened = agent.screen.call_count
        page.wait_for_timeout(1500)
        self.assertLessEqual(agent.screen.call_count - reopened, 8)  # one loop asks every 250 ms: about 6

    def test_clauses_end_at_and_then_or_a_comma(self):
        agent = Mock()
        page = self.open_page(agent)
        split = lambda text: page.evaluate("t => clausesOf(t)", text)
        self.assertEqual(split("open google chrome, open youtube, search horses"),
                         [["open google chrome", "open youtube"], "search horses"])
        self.assertEqual(split("Open up Google Chrome and then search up YouTube."),
                         [["Open up Google Chrome"], "search up YouTube."])
        self.assertEqual(split("go to youtube.com and"), [["go to youtube.com"], ""])
        self.assertEqual(split("open google chrome."), [[], "open google chrome."])  # not ended: maybe more words
        self.assertEqual(split("search for horses and click the third video"),
                         [["search for horses"], "click the third video"])
        self.assertEqual(split("type in hello, delete hello and"), [["type in hello", "delete hello"], ""])

    def test_a_clause_is_sent_while_the_user_keeps_talking(self):
        agent = Mock()
        page = self.open_page(agent)
        page.evaluate("press()")
        page.wait_for_function("window.speech")

        def heard(words, final=False, pause=False):
            page.evaluate("m => speech.onmessage({data: JSON.stringify(m)})", {
                "type": "Results", "is_final": final, "speech_final": pause,
                "channel": {"alternatives": [{"transcript": words}]}})

        def submitted(count):
            deadline = time.monotonic() + 5
            while agent.submit.call_count < count and time.monotonic() < deadline:
                time.sleep(0.01)
            return agent.submit.call_args_list

        heard("open up google chrome")
        heard("open up google chrome and")  # no pause after "chrome" in the Pi's recorded requests
        self.assertEqual(submitted(1), [call("open up google chrome", False)])
        heard("Open up Google Chrome and search up", final=True)
        heard("YouTube and", final=True)
        self.assertEqual(submitted(2), [call("open up google chrome", False), call("search up YouTube", True)])
        heard("click the first video.", final=True, pause=True)
        self.assertEqual(submitted(3)[2], call("click the first video.", True))

    def open_page(self, agent):
        agent.screen.return_value = None  # no device connected
        server = page_server(agent, 0, 8781, queue.Queue())
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        page = self.browser.new_page()
        self.addCleanup(page.close)
        page.add_init_script(FAKE_SPEECH)
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        return page


# A silent mic and a speech-service socket the test answers for, as window.speech.
FAKE_SPEECH = """
navigator.mediaDevices.getUserMedia = async () => new AudioContext().createMediaStreamDestination().stream;
window.WebSocket = class { constructor() { window.speech = this; this.readyState = 0; } send() {} close() {} };
"""


if __name__ == "__main__":
    unittest.main()
