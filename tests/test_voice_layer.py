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
from brain.voice.layer import Agent, filler, narrate, page_server, step_label


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

    def test_each_step_becomes_a_checklist_line(self):
        def label(reasoning, *actions):
            return step_label({"ev": "plan", "reasoning": reasoning, "actions": list(actions)})

        self.assertEqual(label("Stage 1/2: The reply window is open", "click(100,200)"), "The reply window is open")
        self.assertEqual(label("Jev next step: fill", "click(1,2)", "hotkey(CTRL+A)", "type_text('lofi')"),
                         "Type 'lofi'")
        self.assertEqual(label("Jev next step: click", "click(1,2)"), "Click")
        self.assertEqual(label("Open the app launcher", "hotkey(GUI)"), "Open the app launcher")
        self.assertIsNone(label("Jev verified the completed goal", "done"))  # the run's end, not a step
        self.assertIsNone(label("Waiting for the app", "skill(wait {})"))
        self.assertIsNone(step_label({"ev": "action", "action": "click(1,2)"}))



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

    def test_a_finished_command_acts_mid_sentence_and_the_rest_follows_on(self):
        self.agent.busy = True
        for heard in ("open a new tab", "open a new tab and", "open a new tab and search"):
            self.agent.hearing(heard, False)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="open a new tab"))
        self.agent.heard("Open a new tab, and search Genghis Khan.", False)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="search Genghis Khan.", after=True))
        self.stop.assert_not_called()  # the tab it opened is kept
        self.assertTrue(self.commands.empty())

    def test_a_segment_that_was_all_commands_sends_nothing_more(self):
        self.agent.hearing("press ctrl c then press", False)
        self.agent.heard("Press ctrl c.", False)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="press ctrl c"))
        self.agent.hearing("press escape and", True)  # a later segment's command follows on
        self.agent.heard("press escape and", True)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="press escape", after=True))
        self.assertTrue(self.commands.empty())

    def test_later_speech_follows_on_unless_it_corrects(self):
        self.agent.busy = True
        self.agent.heard("and then search lofi", True)
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="search lofi", after=True))
        self.stop.assert_not_called()
        self.agent.heard("no, go back", True)
        self.stop.assert_called_once_with("barge-in")
        self.assertEqual(self.commands.get(timeout=2), Command("goal", text="no, go back", continues=True))

    def test_model_work_waits_for_the_pause(self):
        for heard in ("click the red", "click the red button and", "scroll down and", "press control and"):
            self.agent.hearing(heard, False)
        self.assertTrue(self.commands.empty())

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
        self.bus.publish("plan", step=1, reasoning="Jev next step: enter", actions=["hotkey(ENTER)"])
        self.bus.publish("result", status="ask_user", message="This can't be undone. Should I go ahead?")
        got = [lines.get(timeout=2) for _ in range(6)]
        self.assertEqual(got, [{"busy": False}, {"busy": True}, {"step": "Press Enter"}, {"finished": "ask_user"},
                               {"busy": False}, {"say": "This can't be undone. Should I go ahead?"}])
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


class PageServerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock(base="http://127.0.0.1:8770", busy=True)
        self.updates: queue.Queue[dict] = queue.Queue()
        self.server = page_server(self.agent, 0, 8781, self.updates)
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
            page = response.read()
        self.assertIn(b"ws://${location.hostname}:8781", page)

    def test_every_open_page_gets_each_update(self):
        # A stale stream left by a reload must not swallow the update the live page needs.
        with urllib.request.urlopen(self.base + "/narration", timeout=2) as stale, \
                urllib.request.urlopen(self.base + "/narration", timeout=2) as live:
            time.sleep(0.1)  # both streams registered
            self.updates.put({"say": "Done."})
            for stream in (stale, live):
                self.assertEqual([stream.readline() for _ in range(2)], [b'data: {"say": "Done."}\n', b"\n"])

    def test_segment_is_forwarded(self):
        self.assertEqual(self.post("/segment", {"text": "actually go back"}), 200)
        self.agent.heard.assert_called_once_with("actually go back", False)

    def test_a_later_segment_is_forwarded_as_a_continuation(self):
        self.assertEqual(self.post("/segment", {"text": "and search lofi", "continues": True}), 200)
        self.agent.heard.assert_called_once_with("and search lofi", True)

    def test_speech_still_coming_in_is_forwarded(self):
        self.assertEqual(self.post("/partial", {"text": "open a new tab and"}), 200)
        self.agent.hearing.assert_called_once_with("open a new tab and", False)

    def test_talk_barges_in(self):
        self.assertEqual(self.post("/talk", {}), 200)
        self.agent.listen.assert_called_once()

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
        self.assertEqual(self.post("/partial", {"text": "x" * 501}), 400)
        self.agent.heard.assert_not_called()
        self.agent.hearing.assert_not_called()


if __name__ == "__main__":
    unittest.main()
