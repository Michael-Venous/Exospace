"""Home as a global push-to-talk for the agent server: press stops and listens, release sends, lines are spoken."""

import unittest
from unittest.mock import Mock, patch

from bench.arch import home_talk


class HomeTalkTests(unittest.TestCase):
    def setUp(self):
        self.sent, self.speaker = [], Mock()
        self.words = "open firefox"
        utterance = Mock(finish=lambda: self.words)
        self.home = home_talk.HomeTalk(lambda path, body=None: self.sent.append((path, body)), self.speaker,
                                       lambda: utterance)

    def test_press_stops_the_agent_and_release_sends_the_whole_request(self):
        self.home.changed(True)
        self.home.changed(True)  # a second down while held (another keyboard) changes nothing
        self.assertEqual(self.sent, [("/listen", None)])
        self.speaker.stop.assert_called_once()
        self.home.changed(False)
        self.assertEqual(self.sent[-1], ("/goal", {"text": "open firefox"}))

    def test_nothing_heard_drops_the_request(self):
        self.words = ""
        self.home.changed(True)
        self.home.changed(False)
        self.assertEqual(self.sent, [("/listen", None), ("/stop", None)])

    def test_the_agents_lines_and_quiet_ends_are_spoken_but_never_over_the_user(self):
        self.home.hear({"ev": "run", "goal": "open firefox"})
        self.home.hear({"ev": "result", "status": "ask_user", "message": "Firefox is open."})
        self.home.hear({"ev": "result", "status": "done", "message": ""})  # a task that finished quietly
        self.home.hear({"ev": "result", "status": "cancelled", "message": ""})  # the user stopped it: no word
        self.home.hear({"ev": "result", "status": "fail", "message": ""})
        self.home.changed(True)
        self.home.hear({"ev": "result", "status": "done", "message": ""})
        self.assertEqual([c.args[0] for c in self.speaker.say.call_args_list],
                         ["Firefox is open.", "Done.", "Sorry, I couldn't do that."])

    def test_an_agent_that_is_not_up_does_not_break_the_key(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            home_talk.post("/stop")  # reported, not raised


if __name__ == "__main__":
    unittest.main()
