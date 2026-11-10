"""Home as a global push-to-talk for the agent server: press stops and listens, release sends, lines are spoken."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bench.arch import home_talk
from brain.voice.controls import Controls


class HomeTalkTests(unittest.TestCase):
    def setUp(self):
        self.sent, self.speaker = [], Mock()
        self.words = "open firefox"
        utterance = Mock(finish=lambda: self.words)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.home = home_talk.HomeTalk(lambda path, body=None: self.sent.append((path, body)), self.speaker,
                                       lambda: utterance, Controls(Path(folder.name) / "voice.json"))

    def say(self, words):
        """One press and release of Home with `words` heard; the box's own answer, if any, has been spoken."""
        self.words = words
        self.home.changed(True)
        self.home.changed(False)
        if self.home.answering:
            self.home.answering.join(2)

    def test_a_command_for_the_box_is_answered_aloud_and_never_reaches_the_agent(self):
        self.say("Talk at 2x.")
        self.assertEqual(self.sent, [("/listen", None), ("/stop", None)])
        self.speaker.say.assert_called_with("Speaking at 2 times speed.", 2.0)
        self.home.hear({"ev": "result", "status": "ask_user", "message": "Firefox is open."})
        self.speaker.say.assert_called_with("Firefox is open.", 2.0)
        self.say("Repeat that.")
        self.speaker.say.assert_called_with("Firefox is open.", 2.0)
        self.assertNotIn("/goal", [path for path, _ in self.sent])
        self.say("type talk faster in the doc")
        self.assertEqual(self.sent[-1], ("/goal", {"text": "type talk faster in the doc"}))

    def test_dictation_sends_typing_goals_reads_them_back_and_skips_done(self):
        self.say("Start dictating.")
        self.speaker.say.assert_called_with("Dictating. Say stop dictating when you're done.", 1.0)
        self.say("Dear Janet,")
        path, body = self.sent[-1]
        self.assertEqual(path, "/goal")
        self.assertTrue(body["text"].startswith("Dictation: type exactly"))
        self.assertTrue(body["text"].endswith(": Dear Janet,"))
        self.speaker.say.assert_called_with("Dear Janet,", 1.0)
        self.home.hear({"ev": "result", "status": "done", "message": ""})  # read back already: no "Done."
        self.speaker.say.assert_called_with("Dear Janet,", 1.0)
        self.say("Stop dictating.")
        self.speaker.say.assert_called_with("Stopped dictating.", 1.0)
        self.say("open firefox")
        self.assertEqual(self.sent[-1], ("/goal", {"text": "open firefox"}))

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

    def test_private_mode_stops_the_agent_and_never_sends_or_prints_the_words(self):
        with patch("builtins.print") as printed:
            self.say("I'll type my password.")
            self.assertEqual(self.sent, [("/listen", None), ("/stop", None)])
            self.speaker.say.assert_called_with(
                "I'm not looking or listening. Type it, then press the button and say done.", 1.0)
            self.say("my password is hunter2")
            self.say("open firefox")
        self.assertNotIn("/goal", [path for path, _ in self.sent])
        shown = " ".join(str(arg) for call in printed.call_args_list for arg in call.args)
        self.assertNotIn("hunter2", shown)
        self.assertNotIn("open firefox", shown)
        self.assertIn("you: (private)", shown)
        self.say("I'm done")
        path, body = self.sent[-1]
        self.assertEqual(path, "/goal")
        self.assertTrue(body["text"].startswith("The user finished typing their password themselves."))

    def test_cancel_leaves_private_mode_and_sends_nothing(self):
        self.say("stop looking")
        self.say("cancel")
        self.assertNotIn("/goal", [path for path, _ in self.sent])
        self.say("open firefox")
        self.assertEqual(self.sent[-1], ("/goal", {"text": "open firefox"}))

    def test_an_agent_that_is_not_up_does_not_break_the_key(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            home_talk.post("/stop")  # reported, not raised


if __name__ == "__main__":
    unittest.main()
