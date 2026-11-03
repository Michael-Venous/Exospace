"""The voice page's agent API over the optimized queue: requests in, spoken results and per-request logs out."""

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from bench.arch import serve_agent
from bench.arch.optimized_queue import create
from tests.test_optimized_queue import Target


class ServeAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
             patch("bench.arch.optimized_queue.OptimizedExecutor.start"), \
             patch("bench.arch.optimized_queue.OptimizedBrain.start"):
            self.agent = create(Target(), "deepseek/deepseek-v4.1-flash-1017", "", "test", self.dir, "")
        self.addCleanup(lambda: (self.agent.executor.checks.close(), self.agent.jev.close(),
                                 self.agent.brain.client.close(), self.agent.control.events_log.close()))
        self.events = serve_agent.Events()
        self.seen = []
        self.events.emit = lambda ev, **f: self.seen.append({"ev": ev, **f})

    def post(self, httpd, path, body):
        request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_port}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request, timeout=2).status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_each_goal_is_a_new_whole_request_and_stop_cancels_it(self):
        httpd = serve_agent.server(self.agent, self.agent.controller.capture, self.events, port=0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        first = self.agent.shared.request
        self.assertEqual(self.post(httpd, "/goal", {"text": "open firefox"}), 200)
        self.assertEqual((self.agent.shared.request, self.agent.brain.words, self.agent.brain.speaking),
                         (first + 1, "open firefox", False))
        self.assertEqual(self.post(httpd, "/goal", {"text": "  "}), 400)
        self.assertEqual(self.post(httpd, "/stop", {}), 200)
        self.assertEqual(self.agent.control.requests[self.agent.shared.request]["status"], "cancelled")

    def test_results_are_spoken_once_and_quiet_ends_still_end_the_wait(self):
        timings = self.dir / "timings.jsonl"
        reporter = serve_agent.Reporter(self.agent, self.events, self.dir, timings)
        brain, control = self.agent.brain, self.agent.control
        brain.end_turn(); brain.hear("what's my battery", False)  # as POST /goal does
        with self.agent.cond:
            control.finish("verified", "Your battery is at 63%.")
        reporter.step()
        reporter.step()  # nothing twice
        brain.end_turn(); brain.hear("minimize it", False)
        with self.agent.cond:
            control.finish("verified", "The terminal is minimized.", speak=False)  # a quiet done
        brain.end_turn()  # interrupted while listening: no line for it
        reporter.step()
        self.assertEqual([(e["status"], e["message"]) for e in self.seen if e["ev"] == "result"],
                         [("ask_user", "Your battery is at 63%."), ("done", "")])
        rows = [json.loads(line) for line in timings.read_text().splitlines()]
        self.assertEqual([r["status"] for r in rows], ["verified", "verified"])
        self.assertTrue((self.dir / "control.json").exists())

    def test_the_pi_is_the_default_target_and_the_laptop_is_opt_in(self):
        with patch("bench.arch.laptop.LaptopTarget", return_value="laptop rig"):
            self.assertEqual(serve_agent.make_target("laptop"), "laptop rig")
        with patch("bench.arch.pi_target.PiTarget", return_value="pi rig"), patch("brain.config.Config.load"):
            self.assertEqual(serve_agent.make_target("pi"), "pi rig")
        with self.assertRaises(SystemExit):
            serve_agent.make_target("phone")


if __name__ == "__main__":
    unittest.main()
