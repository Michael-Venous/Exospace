"""Per-request recordings and the read-only phone player server."""

import http.client
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.display.recordings import Recorder, RecordingsServer, read_log, recordings, summary
from brain.events import EventBus


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bus = EventBus()
        self.recorder = Recorder(self.bus, self.root)
        self.addCleanup(self.recorder._close)

    def pump(self):
        while not self.recorder._client.empty():
            self.recorder.handle(self.recorder._client.get_nowait())

    def request(self, run_id, goal="open mail", test=False):
        self.bus.publish("run", goal=goal, run_id=run_id, test=test)
        self.bus.set_frame(b"live")
        self.pump()  # the recorder keeps up, so it sees each frame
        step = self.bus.set_frame(b"step", keep=True)
        self.bus.publish("plan", step=1, reasoning="click Mail", actions=["click(0.5,0.5)"], points=[[0.5, 0.5]],
                         rejected=[], frame=step, trace=[{"label": "triage", "answers": {}}], t_model=1.2)
        self.bus.publish("action", step=1, action="click(0.5,0.5)", point=[0.5, 0.5], ok=True)
        self.bus.publish("result", status="done", message="Opened Mail", steps=1, total_s=2.0, trace=[])
        self.pump()

    def test_each_request_is_saved_with_its_events_and_frames(self):
        self.bus.set_frame(b"idle")  # before any request: not recorded
        self.request("abc123")
        [name] = recordings(self.root)
        folder = self.root / name
        events = read_log(folder / "log.jsonl")
        self.assertEqual([e["ev"] for e in events], ["run", "frame", "frame", "plan", "action", "result"])
        plan = events[3]
        self.assertEqual((folder / f"{plan['frame']}.jpg").read_bytes(), b"step")
        self.assertEqual(sorted(p.name for p in folder.glob("*.jpg")), ["2.jpg", "3.jpg"])
        self.assertEqual(summary(self.root, name)["status"], "done")

    def test_a_step_screen_is_saved_even_if_its_notice_was_missed(self):
        self.bus.publish("run", goal="open mail", run_id="abc")
        self.pump()
        step = self.bus.set_frame(b"step", keep=True)
        self.bus.set_frame(b"newer live")  # the recorder sees only the newest frame for both notices
        self.bus.publish("plan", step=1, reasoning="", actions=[], points=[], rejected=[], frame=step, trace=[], t_model=0)
        self.pump()
        [name] = recordings(self.root)
        self.assertEqual((self.root / name / f"{step}.jpg").read_bytes(), b"step")

    def test_unfinished_run_reads_as_running_and_old_runs_are_dropped(self):
        with patch("brain.display.recordings.KEEP_RUNS", 2):
            for run_id in ("a1", "b2", "c3"):
                self.request(run_id)
            self.bus.publish("run", goal="still going", run_id="d4")
            self.pump()
        names = recordings(self.root)
        self.assertEqual([n.rsplit("-", 1)[1] for n in names], ["d4", "c3"])
        self.assertEqual(summary(self.root, names[0])["status"], "running")

    def test_test_runs_never_push_out_the_users_runs(self):
        with patch("brain.display.recordings.KEEP_RUNS", 2):
            self.request("a1")
            for run_id in ("t1", "t2", "t3"):
                self.request(run_id, test=True)
        self.assertEqual([n.split("-", 2)[2] for n in recordings(self.root)], ["t3-test", "t2-test", "a1"])

    def test_a_disk_error_drops_the_recording_not_the_agent(self):
        self.recorder.root = self.root / "file"
        self.recorder.root.write_text("not a folder")
        with patch("sys.stderr"):
            self.request("abc")  # must not raise
        self.assertEqual(recordings(self.recorder.root), [])


class RecordingsServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bus = EventBus()
        recorder = Recorder(bus, self.root)
        bus.publish("run", goal="open mail", run_id="abc")
        step = bus.set_frame(b"\xff\xd8step", keep=True)
        bus.publish("plan", step=1, reasoning="", actions=[], points=[], rejected=[], frame=step, trace=[], t_model=0)
        while not recorder._client.empty():
            recorder.handle(recorder._client.get_nowait())
        recorder._close()
        [self.name] = recordings(self.root)
        self.step = step
        self.server = RecordingsServer(self.root, "127.0.0.1", 0)
        self.server.start()
        self.addCleanup(self.server.close)
        self.host, self.port = self.server.url.removeprefix("http://").split(":")

    def get(self, path, **headers):
        conn = http.client.HTTPConnection(self.host, int(self.port), timeout=2)
        self.addCleanup(conn.close)
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        return response.status, response.read()

    def test_lists_and_serves_recordings(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Exospace Recordings</title>", body)
        runs = json.loads(self.get("/runs.json")[1])
        self.assertEqual([(r["name"], r["goal"], r["status"]) for r in runs], [(self.name, "open mail", "running")])
        events = json.loads(self.get(f"/runs/{self.name}/log.json")[1])
        self.assertEqual([e["ev"] for e in events], ["run", "frame", "plan"])
        self.assertEqual(self.get(f"/runs/{self.name}/{self.step}.jpg"), (200, b"\xff\xd8step"))

    def test_only_recording_files_are_reachable(self):
        (self.root / "secret.txt").write_text("x")
        for path in (f"/runs/{self.name}/99.jpg", "/runs/../secret.txt", f"/runs/{self.name}/../../secret.txt",
                     "/runs/x/log.json", f"/runs/{self.name}/log.jsonl", "/events"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_refuses_unknown_hosts_and_all_interfaces(self):
        self.assertEqual(self.get("/runs.json", Host=f"evil.example:{self.port}")[0], 403)
        with self.assertRaises(ValueError):
            RecordingsServer(self.root, "0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
