"""Pipelined commit gate: no collisions between Jev and LLM proposals."""

import threading
import time
import unittest

from bench.arch.arbiter import Arbiter, Proposal, Verdict
from brain.types import Action

CLICK = (Action("click", x=0.5, y=0.5),)


class ArbiterTests(unittest.TestCase):
    def setUp(self):
        self.executed = []
        self.screen = 1
        self.arbiter = Arbiter(execute=self.executed.append, current_screen=lambda: self.screen)

    def proposal(self, segment=0, source="jev", generation=0, screen=1):
        return Proposal(segment, generation, source, CLICK, screen)

    def test_first_confident_proposal_wins_and_the_other_is_discarded(self):
        self.assertEqual(self.arbiter.submit(self.proposal(source="jev")), Verdict.COMMITTED)
        self.assertEqual(self.arbiter.submit(self.proposal(source="llm")), Verdict.LOST_RACE)
        self.assertEqual(len(self.executed), 1)

    def test_late_llm_answer_after_interruption_is_rejected(self):
        old = self.proposal(source="llm", generation=self.arbiter.generation)
        self.arbiter.interrupt()
        self.assertEqual(self.arbiter.submit(old), Verdict.STALE_GENERATION)
        self.assertEqual(self.executed, [])

    def test_later_segment_waits_for_earlier_one(self):
        results = {}
        worker = threading.Thread(target=lambda: results.update(v=self.arbiter.submit(self.proposal(segment=1))))
        worker.start()
        time.sleep(0.05)
        self.assertEqual(self.executed, [], "segment 1 must not run before segment 0")
        self.arbiter.submit(self.proposal(segment=0))
        self.arbiter.finish_segment(0)
        worker.join(1)
        self.assertEqual(results["v"], Verdict.COMMITTED)
        self.assertEqual(len(self.executed), 2)

    def test_interruption_cancels_queued_segment(self):
        results = {}
        worker = threading.Thread(target=lambda: results.update(v=self.arbiter.submit(self.proposal(segment=2))))
        worker.start()
        time.sleep(0.05)
        self.arbiter.interrupt()
        worker.join(1)
        self.assertEqual(results["v"], Verdict.CANCELLED)
        self.assertEqual(self.executed, [])

    def test_stale_screen_is_regrounded_not_executed(self):
        self.screen = 2
        self.assertEqual(self.arbiter.submit(self.proposal(screen=1)), Verdict.STALE_SCREEN)
        self.assertEqual(self.arbiter.submit(self.proposal(screen=2, source="llm")), Verdict.COMMITTED)
        self.assertEqual(len(self.executed), 1)


if __name__ == "__main__":
    unittest.main()
