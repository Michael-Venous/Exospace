"""Nested model/HTTP cancellation wrappers must not exhaust the worker pool."""
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from brain.cancel import Cancelled, CancelToken


class NestedCancellationTests(unittest.TestCase):
    def test_concurrent_nested_calls_finish_with_all_workers_occupied(self):
        tokens = [CancelToken(), CancelToken()]
        entered = threading.Barrier(2)
        results, failures = [], []

        def caller(token):
            try:
                def outer():
                    entered.wait(timeout=1)
                    return token.run(lambda: 42)
                results.append(token.run(outer))
            except Exception as exc:
                failures.append(exc)

        with ThreadPoolExecutor(max_workers=2) as pool, patch('brain.cancel._pool', pool):
            threads = [threading.Thread(target=caller, args=(token,)) for token in tokens]
            for thread in threads:
                thread.start()
            try:
                for thread in threads:
                    thread.join(1)
                self.assertEqual(results, [42, 42])
                self.assertEqual(failures, [])
            finally:
                for token in tokens:
                    token.cancel('test cleanup')
                for thread in threads:
                    thread.join(1)

    def test_cancellation_returns_before_nested_http_finishes(self):
        token, entered, finish = CancelToken(), threading.Event(), threading.Event()
        results = []

        def http():
            entered.set()
            finish.wait(2)
            return 'late response'

        def caller():
            try:
                results.append(token.run(lambda: token.run(http)))
            except Cancelled:
                results.append('cancelled')

        thread = threading.Thread(target=caller)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            token.cancel('barge-in')
            thread.join(.3)
            self.assertEqual(results, ['cancelled'])
        finally:
            finish.set()
            thread.join(1)


    def test_reset_cannot_revive_an_abandoned_worker(self):
        token = CancelToken()
        entered, finish, exited = threading.Event(), threading.Event(), threading.Event()
        committed = []
        def old_work():
            entered.set()
            try:
                finish.wait(2)
                token.check()
                committed.append('old request')
            finally:
                exited.set()
        def caller():
            try:
                token.run(old_work)
            except Cancelled:
                pass
        thread = threading.Thread(target=caller)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            token.cancel('barge-in')
            thread.join(.3)
            self.assertFalse(thread.is_alive())
            token.reset()
            self.assertEqual(token.run(lambda: 'new request'), 'new request')
            finish.set()
            self.assertTrue(exited.wait(1))
            self.assertEqual(committed, [])
        finally:
            finish.set()
            thread.join(1)
