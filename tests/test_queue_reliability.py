"""Regression tests for effect ordering and independent outcome checks. No API calls."""
import tempfile
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from bench.arch.reliability_oracles import exact_files
from bench.arch.queue_agent import Executor, Shared, Brain
from brain.types import DeviceProfile, Rect
from bench.arch.openrouter_study import OpenRouterJSON


class ExactOutcomeTests(unittest.TestCase):
    def test_duplicate_text_and_files_do_not_pass(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            note = root/'note.txt'
            note.write_text('buy milk\n')
            self.assertTrue(exact_files(root, {'note.txt': 'buy milk'})[0])
            note.write_text('buy milk\nbuy milk\n')
            self.assertFalse(exact_files(root, {'note.txt': 'buy milk'})[0])
            note.write_text('buy milk\n')
            (root/'note (1).txt').write_text('buy milk\n')
            self.assertFalse(exact_files(root, {'note.txt': 'buy milk'})[0])
            (root/'note (1).txt').unlink()
            (root/'extra').mkdir()
            self.assertFalse(exact_files(root, {'note.txt': 'buy milk'})[0])

    def test_a_copy_must_preserve_the_original_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root/'original.txt').write_text('changed\n')
            (root/'new.txt').write_text('changed\n')
            self.assertFalse(exact_files(root, {'original.txt': b'original\n', 'new.txt': 'changed'})[0])


def executor():
    shared = Shared(goal='save this note')
    controller = Mock()
    ex = Executor(controller, Mock(), DeviceProfile('linux', Rect(0, 0, 1, 1), 'mouse', 0), shared, threading.Condition(), time.monotonic(), [])
    ex.strict_reports = True
    return ex


class CompletionOrderingTests(unittest.TestCase):
    def test_unverified_speech_is_not_delivered(self):
        ex = executor()
        ex._verify = Mock(return_value=('not_verified', 'save dialog still open'))
        ex.report = Mock()
        self.assertEqual(ex._do({'say': 'Saved.'})[0], 'not_verified')
        ex.report.assert_not_called()

    def test_verified_report_is_delivered_after_check(self):
        ex = executor()
        events = []
        ex._verify = lambda text: (events.append('check') or 'verified', '')
        ex.report = lambda text: events.append('report')
        self.assertEqual(ex._do({'say': 'Saved.'})[0], 'verified')
        self.assertEqual(events, ['check', 'report'])

    def test_ask_is_distinct_from_verification(self):
        ex = executor()
        ex._verify = Mock(side_effect=AssertionError('asking is not completion'))
        self.assertEqual(ex._do({'ask': 'Which file?'})[0], 'asked')

    def test_missing_destination_field_sends_no_input(self):
        ex = executor()
        ex._find = lambda text: None
        self.assertEqual(ex._do({'type': 'note.txt', 'into': 'file name', 'replace': True})[0], 'not_found')
        ex.controller.type_text.assert_not_called()
        ex.controller.hotkey.assert_not_called()

    def test_field_is_focused_before_selection_and_typing(self):
        ex = executor()
        ex._find = lambda text: (.5, .5)
        ex._input = lambda send: send()
        self.assertEqual(ex._do({'type': 'note.txt', 'into': 'file name', 'replace': True})[0], 'done')
        self.assertEqual([call[0] for call in ex.controller.mock_calls], ['execute', 'hotkey', 'type_text'])

    def test_replace_without_a_field_selects_the_old_text(self):
        ex = executor()
        ex._input = lambda send: send()
        self.assertEqual(ex._do({'type': 'hello world', 'replace': True})[0], 'done')
        self.assertEqual([call[0] for call in ex.controller.mock_calls], ['hotkey', 'type_text'])

    def test_reports_are_queued_with_the_actions_that_precede_them(self):
        ex = executor()
        brain = Brain.__new__(Brain)
        brain.executor, brain.shared, brain.cond = ex, ex.shared, ex.cond
        brain.cancel_call = threading.Event()
        brain.speaking = False
        brain.said, brain.conversation = [], []
        record = {'lines': []}
        brain._take({'key': ['CTRL', 'S']}, record, time.monotonic())
        brain._take({'say': 'Saved.'}, record, time.monotonic())
        self.assertEqual(list(ex.shared.queue), [{'key': ['CTRL', 'S']}, {'say': 'Saved.'}])
        self.assertEqual(brain.said, [])

    def test_cancel_turn_drops_words_streamed_for_a_box_command(self):
        ex = executor()
        brain = Brain.__new__(Brain)
        brain.executor, brain.shared, brain.cond = ex, ex.shared, ex.cond
        brain.cancel_call = threading.Event()
        brain.words, brain.speaking, brain.dirty = "i'll type my password", True, True
        ex.shared.queue.append({'click': 'the password field'})
        brain.cancel_turn()
        self.assertEqual((brain.words, brain.speaking, brain.dirty), ('', False, False))
        self.assertEqual(list(ex.shared.queue), [])
        self.assertTrue(brain.cancel_call.is_set() and ex.shared.finished)

    def test_no_terminal_report_while_the_user_is_still_speaking(self):
        ex = executor()
        brain = Brain.__new__(Brain)
        brain.executor, brain.shared, brain.cond = ex, ex.shared, ex.cond
        brain.cancel_call = threading.Event()
        brain.speaking = True
        record = {'lines': []}
        brain._take({'say': 'Finished.'}, record, time.monotonic())
        self.assertFalse(ex.shared.queue)

    def test_new_speech_during_verification_cannot_emit_old_success(self):
        ex = executor()
        def verify(text):
            ex.shared.request += 1
            return 'verified', ''
        ex._verify = verify
        ex.report = Mock()
        self.assertEqual(ex._do({'say': 'Saved.'})[0], 'superseded')
        ex.report.assert_not_called()

    def test_new_request_keeps_its_queue_when_old_check_returns(self):
        ex = executor()
        ex.shared.queue.append({'done': 'old request'})
        executed = []
        def do(action):
            executed.append(action)
            if 'done' in action:
                ex.shared.request += 1
                ex.shared.queue.append({'key': ['ENTER']})
                return 'verified', ''
            ex.closed = True
            return 'done', ''
        ex._do = do
        ex.run()
        self.assertEqual(executed, [{'done': 'old request'}, {'key': ['ENTER']}])
        self.assertFalse(ex.shared.finished)
        self.assertEqual(ex.shared.log[0].result, 'superseded')

    def test_stale_grounding_cannot_send_input(self):
        ex = executor()
        ex.active_request = 0
        ex.shared.request = 1
        send = Mock()
        with self.assertRaises(RuntimeError):
            ex._input(send)
        send.assert_not_called()

    def test_terminal_result_discards_later_actions(self):
        ex = executor()
        ex.shared.queue.extend([{'done': 'finished'}, {'type': 'unwanted'}])
        def do(action):
            ex.closed = True
            return 'verified', ''
        ex._do = do
        ex.run()
        self.assertFalse(ex.shared.queue)

    def test_missing_evidence_ends_with_uncertainty_without_repeating_input(self):
        ex = executor()
        ex._jpeg = lambda: b'frame'
        ex.checks = SimpleNamespace(completion=lambda *args, **kwargs: 0.0,
                                   last_status='unknown', last_reason='Destination is not visible.')
        ex.report = Mock()
        self.assertEqual(ex._verify('Saved.')[0], 'uncertain')
        ex.report.assert_called_once_with("I can't fully verify the result from the screen. Destination is not visible.")
        self.assertFalse(ex.controller.mock_calls)

    def test_alternate_checker_deadline_is_not_retried_by_the_executor(self):
        ex = executor()
        ex.checks = object()
        ask = Mock(side_effect=TimeoutError('total deadline'))
        with self.assertRaises(TimeoutError):
            ex._vision_call(ask)
        ask.assert_called_once()

    def test_a_long_executor_step_still_gets_a_followup_model_call(self):
        ex = executor()
        brain = Brain.__new__(Brain)
        brain.executor, brain.shared = ex, ex.shared
        brain.dirty = False
        brain.active_until = time.monotonic() - 20
        ex.shared.activity_at = time.monotonic()
        self.assertTrue(brain._due())
        ex.shared.finished = True
        self.assertFalse(brain._due())

    def test_ambiguous_gesture_is_rejected_instead_of_silently_becoming_single_click(self):
        ex = executor()
        self.assertEqual(ex._do({'click': 'a file', 'double_click': True})[0], 'error')
        self.assertFalse(ex.controller.mock_calls)

    def test_auxiliary_field_order_does_not_change_the_action(self):
        ex = executor()
        ex._find = lambda text: (.5, .5)
        ex._input = lambda send: send()
        self.assertEqual(ex._do({'replace': True, 'into': 'name', 'type': 'note.txt'})[0], 'done')
        ex.controller.type_text.assert_called_once_with('note.txt')


class CheckerTransportTests(unittest.TestCase):
    def test_keepalive_lines_do_not_defeat_total_deadline(self):
        client = OpenRouterJSON('model', '', '', total_s=2)
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=': keepalive\n\n: keepalive\n\n')))
        try:
            with patch('bench.arch.openrouter_study.time.monotonic', side_effect=range(20)):
                with self.assertRaises(TimeoutError):
                    client.call(None, 'check', 'system', {})
        finally:
            client.client.close()

    def test_output_limit_reaches_provider(self):
        sent = {}
        def respond(request):
            sent.update(json.loads(request.content))
            delta = {'choices': [{'delta': {'content': '{"status":"unknown","reason":"missing"}'}}]}
            return httpx.Response(200, text='data: '+json.dumps(delta)+'\n\ndata: [DONE]\n\n')
        client = OpenRouterJSON('model', '', '', max_tokens=240)
        client.client.close()
        client.client = httpx.Client(transport=httpx.MockTransport(respond))
        try:
            result, _ = client.call(None, 'check', 'system', {})
            self.assertEqual(result['status'], 'unknown')
            self.assertEqual(sent['max_tokens'], 240)
        finally:
            client.client.close()


if __name__ == '__main__':
    unittest.main()
