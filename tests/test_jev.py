import gzip
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import httpx
import numpy as np

from brain.agent.jev import JevVision, JevError, probability, prepare_image
from brain.cancel import CancelToken


class JevTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name)/'timings.jsonl'
        self.token = CancelToken()
        self.jev = JevVision('private-test-key', self.token, self.log)
        self.jev.client.close()
        self.jev.client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.jev.close)
        self.image = cv2.imencode('.jpg',np.full((600,900,3),240,np.uint8))[1].tobytes()
        self.answers = {'present':{'noul':.99},'unique':{'noul':.99},
                        'location':{'confidence':.9,'box':{'left':.3,'right':.5,'top':.2,'bottom':.4}}}
        self.status = 200

    def respond(self, request):
        self.assertEqual(request.headers['content-encoding'], 'gzip')
        self.body = json.loads(gzip.decompress(request.content))
        return httpx.Response(self.status,json={'answers':self.answers})

    def test_grounding_and_private_logs(self):
        point = self.jev.locate(self.image,'private target')
        self.assertAlmostEqual(point[0],.4)
        self.assertAlmostEqual(point[1],.3)
        self.assertEqual(set(self.body['questions']),{'present','unique','location'})
        self.assertNotIn('private',self.log.read_text())
        self.assertNotIn('assets',self.log.read_text())

    def test_absent_and_ambiguous_targets_never_click(self):
        for name in ('present','unique'):
            with self.subTest(name=name):
                self.answers[name]['noul']=.1
                self.assertIsNone(self.jev.locate(self.image,'button'))
                self.answers[name]['noul']=.99

    def test_low_confidence_never_clicks(self):
        self.answers['location']['confidence']=.2
        self.assertIsNone(self.jev.locate(self.image,'button'))

    def test_malformed_coordinates_fail_closed(self):
        for value in (float('nan'),float('inf'),-1,2,True,'0.2'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): probability(value)
        self.answers['location']['box']['left']=.8
        with self.assertRaises(ValueError): self.jev.locate(self.image,'button')

    def test_padding_coordinates_are_unmapped(self):
        self.image=cv2.imencode('.jpg',np.full((100,500,3),240,np.uint8))[1].tobytes()
        self.answers['location']['box'].update(top=.1,bottom=.2)
        point=self.jev.locate(self.image,'button')
        self.assertAlmostEqual(point[1],.42)
        self.answers['location']['box'].update(top=.8,bottom=.9)
        self.assertIsNone(self.jev.locate(self.image,'button'))

    def test_trace_records_answers_gates_and_boxes_on_screen(self):
        self.jev.locate(self.image,'button')
        call=self.jev.trace.pop()
        self.assertEqual((call['label'],call['gates']['location']),('locate',.6))
        self.assertEqual(call['answers']['location']['box'],[.3,.2,.5,.4])
        self.assertEqual(call['answers']['present'],{'noul':.99})
        # A padded crop of the screen's bottom-right quarter: boxes land back on the whole screen.
        self.image=cv2.imencode('.jpg',np.full((100,500,3),240,np.uint8))[1].tobytes()
        self.answers['location']['box'].update(top=.1,bottom=.2)
        self.jev.evaluate(self.image,{'location':{}},region=(.5,.5,1,1))
        self.assertEqual(self.jev.trace.pop()['answers']['location']['box'],[.65,.64,.75,.78])
        self.answers['location']['box']['left']='bad'
        self.jev.evaluate(self.image,{'location':{}})
        self.assertIsNone(self.jev.trace.pop()['answers']['location']['box'])

    def test_api_error_and_incomplete_response(self):
        self.status=401
        with self.assertRaisesRegex(JevError,'401'): self.jev.locate(self.image,'button')
        self.status=200
        del self.answers['unique']
        with self.assertRaises(JevError): self.jev.locate(self.image,'button')

    def test_one_retry_on_timeout_or_overload(self):
        calls = []
        def flaky(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout('slow', request=request)
            return httpx.Response(200, json={'answers': self.answers})
        self.jev.client = httpx.Client(transport=httpx.MockTransport(flaky))
        self.token.sleep = Mock()
        self.assertIn('present', self.jev.evaluate(self.image, {'present': {}})[0])
        self.assertEqual(len(calls), 2)
        self.jev.client = httpx.Client(transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(503)))
        with self.assertRaisesRegex(JevError, '503'): self.jev.evaluate(self.image, {'present': {}})
        self.assertEqual(len(calls), 4)

    def test_a_slow_call_is_raced_by_a_second_copy(self):
        release, calls = threading.Event(), []
        def slow_first(request):
            calls.append(request)
            if len(calls) == 1:
                release.wait(5)
            return httpx.Response(200, json={'answers': self.answers})
        self.jev.client = httpx.Client(transport=httpx.MockTransport(slow_first))
        self.addCleanup(release.set)
        started = time.monotonic()
        with patch('brain.agent.jev.HEDGE_S', .05):
            self.assertIn('present', self.jev.evaluate(self.image, {'present': {}})[0])
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(len(calls), 2)

    def test_parallel_calls_prepare_their_screen_once(self):
        image = cv2.imencode('.jpg', np.full((600, 900, 3), 7, np.uint8))[1].tobytes()
        with patch('brain.agent.jev.prepare_image', wraps=prepare_image) as prepare, ThreadPoolExecutor(5) as pool:
            list(pool.map(lambda _: self.jev.evaluate(image, {'present': {}}), range(5)))
        self.assertEqual(prepare.call_count, 1)

    def test_cancelled_request_does_not_hit_api(self):
        self.token.cancel('test')
        from brain.cancel import Cancelled
        with self.assertRaises(Cancelled): self.jev.locate(self.image,'button')
        self.assertFalse(hasattr(self,'body'))

    def test_image_limits(self):
        raw=cv2.imencode('.jpg',np.ones((200,3000,3),np.uint8))[1].tobytes()
        encoded,size,padded=prepare_image(raw)
        im=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(im.shape[:2],(280,1568))
        self.assertEqual(size,(1568,105))

    def test_explicit_route_is_independent_of_box_confidence(self):
        self.answers={
            'route':{'choice':'execute','confidence':.4},
            'target_present':{'noul':.99},'target_unique':{'noul':.99},
            'target_clear':{'noul':.99},
            'target_box':{'confidence':.95,'box':{'left':.3,'right':.5,'top':.2,'bottom':.4}}}
        values,points=self.jev.assess(self.image,{}, {'target':'Continue'})
        self.assertEqual(values['route_execute'],1)
        self.assertIsNotNone(points['target'])
        self.answers['route']['choice']='ask_flash'
        values,points=self.jev.assess(self.image,{}, {'target':'Continue'})
        self.assertEqual(values['route_execute'],0)

    def test_unclear_routing_is_not_overridden_by_high_confidence_box(self):
        self.answers={
            'route':{'choice':'ask_flash','confidence':.9},
            'target_present':{'noul':.99},'target_unique':{'noul':.99},
            'target_clear':{'noul':.2},
            'target_box':{'confidence':.99,'box':{'left':.3,'right':.5,'top':.2,'bottom':.4}}}
        values,points=self.jev.assess(self.image,{}, {'target':'the thing'})
        self.assertEqual(values['route_execute'],0)

    def test_irreversible_proposal_is_scored_in_the_same_call(self):
        self.answers={'route':{'choice':'execute','confidence':.9},'side_effects':{'noul':.1},
                      'irreversible':{'noul':.93}}
        send=[{'type':'click','target':'Send button'}]
        values,_=self.jev.assess(self.image,{},{},proposal=send,irreversible=True)
        self.assertEqual(values['irreversible'],.93)
        self.assertIn('Send button',self.body['state']['actions'])
        self.jev.assess(self.image,{},{},proposal=send)
        self.assertNotIn('irreversible',self.body['questions'])


if __name__=='__main__': unittest.main()
