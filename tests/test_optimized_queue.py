"""Control invariants from the live stress run, without APIs or desktop access."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np

from bench.arch.confirm import Verdict, candidate, is_yes
from bench.arch.optimized_queue import PROMPT, Planned, Superseded, create
from bench.arch.openrouter_study import ModelHTTPError
from bench.arch.pointer import Pointer
from bench.arch.queue_agent import RUNAWAY, Brain
from bench.arch.queue_control import changed, thumbnail
from brain.cancel import CancelToken


class Target:
    def __init__(self):
        self.sent=[]
        self.on_send=None
    def grab(self):
        return np.zeros((200,320,3),np.uint8)
    def send(self, command):
        self.sent.append(command)
        if self.on_send:self.on_send(command)
    def is_mounted(self):return True
    def signal_present(self):return True


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.target=Target()
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test'}), \
             patch('bench.arch.optimized_queue.OptimizedExecutor.start'), \
             patch('bench.arch.optimized_queue.OptimizedBrain.start'):
            self.agent=create(self.target,'deepseek/deepseek-v4.1-flash-1017','','test',Path(self.tmp.name),'')
        self.c,self.brain,self.ex=self.agent.control,self.agent.brain,self.agent.executor
        # No network: an input the prefilter picks (ENTER) is judged reversible unless a test says otherwise.
        self.ex.gate.judge=Mock(return_value=Verdict('reversible',0.05,'Search'))
    def tearDown(self):
        self.brain.client.close()
        self.ex.checks.close()
        self.ex.gate.close()
        self.agent.jev.close()
        self.c.events_log.close()
        self.tmp.cleanup()
    def plan(self, action):
        self.brain.hear('test request',False)
        self.agent.cancel.reset()
        self.c.observe(self.target.grab())
        p=Planned(action,self.c.epoch,self.agent.shared.request,self.c.view)
        self.ex.active_plan=p
        return p
    def test_partial_transcripts_do_not_plan_or_type(self):
        self.brain.hear('Type the wrong',True)
        epoch=self.c.epoch
        self.brain.hear('Actually type the right value',True)
        self.assertFalse(self.brain._due())
        self.assertGreater(self.c.epoch,epoch)
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
        self.brain.hear('Actually type the right value',False)
        self.assertTrue(self.brain._due())
    def test_press_stops_old_authority_before_any_new_transcript_arrives(self):
        p=self.plan({'type':'old'})
        self.brain.end_turn()
        self.assertFalse(self.c.valid(p.epoch))
        with self.assertRaises(Superseded):
            self.agent.controller.hid.send({'op':'type','text':'old'})
    def test_stale_stream_cannot_restore_queue_after_interruption(self):
        p=self.plan({'key':['ENTER']})
        self.brain.end_turn()
        record={'epoch':p.epoch,'request':p.request,'view':p.view,'lines':[]}
        self.brain._take(p.action,record,time.monotonic())
        self.assertFalse(self.agent.shared.queue)
    def test_long_typing_is_cancelled_at_the_packet_boundary(self):
        p=self.plan({'type':'a'*256})
        def stop(command):
            if command['op']=='type':self.brain.end_turn()
        self.target.on_send=stop
        self.ex._do(p.action)
        typed=[x for x in self.target.sent if x['op']=='type']
        self.assertEqual(len(typed),1)
        self.assertLessEqual(len(typed[0]['text']),32)
    def test_stopped_plan_cannot_click_after_slow_grounding_returns(self):
        p=self.plan({'click':'a control'})
        def found(text):
            self.brain.end_turn()
            return .5,.5
        self.ex._find=found
        self.ex._do(p.action)
        self.assertFalse([x for x in self.target.sent if x['op']=='click'])
    def test_failure_report_does_not_need_goal_success(self):
        p=self.plan({'say':'That collection is not available.','status':'blocked'})
        self.ex.checks.claim=Mock(side_effect=AssertionError('should not check task success'))
        self.assertEqual(self.ex._do(p.action)[0],'blocked')
        self.assertEqual(self.brain.said,['That collection is not available.'])
        self.assertTrue(self.agent.shared.finished)
        self.assertFalse(self.c.valid(p.epoch))
    def test_new_observation_required_after_input_before_claim(self):
        p=self.plan({'key':['ENTER']})
        self.ex._do(p.action)
        self.ex.checks.claim=Mock(side_effect=AssertionError('no fresh view'))
        self.assertEqual(self.ex._do({'done':'the result is visible'})[0],'observed')
        self.assertFalse(self.agent.shared.finished)
    def test_contradicted_claim_ends_without_redoing_effect(self):
        p=self.plan({'say':'Saved.','status':'complete'})
        self.ex.checks.claim=Mock(return_value=(False,'unknown','The dialog is still open.'))
        self.assertEqual(self.ex._do(p.action)[0],'uncertain')
        self.assertTrue(self.agent.shared.finished)
        record={'epoch':p.epoch,'request':p.request,'view':p.view,'lines':[]}
        self.brain._take({'key':['ENTER']},record,time.monotonic())
        self.assertFalse(self.agent.shared.queue)
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_close_shortcut_cannot_target_whatever_window_is_in_front(self):
        p=self.plan({'key':['ALT','F4']})
        self.assertEqual(self.ex._do(p.action)[0],'error')
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_close_shortcut_focuses_its_named_window_before_dispatch(self):
        p=self.plan({'key':['ALT','F4'],'in':'draft.txt editor'})
        self.ex._find=Mock(return_value=(.5,.5))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        sent=[x['op'] for x in self.target.sent if x['op']!='release_all']
        self.assertEqual(sent,['click','key'])
    def test_conflicting_text_parameter_does_not_type_a_placeholder(self):
        p=self.plan({'type':'text','text':'actual value'})
        self.assertEqual(self.ex._do(p.action)[0],'error')
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_check_timeout_is_uncertainty_not_another_ui_attempt(self):
        p=self.plan({'done':'Saved.'})
        self.ex.checks.claim=Mock(side_effect=httpx.ReadTimeout('slow service'))
        self.assertEqual(self.ex._do(p.action)[0],'uncertain')
        self.assertTrue(self.agent.shared.finished)
    def test_a_dropped_check_connection_is_asked_again(self):
        p=self.plan({'done':'The terminal is minimized.'})
        self.ex.checks.claim=Mock(side_effect=[httpx.ReadError('bad record mac'),(True,'complete','')])
        self.assertEqual(self.ex._do(p.action)[0],'verified')
        self.assertEqual(self.ex.checks.claim.call_count,2)
    def test_a_done_task_whose_check_never_answers_is_not_called_a_failure(self):
        p=self.plan({'done':'The terminal is minimized.'})
        self.ex.checks.claim=Mock(side_effect=httpx.ReadError('bad record mac'))
        self.assertEqual(self.ex._do(p.action)[0],'uncertain')
        self.assertEqual(self.brain.said[-1],"I think that's done (The terminal is minimized), but I couldn't double-check it on the screen.")
    def test_a_contradicted_report_gets_one_correction_then_stops(self):
        p=self.plan({'say':'It needs 2 1/4 cups of flour.','status':'answer'})
        # The shared fields say something else: only this check's own verdict counts (a check abandoned
        # after an interrupt may still write them).
        self.ex.checks.last_status,self.ex.checks.last_reason='complete','stale verdict of an abandoned check'
        self.ex.checks.claim=Mock(return_value=(False,'incomplete','The flour shown is 2 1/3 cups.'))
        result,detail=self.ex._do(p.action)
        self.assertEqual(result,'not_verified')
        self.assertIn('2 1/3',detail)
        self.assertFalse(self.agent.shared.finished)
        self.assertEqual(self.brain.said,[])
        p=self.plan({'say':'It needs 2 1/4 cups of flour.','status':'answer'})
        self.c.corrected=True  # same request: plan() starts a new one, so restore its state
        self.assertEqual(self.ex._do(p.action)[0],'uncertain')
        self.assertTrue(self.agent.shared.finished)
    def test_triple_click_selects_a_line(self):
        self.target.features=frozenset({'triple_click'})  # the laptop rig; the Pico's fallback is test_pi_target's
        p=self.plan({'triple_click':'the flour ingredient line'})
        self.ex._find=Mock(return_value=(.5,.25))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        clicks=[x for x in self.target.sent if x['op']=='click']
        self.assertEqual([(c['count'],c['button']) for c in clicks],[(3,'left')])
    def test_a_click_spelled_with_a_count_becomes_that_click(self):
        p=self.plan({'click':'x'})
        record={'epoch':p.epoch,'request':p.request,'view':p.view,'lines':[]}
        for action,want in (({'click':'the flour line','click_count':3},{'triple_click':'the flour line'}),
                            ({'click':'the flour line','triple_click':True},{'triple_click':'the flour line'}),
                            ({'click':'Preheat','double_click':True},{'double_click':'Preheat'})):
            self.agent.shared.queue.clear()
            self.brain._take(action,record,time.monotonic())
            self.assertEqual([q.action for q in self.agent.shared.queue],[want])
    def test_scrolling_through_a_page_is_not_going_in_circles(self):
        scroll={'lines':[{'scroll':5,'at':'the recipe'}]}
        self.assertFalse(self.brain._counts_as_repeat(scroll))
        self.assertFalse(self.brain._counts_as_repeat({'lines':[{'scroll':5},{'until':'the ingredients'}]}))
        self.assertTrue(self.brain._counts_as_repeat({'lines':[{'click':'Next'}]}))
    def test_a_new_request_may_correct_again(self):
        self.c.corrected=True
        self.brain.end_turn()
        self.assertFalse(self.c.corrected)
    def test_an_input_landing_for_the_previous_request_is_not_this_ones_step(self):
        p=self.plan({'key':['ENTER']})
        def slow(action):
            self.brain.end_turn()  # the user spoke while the key went out
            return 'done',''
        self.ex._do=slow
        with self.agent.cond:self.agent.shared.queue.append(p)
        thread=threading.Thread(target=self.ex.run,daemon=True);thread.start()
        deadline=time.monotonic()+1
        while not self.agent.shared.log and time.monotonic()<deadline:time.sleep(.005)
        self.ex.closed=True;thread.join(1)
        self.assertEqual(self.agent.shared.log[-1].result,'done_earlier')
    def test_a_wait_pauses_at_most_three_seconds(self):
        p=self.plan({'wait':60})
        with patch.object(self.agent.cancel,'sleep') as sleep:
            self.assertEqual(self.ex._do(p.action)[0],'done')
            self.ex._do({'wait':'a moment'})
        self.assertEqual([c.args[0] for c in sleep.call_args_list],[3.0,1.0])
    def test_the_launcher_wait_belongs_to_slow_launchers_only(self):
        p=self.plan({'key':['GUI']})
        grabs=[]
        self.target.grab=lambda:(grabs.append(1),np.zeros((200,320,3),np.uint8))[1]
        self.ex._do(p.action)
        self.assertEqual(grabs,[])  # the laptop's KDE launcher: typing goes right after the key
        self.target.launcher_settles=True
        screens=iter([np.zeros((200,320,3),np.uint8),np.full((200,320,3),200,np.uint8)])
        self.target.grab=lambda:next(screens,np.full((200,320,3),200,np.uint8))
        with patch.object(self.agent.cancel,'sleep') as sleep:
            self.assertEqual(self.ex._do({'key':['super']})[0],'done')
        self.assertEqual(sleep.call_args_list[-1].args[0],.15)  # the menu drew: a beat for its search box
    def test_a_broken_line_drops_the_rest_of_its_reply_and_tells_the_planner(self):
        p=self.plan({'key':['ENTER']})
        record={'epoch':p.epoch,'request':p.request,'view':p.view,'lines':[]}
        self.brain._apply('{"key":["CTRL","L"]}',record,time.monotonic())  # before the broken line: runs
        self.brain._apply('{"click":"the \'Khentii Mountains" link in the Born row"}',record,time.monotonic())
        self.brain._apply('{"until":"the Khentii Mountains article opens"}',record,time.monotonic())
        self.assertEqual([q.action for q in self.agent.shared.queue],[{'key':['CTRL','L']}])
        self.assertEqual(self.agent.shared.log[-1].result,'error')
        self.assertIn('valid JSON',self.agent.shared.log[-1].detail)
    def test_an_input_carrying_a_status_is_refused_before_anything_is_sent(self):
        for action in ({'type':'born','status':'answer'},{'click':'Born','status':'answer'}):
            p=self.plan(action)
            self.ex._find=Mock(return_value=(.5,.5))
            result,detail=self.ex._do(p.action)
            self.assertEqual(result,'error')
            self.assertIn('status only goes with say',detail)
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_conversation_reply_is_spoken_without_a_screen_check(self):
        p=self.plan({'say':'Hello! How can I help?','status':'reply'})
        self.ex.checks.claim=Mock(side_effect=AssertionError('a greeting needs no evidence'))
        self.assertEqual(self.ex._do(p.action)[0],'replied')
        self.assertEqual(self.brain.said,['Hello! How can I help?'])
        self.assertEqual(self.c.requests[p.request]['status'],'replied')
    def test_answer_whose_check_never_returns_is_said_flagged_unchecked(self):
        p=self.plan({'say':'Your battery is at 63%.','status':'answer'})
        self.ex.checks.claim=Mock(side_effect=httpx.ReadTimeout('slow check'))
        self.assertEqual(self.ex._do(p.action)[0],'uncertain')
        self.assertEqual(self.brain.said,["Your battery is at 63%. I couldn't double-check that on the screen."])
    def test_claim_check_has_its_own_limits_and_at_most_four_images(self):
        self.assertEqual((self.ex.checks.client.client.timeout.read,self.ex.checks.client.total_s),(4.0,8.0))
        for i in range(6):self.ex.checks.observe(bytes([i]),1)
        sent=[]
        self.ex.checks.client.call=lambda images,*a:(sent.append(images) or ({'status':'complete','observed':'x'},{}))
        self.assertEqual(self.ex.checks.claim(b'final','goal','claim',True,[]),(True,'complete',''))
        self.assertLessEqual(len(sent[0]),4)
    def test_wait_does_not_call_another_model(self):
        p=self.plan({'until':'new page'})
        self.ex.checks.evaluate=Mock(side_effect=AssertionError('extra model call'))
        with patch('bench.arch.optimized_queue.UNTIL_STILL_S',.05):
            self.assertEqual(self.ex._do(p.action)[0],'no_change')
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_wait_returns_once_the_screen_has_changed_and_settled(self):
        p=self.plan({'until':'the window'})
        screens=iter([np.zeros((200,320,3),np.uint8)])
        drawn=np.full((200,320,3),200,np.uint8)
        self.target.grab=lambda:next(screens,drawn)  # unchanged, then drawn and held
        result,detail=self.ex._do(p.action)
        self.assertEqual(result,'observed')
        self.assertIn('settled',detail)
    def test_a_wait_with_no_change_at_all_hands_back_early(self):
        p=self.plan({'until':'the mute icon'})
        with patch('bench.arch.optimized_queue.UNTIL_STILL_S',.1),patch('bench.arch.optimized_queue.UNTIL_S',5):
            started=time.monotonic()
            result,detail=self.ex._do(p.action)
        self.assertEqual(result,'no_change')
        self.assertLess(time.monotonic()-started,1)
        self.assertIn('did not respond',detail)
    def test_hover_moves_without_clicking(self):
        p=self.plan({'hover':'the File menu'})
        self.ex._find=Mock(return_value=(.25,.75))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        self.assertEqual([x['op'] for x in self.target.sent if x['op']!='release_all'],['move'])
    def test_hover_away_parks_the_pointer_at_the_edge(self):
        p=self.plan({'hover':'away'})
        self.ex._find=Mock(side_effect=AssertionError('parking needs no lookup'))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        moves=[x for x in self.target.sent if x['op']=='move']
        self.assertEqual(len(moves),1)
        self.assertGreater(moves[0]['x'],32000)
    def test_ordinary_keys_go_to_the_window_in_front(self):
        p=self.plan({'key':['M'],'in':'Instagram'})
        self.ex._find_window=Mock(side_effect=AssertionError('no lookup for an ordinary key'))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        self.assertEqual([x['op'] for x in self.target.sent if x['op']!='release_all'],['key'])
    def test_out_of_attempts_gets_one_report_only_call(self):
        self.brain.hear('set 60 hertz',False)
        self.c.calls,self.c.changed_at=5,time.monotonic()-60
        notes=[]
        with patch.object(Brain,'_call',lambda brain:notes.append(brain.note)):
            self.brain._call()
        self.assertFalse(self.agent.shared.finished)  # not the generic stop yet
        self.assertIn('out of attempts',notes[0])
        record={'epoch':self.brain.call_epoch,'request':self.agent.shared.request,'view':self.c.view,'lines':[]}
        self.brain._take({'click':'120.00 Hz'},record,time.monotonic())
        self.brain._take({'say':'Only 120 Hz is offered.','status':'answer'},record,time.monotonic())
        self.assertEqual([p.action for p in self.agent.shared.queue],[{'say':'Only 120 Hz is offered.','status':'blocked'}])
        with patch.object(Brain,'_call',lambda brain:None):
            self.brain._call()  # the report call brought nothing that ended the request
        self.assertEqual(self.c.requests[self.agent.shared.request]['status'],'blocked')
        self.assertIn('enough progress',self.brain.said[-1])
    def test_a_new_request_has_its_attempts_back(self):
        self.brain.final_report=self.brain.reported=True
        self.c.unanswered=2
        self.brain.end_turn()
        self.assertFalse(self.brain.final_report or self.brain.reported or self.c.unanswered)
    def test_a_control_that_ignores_two_inputs_leads_to_a_report(self):
        p=self.plan({'until':'the list'})
        for result in ('no_change','observed','no_change','no_change'):
            self.ex._do=Mock(return_value=(result,''))
            with self.agent.cond:
                self.agent.shared.queue.append(Planned({'until':'x'},self.c.epoch,p.request,self.c.view))
            self.ex.closed=False
            thread=threading.Thread(target=self.ex.run,daemon=True);thread.start()
            deadline=time.monotonic()+1
            while self.agent.shared.queue and time.monotonic()<deadline:time.sleep(.005)
            time.sleep(.02);self.ex.closed=True;thread.join(1)
            self.assertEqual(self.brain.final_report,result=='no_change' and self.c.unanswered>=2)
    def test_wait_stops_when_the_user_presses_talk(self):
        p=self.plan({'until':'the window'})
        def grab():
            self.brain.end_turn()
            return np.zeros((200,320,3),np.uint8)
        self.target.grab=grab
        started=time.monotonic()
        self.assertEqual(self.ex._do(p.action)[0],'superseded')
        self.assertLess(time.monotonic()-started,.5)
    def test_terminal_revokes_inflight_input(self):
        p=self.plan({'key':['ENTER']})
        with self.agent.cond:self.c.finish('blocked','Stopped.')
        with self.assertRaises(Superseded):self.ex._input(lambda:None)
    def test_service_refusal_stops_instead_of_clicking_again(self):
        p=self.plan({'click':'Save'})
        self.ex._find=Mock(side_effect=ModelHTTPError(402,b'output allowance'))
        self.assertEqual(self.ex._do(p.action)[0],'blocked')
        self.assertTrue(self.agent.shared.finished)
        self.assertIn('HTTP 402',self.brain.said[0])
        self.assertFalse([x for x in self.target.sent if x['op']=='click'])
    def test_telemetry_measures_hid_dispatch_not_a_check_record(self):
        p=self.plan({'key':['ENTER']})
        self.assertNotIn('first_input_at',self.c.requests[p.request])
        self.ex._do(p.action)
        self.assertIn('first_input_at',self.c.requests[p.request])
        self.assertEqual([e['kind'] for e in self.c.events].count('hid'),1)
    def test_pending_cancel_finishes_before_next_planning_call_resets_token(self):
        self.plan({'type':'old'})
        self.agent.shared.state='running'
        self.brain.end_turn()
        self.brain.hear('new request',False)
        self.assertFalse(self.brain._due())
        self.assertTrue(self.agent.cancel.cancelled)
    def test_press_sets_cancel_before_waiting_for_a_packet_lock(self):
        self.plan({'type':'old'})
        pressed=threading.Event()
        def press():
            pressed.set()
            self.brain.end_turn()
        with self.agent.cond:
            thread=threading.Thread(target=press)
            thread.start()
            self.assertTrue(pressed.wait(.2))
            deadline=time.monotonic()+.2
            while not self.agent.cancel.cancelled and time.monotonic()<deadline:time.sleep(.001)
            self.assertTrue(self.agent.cancel.cancelled)
        thread.join(.5)
        self.assertFalse(thread.is_alive())
    def test_reasoning_heartbeats_do_not_defeat_planner_deadline(self):
        class Heartbeats(httpx.SyncByteStream):
            def __iter__(self):
                for _ in range(5):
                    time.sleep(.03)
                    yield b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n'
        self.brain.client.close()
        self.brain.client=httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200,stream=Heartbeats())))
        self.brain.cancel_call.clear()
        started=time.monotonic()
        with self.assertRaises(TimeoutError):list(Brain._stream(self.brain,{}, {'deadline_s':.02}))
        self.assertLess(time.monotonic()-started,.2)
    def test_permanent_planner_error_does_not_hedge_or_loop(self):
        self.brain.hear('open an app',False)
        sent=[]
        self.brain.client.close()
        def refuse(request):
            sent.append(json.loads(request.content))
            return httpx.Response(402,text='output allowance')
        self.brain.client=httpx.Client(transport=httpx.MockTransport(refuse))
        self.brain.cancel_call.clear()  # Brain.run normally clears this before starting the call.
        self.brain._call()
        self.assertEqual(len(sent),1)
        self.assertEqual(sent[0]['max_tokens'],2048)
        self.assertTrue(self.agent.shared.finished)
        self.assertEqual(self.c.requests[self.agent.shared.request]['status'],'blocked')
    def test_window_focus_failure_is_logged_not_fatal_to_the_executor(self):
        p=self.plan({'key':['ALT','F4'],'in':'draft.txt editor'})
        self.ex._find_window=Mock(side_effect=TimeoutError('no pointing reply in 3.5 s'))
        self.assertEqual(self.ex._do(p.action)[0],'error')
        self.assertFalse([x for x in self.target.sent if x['op']!='release_all'])
    def test_executor_thread_survives_a_failed_action(self):
        p=self.plan({'key':['ALT','F4'],'in':'draft.txt editor'})
        self.ex._find_window=Mock(side_effect=TimeoutError('no pointing reply in 3.5 s'))
        thread=threading.Thread(target=self.ex.run,daemon=True)
        thread.start()
        with self.agent.cond:
            self.agent.shared.queue.append(p)
            self.agent.cond.notify_all()
        deadline=time.monotonic()+1
        while not self.agent.shared.log and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.agent.shared.log[-1].result,'error')
        self.assertEqual(self.agent.shared.state,'idle')
        self.assertTrue(thread.is_alive())
        self.ex.closed=True
        thread.join(1)
    def test_runaway_reply_stops_and_drops_its_queued_repeats(self):
        p=self.plan({'key':['CTRL','Z']})
        record={'epoch':p.epoch,'request':p.request,'view':p.view,'lines':[]}
        for _ in range(RUNAWAY+3):
            self.brain._take({'key':['CTRL','Z']},record,time.monotonic())
        self.assertTrue(record.get('runaway'))
        self.assertFalse(self.agent.shared.queue)
        self.assertFalse(self.c.valid(p.epoch))
    def test_small_screen_changes_count_as_progress_and_a_caret_does_not(self):
        import cv2
        base=np.full((1800,2880,3),245,np.uint8)
        caret=base.copy();cv2.rectangle(caret,(700,375),(701,405),(0,0,0),-1)
        word=base.copy();cv2.putText(word,'hello',(700,400),cv2.FONT_HERSHEY_SIMPLEX,.9,(30,30,30),2)
        self.assertFalse(changed(thumbnail(base),thumbnail(caret)))
        self.assertTrue(changed(thumbnail(base),thumbnail(word)))
    def test_new_request_restarts_the_progress_clock(self):
        self.c.changed_at=time.monotonic()-60
        self.brain.end_turn()
        self.assertLess(time.monotonic()-self.c.changed_at,1)
    def _stalling_client(self,sent):
        def stall(request):
            sent.append(1)
            raise httpx.ReadTimeout('stalled')
        self.brain.client.close()
        return httpx.Client(transport=httpx.MockTransport(stall))
    def test_one_planner_stall_is_retried_and_a_second_stops(self):
        self.brain.hear('open an app',False)
        sent=[]
        self.brain._client=lambda:self._stalling_client(sent)
        self.brain.client=self.brain._client()
        self.brain.hosts=['a']
        self.brain.cancel_call.clear()
        self.brain._call()
        self.assertFalse(self.agent.shared.finished)
        self.assertTrue(self.brain.dirty)
        self.brain.dirty=False
        self.brain.cancel_call.clear()
        self.brain._call()
        self.assertTrue(self.agent.shared.finished)
        self.assertEqual(self.c.requests[self.agent.shared.request]['status'],'blocked')
        self.assertIn('twice in a row',self.brain.said[-1])
    def test_rate_limited_click_is_reported_to_the_model_not_a_stop(self):
        p=self.plan({'click':'Save'})
        self.ex._find=Mock(side_effect=ModelHTTPError(429,b'slow down'))
        self.assertEqual(self.ex._do(p.action)[0],'error')
        self.assertFalse(self.agent.shared.finished)
    def test_reply_ended_by_its_own_until_is_remembered(self):
        self.brain.hear('open the page',False)
        self.brain.client.close()
        lines=b'data: {"choices":[{"delta":{"content":"{\\"key\\":[\\"ENTER\\"]}\\n{\\"until\\":\\"the page\\"}\\n"}}]}\n\n'
        class Slow(httpx.SyncByteStream):
            def __iter__(inner):
                yield lines
                deadline=time.monotonic()+1
                while len(self.agent.shared.queue)<2 and time.monotonic()<deadline:time.sleep(.005)
                with self.agent.cond:  # the executor ran the until and ended the batch
                    self.c.invalidate(batch_ended=True)
                time.sleep(.2)
                yield b'data: [DONE]\n\n'
        self.brain.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=Slow())))
        self.brain.cancel_call.clear()
        self.brain._call()
        self.assertFalse(self.brain.calls[-1]['cancelled'])
        self.assertEqual(len(self.brain.history),1)
    def test_events_stream_to_a_log_and_save_stays_small(self):
        p=self.plan({'key':['ENTER']})
        for _ in range(3):self.ex._do(p.action)
        self.c.save(Path(self.tmp.name)/'control.json')
        saved=json.loads((Path(self.tmp.name)/'control.json').read_text())
        self.assertIn(p.request,map(int,saved['requests']))
        logged=[json.loads(x) for x in (Path(self.tmp.name)/'control-events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind']=='hid' for e in logged),3)
    def test_planner_screenshot_fills_the_hosts_pixel_budget_once(self):
        import cv2
        from bench.arch.optimized_queue import planner_jpeg
        image=cv2.imdecode(np.frombuffer(planner_jpeg(np.zeros((1800,2880,3),np.uint8)),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2],(1000,1600))
        small=cv2.imdecode(np.frombuffer(planner_jpeg(np.zeros((800,1280,3),np.uint8)),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(small.shape[:2],(800,1280))  # never enlarged
    def test_budgets_are_pixels_whatever_the_aspect_ratio(self):
        import cv2
        from bench.arch.frames import fit_jpeg
        for h,w in ((1440,3440),(1440,5120),(1920,1080),(2160,3840)):
            image=cv2.imdecode(np.frombuffer(fit_jpeg(np.zeros((h,w,3),np.uint8),2_500_000),np.uint8),cv2.IMREAD_COLOR)
            self.assertAlmostEqual(image.shape[0]*image.shape[1]/min(2_500_000,h*w),1,delta=.01)  # never enlarged
            self.assertAlmostEqual(image.shape[1]/image.shape[0],w/h,delta=.01)
    def test_optimized_pointer_gets_its_pixel_budget(self):
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test'}), \
             patch('bench.arch.optimized_queue.OptimizedExecutor.start'), \
             patch('bench.arch.optimized_queue.OptimizedBrain.start'):
            agent=create(Target(),'deepseek/deepseek-v4.1-flash-1017','','test',Path(self.tmp.name),'qwen')
        try:
            self.assertEqual(agent.executor.pointer.budget,2_500_000)
        finally:
            agent.executor.pointer.close();agent.brain.client.close();agent.executor.checks.close()
            agent.jev.close();agent.control.events_log.close()
    def test_click_report_uses_the_planner_screenshot_pixels(self):
        from brain.types import DeviceProfile, Rect
        self.ex.profile=DeviceProfile('linux',Rect(0,0,2880,1800),'mouse',0)
        self.assertEqual(self.ex._where((.5,.5)),'at (800, 500)')


class PointingServiceTests(unittest.TestCase):
    def test_permanent_pointing_error_is_not_retried_or_hedged(self):
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test'}):
            pointer=Pointer(CancelToken(),Mock())
        ask=Mock(side_effect=ModelHTTPError(402,b'output allowance'))
        try:
            with self.assertRaises(ModelHTTPError):pointer._hedged(ask)
            self.assertEqual(ask.call_count,1)
            self.assertEqual(pointer.client.max_tokens,128)
        finally:pointer.close()


class ConfirmTests(unittest.TestCase):
    """Confirm before act: a possibly irreversible input waits for the user's spoken yes."""
    setUp,tearDown,plan=ControlTests.setUp,ControlTests.tearDown,ControlTests.plan
    def judged(self, decision, score, effect=''):
        self.ex.gate.judge=Mock(return_value=Verdict(decision,score,effect))
    def sent(self):
        return [x for x in self.target.sent if x['op']!='release_all']
    def held_enter(self):
        self.judged('irreversible',.92,'Send the reply to Maria')
        p=self.plan({'key':['ENTER']})
        self.assertEqual(self.ex._do(p.action)[0],'asked')
        return p
    def answer(self, words):
        self.brain.end_turn()
        self.brain.hear(words,False)
        self.agent.cancel.reset()
        self.c.observe(self.target.grab())
        self.ex.active_plan=Planned({},self.c.epoch,self.agent.shared.request,self.c.view)
    def test_enter_in_a_chat_compose_field_asks_and_sends_nothing(self):
        p=self.held_enter()
        self.assertEqual(self.sent(),[])
        self.assertEqual(self.brain.said[-1],'Send the reply to Maria? Say yes or no.')
        self.assertEqual(self.c.requests[p.request]['status'],'asked')
        self.assertTrue(self.agent.shared.finished)
        gate=[e for e in self.c.events if e['kind']=='gate'][-1]
        self.assertEqual((gate['decision'],gate['input']),('irreversible',{'key':['ENTER']}))
    def test_enter_in_a_search_box_is_sent(self):
        p=self.plan({'key':['ENTER']})
        self.assertEqual(self.ex._do(p.action)[0],'done')
        self.assertEqual([x['op'] for x in self.sent()],['key'])
        self.assertEqual(self.ex.gate.judge.call_count,1)
    def test_clicking_send_asks(self):
        self.judged('irreversible',.88,'Send the email to John')
        p=self.plan({'click':'the Send button'})
        self.ex._find=Mock(return_value=(.5,.5))
        self.assertEqual(self.ex._do(p.action)[0],'asked')
        self.assertEqual(self.sent(),[])
        self.ex._find.assert_not_called()  # judged before it is even placed
    def test_an_unsure_or_failed_classifier_asks(self):
        for verdict in (Verdict('unsure',.4,'Submit the form'),Verdict('error',None,'')):
            self.ex.gate.judge=Mock(return_value=verdict)
            p=self.plan({'key':['CTRL','ENTER']})
            self.assertEqual(self.ex._do(p.action)[0],'asked')
        self.assertEqual(self.sent(),[])
        self.assertIn('Should I go ahead? Say yes or no.',self.brain.said[-1])
    def test_a_classifier_error_is_a_question_not_a_send(self):
        from bench.arch.confirm import IrreversibleCheck
        check=IrreversibleCheck([],'qwen','host','test')
        try:
            check.client.call=Mock(side_effect=httpx.ReadTimeout('slow'))
            self.assertEqual(check.judge(b'jpeg','send it',{'key':['ENTER']},[]).decision,'error')
            check.client.call=Mock(return_value=({'observed':'x','irreversible':'no','score':'n/a','effect':''},{}))
            self.assertEqual(check.judge(b'jpeg','send it',{'key':['ENTER']},[]).decision,'error')
        finally:check.close()
    def test_an_ordinary_click_makes_no_extra_model_call(self):
        self.ex.gate.judge=Mock(side_effect=AssertionError('extra model call'))
        p=self.plan({'click':'the Downloads folder'})
        self.ex._find=Mock(return_value=(.5,.5))
        self.assertEqual(self.ex._do(p.action)[0],'done')
        for action in ({'type':'hello Maria'},{'key':['CTRL','L']},{'scroll':3}):
            self.assertEqual(self.ex._do(action)[0],'done')
    def test_delete_right_after_typing_edits_the_text(self):
        self.assertIsNone(candidate({'key':['DELETE']},after_typing=True))
        self.assertIsNotNone(candidate({'key':['DELETE']},after_typing=False))
        self.assertIsNotNone(candidate({'key':['SHIFT','DELETE']},after_typing=True))
        self.assertIsNotNone(candidate({'type':'thanks!\n'},after_typing=False))
    def test_yes_approves_only_the_held_input_once(self):
        self.held_enter()
        self.answer('Yeah, go ahead.')
        self.assertIn('Send exactly this input now',self.c.approvals.note(self.agent.shared.request))
        self.ex.gate.judge=Mock(side_effect=AssertionError('the approved input is not judged again'))
        self.assertEqual(self.ex._do({'key':['ENTER']})[0],'done')
        self.assertEqual([x['op'] for x in self.sent()],['key'])
        self.judged('irreversible',.9,'Send the reply to Maria')
        self.assertEqual(self.ex._do({'key':['ENTER']})[0],'asked')  # once: a second send asks again
        self.assertEqual(len(self.sent()),1)
        answers=[e for e in self.c.events if e['kind']=='gate-answer']
        self.assertEqual((answers[-1]['answer'],answers[-1]['approved']),('Yeah, go ahead.',True))
    def test_yes_does_not_approve_a_different_input(self):
        self.held_enter()
        self.answer('yes')
        self.judged('irreversible',.9,'Delete the folder')
        self.ex._find=Mock(return_value=(.5,.5))
        self.assertEqual(self.ex._do({'click':'Delete folder'})[0],'asked')
        self.assertEqual(self.sent(),[])
    def test_a_yes_to_an_earlier_request_does_not_approve_a_new_one(self):
        self.held_enter()
        self.answer('open firefox')  # a new request in between drops the held input
        self.answer('yes')
        self.judged('irreversible',.9,'Send the reply to Maria')
        self.assertEqual(self.ex._do({'key':['ENTER']})[0],'asked')
        self.assertEqual(self.sent(),[])
    def test_anything_but_a_clear_yes_drops_the_held_input(self):
        for words in ('no','yes but send it to John instead','wait','please'):
            self.assertFalse(is_yes(words),words)
        for words in ('yes','Yeah.','go ahead','do it','send it','yes please, do it','OK'):
            self.assertTrue(is_yes(words),words)
        self.held_enter()
        self.answer('no')
        self.judged('irreversible',.9,'Send the reply to Maria')
        self.assertEqual(self.ex._do({'key':['ENTER']})[0],'asked')
        self.assertEqual(self.sent(),[])
    def test_the_prompt_never_types_credentials(self):
        self.assertIn('Never type passwords or other credentials',PROMPT)

if __name__=='__main__':unittest.main()
