import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from brain.agent.jev_flash import JevFlashExecutor, validate_steps
from brain.agent.executor import StepContext
from brain.agent.memory import Memory
from brain.cancel import CancelToken, Cancelled
from brain.types import DeviceProfile, Rect


def stage(target, expected='next screen visible', kind='click'):
    return {'ready':'screen ready','expected':expected,'actions':[{'type':kind,'target':target}]}


class PlannerCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.executor=object.__new__(JevFlashExecutor)
        self.executor.cancel=CancelToken();self.executor.model=Mock();self.executor.jev=Mock()
        self.executor.replanner=self.executor.model  # the same mock: call counts cover both
        self.executor.check_window=False  # the window-size check has its own test
        self.executor.jev.min_presence=.85
        self.executor._key=None;self.executor.executed=[];self.executor._reset()
        self.ctx=StepContext('create event',DeviceProfile('linux',Rect(0,0,900,600),'mouse',1),
                             Memory.load(Path(self.temp.name)/'memory.json'),[],1,10,run_id='a')
        self.plan={'reasoning':'Open and save','completion':'saved summary visible',
                   'steps':[stage('Open'),stage('Save','saved summary visible')]}
        self.executor.model.generate.return_value=self.plan
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: ({k:.99 for k in checks},{k:(.25,.75) for k in targets})
        self.executor.jev.completion.return_value=.99
        self.executor.jev.last_scores={}
        self.executor.jev.last_points={}
        zoom=patch('brain.agent.jev_flash.refine',return_value=None)  # zoom keeps each point
        self.refine=zoom.start();self.addCleanup(zoom.stop)


class SemanticTests(PlannerCase):
    def test_cached_sequence_calls_flash_once(self):
        for expected in ('click','click','done'):
            result=self.executor.next_actions(b'image',self.ctx)
            self.assertEqual(result.actions[0].type,expected)
            self.executor.acknowledge(True)
        self.executor.model.generate.assert_called_once()
        self.assertEqual(self.executor.jev.assess.call_count,2)
        goal,executed=self.executor.jev.completion.call_args.args[1:3]
        self.assertEqual((goal,executed),(self.ctx.goal,['click Open','click Save']))

    def test_already_complete_plan_requires_independent_jev_verification(self):
        self.plan['steps']=[]
        result=self.executor.next_actions(b'completed screen',self.ctx)
        self.assertEqual(result.actions[0].type,'done')
        self.executor.model.generate.assert_called_once()
        self.executor.jev.assess.assert_not_called()
        self.assertEqual(self.executor.jev.completion.call_args.args[1:3],(self.ctx.goal,[]))

    def test_completion_cannot_replace_the_original_goal_with_weaker_evidence(self):
        from dataclasses import replace
        ctx=replace(self.ctx,goal='Open the catalog then click the first result and verify it opened')
        self.plan['steps']=[]
        self.plan['completion']='The first result is visible'
        # The user required opening a result; nothing was executed, so it can't be done.
        self.executor.jev.completion.side_effect=lambda image,goal,executed,**kw: float(bool(executed))
        result=self.executor.next_actions(b'catalog list only',ctx)
        self.assertFalse(result.actions)
        self.assertTrue(result.rejected)
        self.assertTrue(all(c.args[1]==ctx.goal for c in self.executor.jev.completion.call_args_list))

    def test_empty_plan_cannot_claim_completion_when_jev_disagrees(self):
        self.plan['steps']=[]
        self.executor.jev.completion.return_value=0.
        result=self.executor.next_actions(b'incomplete screen',self.ctx)
        self.assertFalse(result.actions)
        self.assertTrue(result.rejected)

    def test_empty_plan_without_a_completion_criterion_is_invalid(self):
        with self.assertRaises(ValueError):
            validate_steps({'steps':[],'completion':''})

    def test_discarded_action_does_not_advance(self):
        first=self.executor.next_actions(b'image',self.ctx)
        second=self.executor.next_actions(b'new frame',self.ctx)
        self.assertEqual(self.executor.cursor,0)
        self.assertEqual(first.actions,second.actions)
        self.executor.model.generate.assert_called_once()

    def test_hover_is_separate_from_submenu_click(self):
        self.plan['steps']=[stage('Tools','submenu visible','hover'),stage('Export')]
        result=self.executor.next_actions(b'image',self.ctx)
        self.assertEqual((result.actions[0].type,result.actions[0].name),('skill','hover'))
        self.executor.acknowledge(True)
        result=self.executor.next_actions(b'submenu',self.ctx)
        self.assertEqual(result.actions[0].type,'click')
        self.executor.model.generate.assert_called_once()

    def test_failed_expectation_replans_before_next_click(self):
        self.executor.next_actions(b'image',self.ctx);self.executor.acknowledge(True)
        self.executor.jev.assess.side_effect=[({'previous':.1,'ready':.99},{'target':(.5,.5)}),
                                              ({'ready':.99},{'target':(.1,.2)})]
        result=self.executor.next_actions(b'wrong screen',self.ctx)
        self.assertEqual(self.executor.model.generate.call_count,2)
        self.assertEqual(result.actions[0].x,.1)

    def test_goal_change_clears_old_plan(self):
        from dataclasses import replace
        self.executor.next_actions(b'image',self.ctx);self.executor.acknowledge(True)
        self.executor.next_actions(b'new',replace(self.ctx,goal='different goal'))
        self.assertEqual(self.executor.model.generate.call_count,2)
        self.assertEqual(self.executor.cursor,0)

    def test_cancelled_grounding_never_returns_action(self):
        def late(*a,**kw):
            self.executor.cancel.cancel('interrupt')
            return {'ready':.99},{'target':(.5,.5)}
        self.executor.jev.assess.side_effect=late
        with self.assertRaises(Cancelled):self.executor.next_actions(b'image',self.ctx)

    def test_invalid_coordinates_tabs_and_key_sequences(self):
        for action in ({'type':'click','target':'field','x':1},
                       {'type':'type_text','text':'a\tb'},
                       {'type':'hotkey','keys':['TAB','TAB']}):
            self.plan['steps']=[{'ready':'ready','expected':'done','actions':[action]}]
            with self.assertRaises(ValueError): validate_steps(self.plan)

    def test_ambiguous_request_asks_through_flash_without_grounding(self):
        from dataclasses import replace
        self.executor.model.generate.return_value={
            'reasoning':'Two matching buttons','completion':'chosen button clicked',
            'steps':[{'ready':'screen visible','expected':'user clarifies',
                      'actions':[{'type':'ask_user','message':'Left or right Continue button?'}]}]}
        result=self.executor.next_actions(b'image',replace(self.ctx,goal='Click Continue'))
        self.assertEqual(result.actions[0].type,'ask_user')
        self.executor.model.generate.assert_called_once()
        self.executor.jev.assess.assert_not_called()


    def test_ask_flash_choice_replans_on_a_changed_screen_only(self):
        self.executor.jev.assess.side_effect=[
            ({'ready':.99,'route_execute':0},{'target':(.5,.5)}),  # the planned screen: the vote is ignored
            ({'ready':.99,'previous':.99,'route_execute':0},{'target':(.4,.4)}),  # a new screen: replan
            ({'ready':.99,'route_execute':1},{'target':(.2,.3)})]
        self.assertEqual(self.executor.next_actions(b'image',self.ctx).actions[0].x,.5)
        self.executor.acknowledge(True)
        result=self.executor.next_actions(b'image2',self.ctx)
        self.assertEqual(self.executor.model.generate.call_count,2)
        self.assertEqual(result.actions[0].x,.2)


    def test_visual_fallback_is_one_stage_and_never_reuses_stale_coordinates(self):
        fallback={'reasoning':'Locate visually','completion':'heading hovered',
                  'steps':[{'ready':'heading visible','expected':'menu hidden',
                            'actions':[{'type':'hover','target':'heading','fallback_x':100,'fallback_y':200}]}]}
        fresh={'reasoning':'Locate again','completion':'heading hovered',
               'steps':[{'ready':'heading visible','expected':'menu hidden',
                         'actions':[{'type':'hover','target':'heading','fallback_x':300,'fallback_y':400}]}]}
        self.executor.model.generate.side_effect=[self.plan,fallback,fresh]
        self.executor.jev.assess.side_effect=[({'ready':.99,'route_execute':0},{'target':None})]
        first=self.executor.next_actions(b'old image',self.ctx)
        self.assertEqual(first.actions[0].args,{'x':.1,'y':.2})
        # No acknowledgement: the single-writer controller discarded this action.
        second=self.executor.next_actions(b'changed image',self.ctx)
        self.assertEqual(second.actions[0].args,{'x':.3,'y':.4})
        self.assertEqual(self.executor.model.generate.call_count,3)
        self.assertEqual(self.executor.jev.assess.call_count,1)
        self.assertNotIn('fallback_x',validate_steps(fallback)[0]['actions'][0])  # unauthorized: ignored, Jev grounds
        validate_steps(fallback,True)
        unnamed={'reasoning':'r','completion':'c','steps':[{'ready':'r','expected':'e',
                 'actions':[{'type':'skill','name':'open_app','text':' '}]}]}
        with self.assertRaises(ValueError):validate_steps(unnamed)


    def test_zoom_reaims_targets_and_rescues_near_misses(self):
        self.refine.return_value=(.3,.7)
        plan=self.executor.next_actions(b'image',self.ctx)
        self.assertEqual((plan.actions[0].x,plan.actions[0].y),(.3,.7))
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: ({k:.99 for k in checks},{k:None for k in targets})
        self.executor.jev.last_scores={'target':{'presence':.6,'unique':.7}}
        self.executor.jev.last_points={'target':(.4,.4)}
        self.executor._key=None
        plan=self.executor.next_actions(b'other',self.ctx)
        self.assertEqual((plan.actions[0].x,plan.actions[0].y),(.3,.7))
        self.assertEqual(self.refine.call_args.args[3],(.4,.4))

    def test_typing_without_focus_clicks_the_field_first(self):
        self.plan['steps']=[{'ready':'search page','expected':'results','actions':[{'type':'type_text','text':'flexbox'}]}]
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: (
            {'ready':.99,'focused':.1},{'field':(.4,.5)})
        plan=self.executor.next_actions(b'image',self.ctx)
        self.assertEqual([a.type for a in plan.actions],['click','type_text'])
        self.assertEqual((plan.actions[0].x,plan.actions[0].y),(.4,.5))
        self.executor._key=None
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: ({'ready':.99,'focused':.95},{})
        self.assertEqual([a.type for a in self.executor.next_actions(b'other',self.ctx).actions],['type_text'])

    def test_a_rejected_plan_is_replanned_with_more_thinking(self):
        self.executor.replanner=Mock()
        self.executor.replanner.generate.return_value=self.plan
        self.executor.jev.assess.side_effect=[({'ready':.2},{'target':(.5,.5)}),({'ready':.99},{'target':(.2,.3)})]
        self.executor.next_actions(b'image',self.ctx)
        self.assertEqual(self.executor.model.generate.call_count,1)  # the first plan: fast
        self.assertEqual(self.executor.replanner.generate.call_count,1)  # after Jev's veto: high thinking

    def test_a_failed_replan_falls_back_to_the_fast_planner(self):
        self.executor.replanner=Mock()
        self.executor.replanner.generate.side_effect=RuntimeError('504 deadline exceeded')
        self.executor.jev.assess.side_effect=[({'ready':.2},{'target':(.5,.5)}),({'ready':.99},{'target':(.2,.3)})]
        self.assertEqual(self.executor.next_actions(b'image',self.ctx).actions[0].x,.2)
        self.assertEqual(self.executor.model.generate.call_count,2)

    def test_a_small_window_is_maximized_before_planning_on(self):
        self.executor.check_window=True
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: (
            {k:.99 for k in checks},{k:((.9,.02) if k=='maximize' else (.25,.75)) for k in targets})
        with patch('brain.agent.jev_flash.refine',return_value=(.91,.02)):
            plan=self.executor.next_actions(b'image',self.ctx)
        self.assertEqual((plan.actions[0].type,plan.actions[0].x,plan.actions[0].y),('click',.91,.02))
        self.assertIsNone(self.executor._key)  # re-planned on the maximized screen
        self.assertNotEqual(self.executor.next_actions(b'image',self.ctx).reasoning,'Maximize the app window first')


class IrreversibleGuardTests(PlannerCase):
    """With race's approval policy, a plan can't send, delete, pay or submit on its own."""

    def guard(self,approved):
        self.executor.approved=approved
        self.plan['steps']=[stage('Send button','message sent')]
        self.executor.jev.assess.side_effect=lambda image,checks,targets,irreversible=False,**kw: (
            {**{k:.99 for k in checks},**({'irreversible':.95} if irreversible else {})},{k:(.25,.75) for k in targets})

    def test_unapproved_irreversible_stage_asks_instead_of_clicking(self):
        self.guard(lambda:False)
        result=self.executor.next_actions(b'compose',self.ctx)
        self.assertEqual((result.actions,result.held),([],'click Send button'))
        self.assertTrue(self.executor.jev.assess.call_args.kwargs['irreversible'])

    def test_approved_run_takes_the_irreversible_stage(self):
        self.guard(lambda:True)
        result=self.executor.next_actions(b'compose',self.ctx)
        self.assertEqual((result.actions[0].type,result.held),('click',''))
        self.assertFalse(self.executor.jev.assess.call_args.kwargs['irreversible'])

    def test_without_a_policy_no_irreversible_question_is_asked(self):
        self.guard(None)
        result=self.executor.next_actions(b'compose',self.ctx)
        self.assertEqual(result.actions[0].type,'click')
        self.assertFalse(self.executor.jev.assess.call_args.kwargs['irreversible'])

    def test_visual_fallback_stage_is_held_too(self):
        self.executor.approved=lambda:False
        fallback={'reasoning':'Locate visually','completion':'message sent',
                  'steps':[{'ready':'compose visible','expected':'message sent',
                            'actions':[{'type':'click','target':'Send button','fallback_x':100,'fallback_y':900}]}]}
        self.plan['steps']=[stage('Send button','message sent')]
        self.executor.model.generate.side_effect=[self.plan,fallback]
        self.executor.jev.assess.side_effect=[({'ready':.99,'route_execute':0,'irreversible':.95},{'target':None}),
                                              ({'route_execute':1,'safe':1,'irreversible':.95},{})]
        result=self.executor.next_actions(b'compose',self.ctx)
        self.assertEqual((result.actions,result.held),([],'click Send button'))
        self.assertEqual(self.executor.jev.assess.call_count,2)

if __name__=='__main__':unittest.main()
