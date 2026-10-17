import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile

from brain.agent.jev_flash import JevFlashExecutor, direct_command, validate_steps
from brain.agent.executor import StepContext
from brain.agent.memory import Memory
from brain.cancel import CancelToken, Cancelled
from brain.types import DeviceProfile, Rect


def stage(target, expected='next screen visible', kind='click'):
    return {'ready':'screen ready','expected':expected,'actions':[{'type':kind,'target':target}]}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.executor=object.__new__(JevFlashExecutor)
        self.executor.cancel=CancelToken();self.executor.model=Mock();self.executor.jev=Mock()
        self.executor.jev.min_presence=.85
        self.executor._key=None;self.executor._reset()
        self.ctx=StepContext('create event',DeviceProfile('linux',Rect(0,0,900,600),'mouse',1),
                             Memory.load(Path(self.temp.name)/'memory.json'),[],1,10,run_id='a')
        self.plan={'reasoning':'Open and save','completion':'saved summary visible',
                   'steps':[stage('Open'),stage('Save','saved summary visible')]}
        self.executor.model.generate.return_value=self.plan
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: ({k:.99 for k in checks},{k:(.25,.75) for k in targets})

    def test_cached_sequence_calls_flash_once(self):
        for expected in ('click','click','done'):
            result=self.executor.next_actions(b'image',self.ctx)
            self.assertEqual(result.actions[0].type,expected)
            self.executor.acknowledge(True)
        self.executor.model.generate.assert_called_once()
        self.assertEqual(self.executor.jev.assess.call_count,4)

    def test_already_complete_plan_requires_independent_jev_verification(self):
        self.plan['steps']=[]
        result=self.executor.next_actions(b'completed screen',self.ctx)
        self.assertEqual(result.actions[0].type,'done')
        self.executor.model.generate.assert_called_once()
        checks=self.executor.jev.assess.call_args.args[1]
        self.assertEqual(set(checks),{'complete'})
        self.assertIn(self.ctx.goal,checks['complete'])
        self.assertIn('saved summary visible',checks['complete'])

    def test_completion_cannot_replace_the_original_goal_with_weaker_evidence(self):
        from dataclasses import replace
        ctx=replace(self.ctx,goal='Open the catalog then click the first result and verify it opened')
        self.plan['steps']=[]
        self.plan['completion']='The first result is visible'
        def verify(image,checks,targets,**kwargs):
            # The user required opening a result, whereas the screen only lists it.
            values={key:float(ctx.goal not in condition) for key,condition in checks.items()}
            return values,{}
        self.executor.jev.assess.side_effect=verify
        result=self.executor.next_actions(b'catalog list only',ctx)
        self.assertFalse(result.actions)
        self.assertTrue(result.rejected)

    def test_empty_plan_cannot_claim_completion_when_jev_disagrees(self):
        self.plan['steps']=[]
        self.executor.jev.assess.side_effect=lambda image,checks,targets,**kw: (
            {k:0. for k in checks},{})
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

    def test_direct_click_uses_no_flash(self):
        from dataclasses import replace
        ctx=replace(self.ctx,goal='Click the Continue button')
        result=self.executor.next_actions(b'image',ctx)
        self.assertEqual(result.actions[0].type,'click')
        self.executor.acknowledge(True)
        self.assertEqual(self.executor.next_actions(b'new',ctx).actions[0].type,'done')
        self.executor.model.generate.assert_not_called()

    def test_compound_command_does_not_bypass_planner(self):
        self.assertIsNone(direct_command('Click Open then select Settings'))
        self.assertIsNone(direct_command('Create an event'))

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

    def test_ambiguous_direct_command_escalates_to_flash_for_clarification(self):
        from dataclasses import replace
        self.executor.jev.assess.side_effect=None
        self.executor.jev.assess.return_value=({}, {'target':None})
        self.executor.model.generate.return_value={
            'reasoning':'Two matching buttons','completion':'chosen button clicked',
            'steps':[{'ready':'screen visible','expected':'user clarifies',
                      'actions':[{'type':'ask_user','message':'Left or right Continue button?'}]}]}
        result=self.executor.next_actions(b'image',replace(self.ctx,goal='Click Continue'))
        self.assertEqual(result.actions[0].type,'ask_user')
        self.executor.model.generate.assert_called_once()
        self.assertEqual(self.executor.jev.assess.call_count,1)


    def test_ask_flash_choice_replans_even_with_valid_coordinates(self):
        self.executor.jev.assess.side_effect=[
            ({'route_execute':0},{}),
            ({'ready':.99,'route_execute':0},{'target':(.5,.5)}),
            ({'ready':.99,'route_execute':1},{'target':(.2,.3)})]
        result=self.executor.next_actions(b'image',self.ctx)
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
        self.executor.jev.assess.side_effect=[({'route_execute':0},{}),
                                             ({'ready':.99,'route_execute':0},{'target':None})]
        first=self.executor.next_actions(b'old image',self.ctx)
        self.assertEqual(first.actions[0].args,{'x':.1,'y':.2})
        # No acknowledgement: the single-writer controller discarded this action.
        second=self.executor.next_actions(b'changed image',self.ctx)
        self.assertEqual(second.actions[0].args,{'x':.3,'y':.4})
        self.assertEqual(self.executor.model.generate.call_count,3)
        self.assertEqual(self.executor.jev.assess.call_count,2)
        with self.assertRaises(ValueError):validate_steps(fallback)
        validate_steps(fallback,True)


if __name__=='__main__':unittest.main()
