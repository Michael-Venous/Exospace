import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bench.experiments.recipe_experiment import RecipeExecutor, clauses, compile_recipe
from brain.agent.executor import StepContext
from brain.agent.memory import Memory
from brain.cancel import CancelToken, Cancelled
from brain.types import DeviceProfile, Rect


class RecipeTests(unittest.TestCase):
    def test_literal_delimiters_are_not_commands(self):
        goal='enter "click Save; then delete and close" in Note; click Save card'
        plan=compile_recipe(goal)
        self.assertEqual(len(plan['steps']),2)
        self.assertEqual(plan['steps'][0]['actions'][-1]['text'],'click Save; then delete and close')

    def test_rejects_incomplete_or_unsupported_requests(self):
        for goal in ['enter "hello in Name', 'click A;', 'click A; calculate the total',
                     'click the cheapest item', 'click A if available', 'enter "café" in Name',
                     'click A and click B', 'type a friendly response in Note']:
            with self.subTest(goal=goal):self.assertIsNone(compile_recipe(goal))

    def test_preserves_sequence_and_double_click(self):
        plan=compile_recipe('hover over Tools, then click Export; double-click Folder')
        self.assertEqual([s['actions'][0]['type'] for s in plan['steps']],['hover','click','double_click'])

    def executor(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        ex=object.__new__(RecipeExecutor);ex.cancel=CancelToken();ex.model=Mock();ex.jev=Mock();ex.replanner=ex.model;ex.check_window=False
        zoom=patch('brain.agent.jev_flash.refine',return_value=None);zoom.start();self.addCleanup(zoom.stop)
        ex.jev.min_presence=.85;ex.jev.last_scores={};ex.jev.last_points={};ex.jev.completion.return_value=.99;ex._key=None;ex._hover_context=None;ex._menu_click=False;ex.executed=[];ex._reset()
        ex.forbid_llm=True;ex.recipe_admitted=False;ex.recipe_attempted=False
        ex.jev.evaluate.return_value=({'eligibility':{'choice':'literal'}},(100,100),(100,100))
        ex.jev.assess.side_effect=lambda image,checks,targets,**kw:({k:1. for k in checks},{k:(.5,.5) for k in targets})
        ctx=StepContext('click Open; click first result',DeviceProfile('linux',Rect(0,0,100,100),'mouse',1),Memory.load(Path(temp.name)/'memory.json'),[],1,10,run_id='r')
        return ex,ctx

    def test_acknowledgement_required_before_advancing(self):
        ex,ctx=self.executor()
        first=ex.next_actions(b'frame',ctx)
        self.assertEqual(ex.next_actions(b'frame',ctx).actions,first.actions)
        self.assertEqual(ex.cursor,0)
        ex.acknowledge(True);ex.next_actions(b'next',ctx)
        self.assertEqual(ex.cursor,1)
        ex.acknowledge(True)
        self.assertEqual(ex.next_actions(b'complete',ctx).actions[0].type,'done')
        ex.model.generate.assert_not_called()

    def test_failed_grounding_cannot_emit_action_without_fallback(self):
        ex,ctx=self.executor()
        ex.jev.assess.return_value=({'ready':0.},{'target':None})
        ex.jev.assess.side_effect=None
        with self.assertRaisesRegex(RuntimeError,'unsupported LLM fallback'):
            ex.next_actions(b'frame',ctx)
        ex.model.generate.assert_not_called()

    def test_cancelled_recipe_cannot_return_action(self):
        ex,ctx=self.executor();ex.cancel.cancel('barge-in')
        with self.assertRaises(Cancelled):ex.next_actions(b'frame',ctx)
        ex.model.generate.assert_not_called()


if __name__=='__main__':unittest.main()
