import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from brain.agent.refined_routine_experiment import RefinedRoutineExecutor


class RefineTests(unittest.TestCase):
    def executor(self):
        ex=object.__new__(RefinedRoutineExecutor);ex.jev=Mock();ex.jev.min_presence=.85;ex.jev.min_confidence=.6
        ex.jev.last_scores={'target':{'presence':.99,'unique':.99,'box_confidence':.3}}
        ex.original_assess=Mock(side_effect=[({'ready':.99,'route_execute':1.},{'target':None}),({'ready':.99,'route_execute':1.},{'target':(.2,.3)})])
        ex.jev.evaluate.return_value=({'target':{'choice':'button','probabilities':{'button':.99}}},None,None)
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);ex.refinement_log=Path(temp.name)/'log'
        return ex

    def test_one_refinement_preserves_checks_and_thresholds(self):
        ex=self.executor();checks={'ready':'control is available'}
        _,points=ex.refined_assess(b'image',checks,{'target':'the interactive control identified by: Open'})
        self.assertEqual(points['target'],(.2,.3));self.assertEqual(ex.original_assess.call_count,2)
        self.assertEqual(ex.original_assess.call_args.args[1],checks)
        self.assertEqual(ex.original_assess.call_args.args[2]['target'],'The button labeled or described as: Open')
        self.assertEqual(ex.jev.min_confidence,.6)

    def test_ambiguous_target_never_uses_refinement_to_bypass_gate(self):
        ex=self.executor();ex.jev.last_scores['target']['unique']=.2
        _,points=ex.refined_assess(b'image',{}, {'target':'Continue'})
        self.assertIsNone(points['target']);ex.jev.evaluate.assert_not_called()
        self.assertEqual(ex.original_assess.call_count,1)

    def test_uncertain_type_stops_without_second_grounding(self):
        ex=self.executor();ex.jev.evaluate.return_value=({'target':{'choice':'uncertain'}},None,None)
        _,points=ex.refined_assess(b'image',{}, {'target':'Open'})
        self.assertIsNone(points['target']);self.assertEqual(ex.original_assess.call_count,1)


if __name__=='__main__':unittest.main()
