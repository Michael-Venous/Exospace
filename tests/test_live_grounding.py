import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from brain.agent.live_grounding_experiment import LiveGroundedRoutine
from brain.agent.refined_routine_experiment import RefinedRoutineExecutor
from brain.cancel import CancelToken


class GroundingTests(unittest.TestCase):
    def executor(self):
        ex=object.__new__(LiveGroundedRoutine);ex.cancel=CancelToken();ex.grounder=Mock();ex.jev=Mock()
        ex.jev.min_presence=.85;ex.jev.last_scores={'target':{'presence':.99,'unique':.99,'box_confidence':.2}}
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);ex.grounding_log=Path(temp.name)/'log'
        return ex

    def test_ambiguous_targets_never_reach_coordinate_fallback(self):
        ex=self.executor();ex.jev.last_scores['target']['unique']=.2
        with patch.object(RefinedRoutineExecutor,'refined_assess',return_value=({'ready':1.},{'target':None})):
            _,points=ex.refined_assess(b'image',{}, {'target':'Continue'})
        ex.grounder.generate.assert_not_called();self.assertIsNone(points['target'])

    def test_only_missing_coordinates_are_replaced(self):
        ex=self.executor();ex.grounder.generate.return_value={'target_x':400,'target_y':300}
        with patch.object(RefinedRoutineExecutor,'refined_assess',return_value=({'ready':1.},{'target':None,'destination':(.8,.7)})):
            _,points=ex.refined_assess(b'image',{}, {'target':'Report','destination':'Archive'})
        self.assertEqual(points,{'target':(.4,.3),'destination':(.8,.7)})

    def test_invalid_coordinates_cannot_be_clamped_into_actions(self):
        ex=self.executor();ex.grounder.generate.return_value={'target_x':2000,'target_y':300}
        with patch.object(RefinedRoutineExecutor,'refined_assess',return_value=({'ready':1.},{'target':None})):
            with self.assertRaisesRegex(ValueError,'Invalid Live grounding'):ex.refined_assess(b'image',{}, {'target':'Report'})


if __name__=='__main__':unittest.main()
