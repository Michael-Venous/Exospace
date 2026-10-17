import unittest
from bench.experiments.routine_experiment import RoutineExecutor
import tests.test_recipe_experiment as recipe_tests


class RoutineTests(recipe_tests.RecipeTests):
    def executor(self):
        ex,ctx=super().executor()
        ex.__class__=RoutineExecutor
        ex.using_routine=False;ex.verified=[]
        return ex,ctx

    def test_ledger_requires_all_stages_and_final_evidence(self):
        ex,ctx=self.executor()
        ex.next_actions(b'first',ctx);ex.acknowledge(True)
        ex.next_actions(b'second',ctx);ex.acknowledge(True)
        ex.jev.assess.side_effect=lambda *a,**kw:({'previous':0.},{})
        with self.assertRaisesRegex(RuntimeError,'unsupported LLM fallback'):
            ex.next_actions(b'no effect',ctx)
        self.assertEqual(ex.verified,[0])
        ex.model.generate.assert_not_called()

    def test_global_eligibility_is_not_required_for_future_controls(self):
        ex,ctx=self.executor();ex.next_actions(b'first',ctx)
        ex.jev.evaluate.assert_not_called()


if __name__=='__main__':unittest.main()
