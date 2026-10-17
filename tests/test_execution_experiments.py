import unittest
from concurrent.futures import Future
from unittest.mock import Mock, patch

from brain.agent.live_direct_experiment import LiveDirectExecutor
from brain.agent.prefix_experiment import PrefixExecutor
from brain.agent.jev_flash import JevFlashExecutor
import tests.test_recipe_experiment as fixtures


class ProtocolTests(unittest.TestCase):
    def context(self):
        helper=fixtures.RecipeTests()
        ex,ctx=helper.executor();self.addCleanup(helper.doCleanups)
        return ex,ctx

    def test_hover_then_hidden_click_rejected(self):
        _,ctx=self.context();ex=object.__new__(LiveDirectExecutor);ex.model=Mock()
        ex.model.generate.return_value={'actions':[{'type':'hover','x':100,'y':200},{'type':'click','x':300,'y':400}]}
        with self.assertRaisesRegex(ValueError,'Observe after'):ex.next_actions(b'image',ctx)

    def test_hover_maps_to_existing_controller_skill(self):
        _,ctx=self.context();ex=object.__new__(LiveDirectExecutor);ex.model=Mock()
        ex.model.generate.return_value={'actions':[{'type':'hover','x':100,'y':200}]}
        action=ex.next_actions(b'image',ctx).actions[0]
        self.assertEqual((action.type,action.name,action.args),('skill','hover',{'x':.1,'y':.2}))

    def test_unexecuted_prefix_never_consumes_background_plan(self):
        source,ctx=self.context();ex=object.__new__(PrefixExecutor);ex.__dict__.update(source.__dict__)
        key=(ctx.run_id,ctx.goal,ctx.profile.os,ctx.profile.connected_at)
        ex._key=key;ex.future_key=key;ex.future=Future()
        future=ex.future;ex.pending=False
        plan=ex.next_actions(b'changed screen',ctx)
        self.assertTrue(plan.refresh_screen);self.assertFalse(plan.actions);self.assertTrue(future.cancelled())

    def test_new_goal_discards_old_background_plan(self):
        source,ctx=self.context();ex=object.__new__(PrefixExecutor);ex.__dict__.update(source.__dict__)
        ex.future_key=('old goal',);ex.future=Future();future=ex.future
        # Prevent starting a new background request; only test old plan invalidation.
        with patch('brain.agent.prefix_experiment.clauses',return_value=None),patch.object(JevFlashExecutor,'next_actions',return_value='fresh'):
            self.assertEqual(ex.next_actions(b'image',ctx),'fresh')
        self.assertTrue(future.cancelled());self.assertIsNone(ex.future)


    def test_invalid_background_proposal_replans_after_completed_prefix(self):
        source,ctx=self.context();ex=object.__new__(PrefixExecutor);ex.__dict__.update(source.__dict__)
        key=(ctx.run_id,ctx.goal,ctx.profile.os,ctx.profile.connected_at)
        ex._key=key;ex.future_key=key;ex.pending=True;ex.future=Future()
        ex.future.set_result({'steps':[{'ready':'ready','expected':'visible','actions':[]}],'completion':'done'})
        ex._new_plan=Mock()
        with patch.object(JevFlashExecutor,'next_actions',return_value='fresh checked plan'):
            self.assertEqual(ex.next_actions(b'after prefix',ctx),'fresh checked plan')
        ex._new_plan.assert_called_once()
        self.assertEqual(ex._new_plan.call_args.args[:2],(b'after prefix',ctx))
        self.assertIn('post-prefix',ex._new_plan.call_args.args[2])


if __name__=='__main__':unittest.main()
