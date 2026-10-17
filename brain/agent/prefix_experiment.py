"""Explicit first-action execution overlapping a background semantic planner."""
from concurrent.futures import ThreadPoolExecutor
import json
import re

from brain.agent.executor import Plan, context_prompt
from brain.agent.jev_flash import JevFlashExecutor, PROMPT, SCHEMA, validate_steps, direct_command
from brain.agent.recipe_experiment import clauses, compile_recipe


class PrefixExecutor(JevFlashExecutor):
    def __init__(self,cfg,token):
        super().__init__(cfg,token)
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.future=None;self.future_key=None;self.overlap_used=False

    def _discard(self):
        if self.future is not None:self.future.cancel()
        self.future=None;self.future_key=None

    def next_actions(self,image,ctx):
        self.cancel.check()
        key=(ctx.run_id,ctx.goal,ctx.profile.os,ctx.profile.connected_at)
        if self.future is not None and self.future_key!=key:
            self._discard();self._key=None
        if self._key!=key and self.future is None:
            parts=clauses(ctx.goal)
            recipe=compile_recipe(parts[0]) if parts and len(parts)>1 else None
            if recipe and len(recipe['steps'][0]['actions'])==1:
                stage=recipe['steps'][0]
                # Only opening/revealing an explicitly named control is eligible.
                # This limited experiment never speculates about submitting/deleting.
                first=parts[0].lower()
                if not first.startswith(('click open ', 'hover over ')):
                    return super().next_actions(image,ctx)
                prompt=context_prompt(ctx)+'\nThe controller is attempting ONLY this explicit first action: '+json.dumps(stage['actions'])+'. Propose the remainder assuming it succeeds; NEVER repeat it. The controller will verify applicability on a fresh screen. All coordinates must be grounded later. Completion covers the whole original goal.'
                self.future=self.pool.submit(self.model.generate,image,prompt,PROMPT,SCHEMA,role='flash',run_id=ctx.run_id,step=ctx.step)
                self.future_key=key;self._key=key;self.prefix=stage
                self.prefix_label=direct_command(parts[0])["target"].lower()
                values,points=self.jev.assess(image,{}, {'target':stage['actions'][0]['target']},request=parts[0],proposal=stage['actions'],run_id=ctx.run_id,step=ctx.step)
                self.cancel.check()
                if all(v>=self.jev.min_presence for v in values.values()) and points.get('target') is not None:
                    self.overlap_used=True;self.stages=[stage]
                    return Plan('Explicit prefix while Flash plans',self._ground(stage,points),[])
                self._discard();self._key=None
                return super().next_actions(image,ctx)
        if self.future is not None:
            if not self.pending:
                # Stale-screen rejection did not execute the prefix. Its old
                # grounding and assumed-success plan are unusable.
                self._discard();self._key=None
                return Plan('Discard unexecuted prefix plan; refresh',[],[],refresh_screen=True)
            proposal=self.cancel.run(self.future.result)
            self.cancel.check();self._discard()
            try:
                stages=validate_steps(proposal)
            except ValueError as exc:
                # A planner may need the post-prefix screen rather than guessing
                # hidden controls. Recover from that screen, preserving the
                # actual prefix acknowledgement/history instead of aborting.
                self._new_plan(image,ctx,'Background proposal unusable; plan from the current post-prefix screen. '+str(exc))
                return super().next_actions(image,ctx)
            target=' '.join(re.findall(r'\w+',self.prefix_label))
            duplicate=any(target in ' '.join(re.findall(r'\w+',a.get('target','').lower())) for s in stages for a in s['actions'])
            if duplicate:
                self._new_plan(image,ctx,'Background plan repeats completed prefix; use execution history to plan only the remainder.')
            else:
                self.pending=False;self.cursor=0;self.direct=None
                self.stages=stages;self.completion=proposal['completion'];self.last_reason=proposal.get('reasoning','')
                # Existing checks gate first remainder action against CURRENT image.
        return super().next_actions(image,ctx)

    def close(self):
        self._discard();self.pool.shutdown(wait=True,cancel_futures=True);super().close()


class RefinedPrefixExecutor(PrefixExecutor):
    """Same overlap algorithm, adding the already-tested Jev control-type retry."""
    def __init__(self,cfg,token):
        super().__init__(cfg,token)
        from brain.agent.refined_routine_experiment import RefinedRoutineExecutor
        self.original_assess=self.jev.assess
        self.refinement_log=cfg.model_log.parent/'refinement.jsonl'
        self.jev.assess=lambda *args,**kwargs: RefinedRoutineExecutor.refined_assess(self,*args,**kwargs)
