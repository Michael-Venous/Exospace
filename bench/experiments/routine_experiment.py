"""Experimental stepwise Jev routines with a verified execution ledger."""
from brain.agent.executor import Plan
from bench.experiments.recipe_experiment import RecipeExecutor, compile_recipe
from brain.types import Action


class RoutineExecutor(RecipeExecutor):
    """The parser binds literal commands; Jev judges only the next visible step.

    Completion requires all parsed clauses acknowledged, prior transitions verified,
    and the last action's visible effect. It does not infer historical clicks from
    a lone final image, or accept a model's partial rewrite of the user's goal.
    """
    def __init__(self,cfg,cancel,*,forbid_llm=False):
        super().__init__(cfg,cancel,forbid_llm=forbid_llm)
        self.using_routine=False
        self.verified=[]

    def next_actions(self,image,ctx):
        self.cancel.check()
        key=(ctx.run_id,ctx.goal,ctx.profile.os,ctx.profile.connected_at)
        if self._key!=key:
            recipe=compile_recipe(ctx.goal)
            self.using_routine=recipe is not None
            self.recipe_attempted=self.using_routine
            self.verified=[]
            if recipe:
                self._reset();self._key=key;self._hover_context=None;self._menu_click=False
                self.stages=recipe['steps'];self.completion=recipe['completion']
                self.recipe_admitted=True
                last=self.stages[-1]
                if last['actions'][0]['type']=='drag':
                    last['expected']='The dragged item '+last['actions'][0]['target']+' is visibly at '+last['actions'][0]['destination']+' or a visible message confirms that move.'
                elif len(last['actions'])==1:
                    last['expected']=('The visible result of activating '+last['actions'][0]['target']+
                        ' is now present. Merely showing an available target or link is NOT evidence of activation. '
                        'A visible success message, opened content, or resulting interface can provide evidence.')
                ctx.memory.set_subtasks([s['expected'] for s in self.stages])
        if not self.using_routine:
            # Bypass RecipeExecutor's global eligibility classifier. This variant
            # falls back directly to the unchanged Flash executor on unsupported syntax.
            from brain.agent.jev_flash import JevFlashExecutor
            return JevFlashExecutor.next_actions(self,image,ctx)
        next_cursor=self.cursor+int(self.pending)
        final=next_cursor==len(self.stages)
        stage=None if final else self.stages[next_cursor]
        checks={}
        if self.pending:checks['previous']=self.stages[self.cursor]['expected']
        if stage:checks['ready']=stage['ready']
        targets={'target':stage['actions'][0]['target']} if stage else {}
        if stage and stage['actions'][0]['type']=='drag':targets['destination']=stage['actions'][0]['destination']
        values,points=self.jev.assess(image,checks,targets,
            request='',proposal=stage['actions'] if stage else None,
            run_id=ctx.run_id,step=ctx.step)
        self.cancel.check()
        if all(v>=self.jev.min_presence for v in values.values()) and all(p is not None for p in points.values()):
            if self.pending:self.verified.append(self.cursor)
            self.pending=False;self.cursor=next_cursor
            if final:
                if self.verified!=list(range(len(self.stages))):
                    raise RuntimeError('Incomplete routine ledger')
                return Plan('Every literal stage executed and its result checked',[Action('done',summary='Verified the entire literal sequence.')],[])
            return Plan(f'Jev routine stage {next_cursor+1}/{len(self.stages)}',self._ground(stage,points),[])
        reason='Routine needs recovery; visual checks='+repr(values)+'; unresolved targets='+repr(points)
        self._new_plan(image,ctx,reason,visual_fallback=any(p is None for p in points.values()))
        self.using_routine=False
        from brain.agent.jev_flash import JevFlashExecutor
        return JevFlashExecutor.next_actions(self,image,ctx)
