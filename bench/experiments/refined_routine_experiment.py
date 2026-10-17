"""One bounded Jev-only target refinement before considering an LLM fallback."""
import json
from brain.agent.jev import probability
from bench.experiments.routine_experiment import RoutineExecutor


class RefinedRoutineExecutor(RoutineExecutor):
    def __init__(self,cfg,cancel,*,forbid_llm=False):
        super().__init__(cfg,cancel,forbid_llm=forbid_llm)
        self.original_assess=self.jev.assess
        self.jev.assess=self.refined_assess
        self.refinement_log=cfg.model_log.parent/'refinement.jsonl'

    def refined_assess(self,image,checks,targets,**kw):
        values,points=self.original_assess(image,checks,targets,**kw)
        scores=dict(self.jev.last_scores)
        if not all(v>=self.jev.min_presence for v in values.values()):return values,points
        eligible=[key for key,point in points.items() if point is None
            and min(scores[key]['presence'],scores[key]['unique'])>=self.jev.min_presence
            and scores[key]['box_confidence']<self.jev.min_confidence]
        if not eligible:return values,points
        questions={key:{'type':'choice','instructions':{
            'question':'Which visible UI control type is the uniquely identified target? Choose uncertain if absent, ambiguous, or none of these. Do not change which target the user requested.',
            'target':targets[key]},'criteria':{
                'button':'A clickable button','link':'A clickable text hyperlink',
                'input':'An editable text input field','menu item':'An item inside a visible menu',
                'icon':'A graphical icon','uncertain':'Ambiguous, unavailable, or a different kind of target'}} for key in eligible}
        answers,_,_=self.jev.evaluate(image,questions,run_id=kw.get('run_id',''),step=kw.get('step',0))
        refined=dict(targets)
        for key in eligible:
            answer=answers[key];kind=answer.get('choice')
            if kind in {'button','link','input','menu item','icon'} and probability(answer.get('probabilities',{}).get(kind,0))>=self.jev.min_presence:
                label=targets[key].removeprefix('the interactive control identified by: ')
                refined[key]=f'The {kind} labeled or described as: {label}'
        if refined==targets:return values,points
        with self.refinement_log.open('a') as f:f.write(json.dumps({'step':kw.get('step'),'original':targets,'scores':scores,'type_answers':answers,'refined':refined})+'\n')
        # Same evidence thresholds, same current frame and checks. Exactly one retry.
        return self.original_assess(image,checks,refined,**kw)
