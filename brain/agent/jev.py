"""TypeSafe image decisions; fail closed on absent or ambiguous targets."""
from __future__ import annotations

import base64
import json
import math
import time
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np

from brain.cancel import CancelToken


ATTEMPTS = 2  # one retry: a single timeout ended 3 of 14 live runs
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAY_S = 0.5


class JevError(RuntimeError):
    """Sanitized provider failure (never contains credentials or image payloads)."""


def probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Expected a numeric probability')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Probability outside [0,1]')
    return float(value)


def prepare_image(jpeg: bytes) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Invalid image')
    h, w = image.shape[:2]
    scale = min(1.0, 1568 / max(h, w))
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    pw, ph = max(w, 280), max(h, 280)
    image = cv2.copyMakeBorder(image, 0, ph-h, 0, pw-w, cv2.BORDER_CONSTANT, value=(255,255,255))
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('Image encoding failed')
    return encoded.tobytes(), (w, h), (pw, ph)


COMPLETION_QUESTION = (
    'Considering the `executed_steps` (what was actually done) together with the current `screen`, '
    "has the user's ENTIRE `goal` been accomplished? Every part of the goal must be covered by an "
    'executed step or be visibly true on the screen. A visible link, button or preview is not evidence '
    'that it was used; text typed but not submitted is not a completed search or send.')
# With the planner's expected evidence, so Jev knows what to look for when it cannot work it out
# itself (a calculator result scored 0.02 without it, 0.7 with it).
EVIDENCE_SUFFICIENT = ('Would a screen showing `expected_evidence` prove that the user\'s ENTIRE `goal` is done, '
                       'not just a first step towards it?')
COMPLETION_HINTED = COMPLETION_QUESTION + (
    ' `expected_evidence` is what the planner expects the screen to show once the goal is done: use it '
    'to know what to look for, but judge the goal itself against the steps and the screen, not the claim.')


SIDE_EFFECT_GATE = 0.7
# Recovery once pressed Alt+Tab, landed on another app and closed it with Alt+F4 (laptop suite).
SIDE_EFFECTS = (
    'Would carrying out the proposed `actions` on this `screen` close, quit, minimize or discard any window, '
    'tab, app, file or unsaved work that the user\'s `request` does not ask to close or discard? Confirming, '
    'saving or closing a dialog, menu or popup as part of doing the request, or dismissing a splash screen, '
    'welcome screen or notification, is not such a side effect.')
IRREVERSIBLE_KINDS = ('do something irreversible or high-impact: '
                      'sending or posting a message, deleting or emptying data, purchasing or paying, submitting a '
                      'form, formatting, changing and saving firmware/BIOS settings, or a public action on the user\'s '
                      'account (starring, liking, following, subscribing, voting, sharing)? Opening, navigating, '
                      'selecting, typing a draft, adding to or removing from a cart, and creating something new are not.')
STEP_IRREVERSIBLE = 'Would carrying out the proposed `actions` on this `screen` ' + IRREVERSIBLE_KINDS


class JevVision:
    def __init__(self, key: str, cancel: CancelToken, log_path: Path,
                 model: str = 'jev-1.14.0', timeout_s: float = 20,
                 min_presence: float = .85, min_confidence: float = .6) -> None:
        if not key:
            raise RuntimeError('TYPESAFE_API_KEY is not set')
        self.cancel, self.log_path, self.model = cancel, log_path, model
        self.min_presence = probability(min_presence)
        self.min_confidence = probability(min_confidence)
        self.client = httpx.Client(timeout=timeout_s, headers={
            'Authorization': f'Bearer {key}', 'TypeSafe-Danger-Zone-Images': 'true'})

    def evaluate(self, jpeg: bytes, questions: dict, *, run_id: str = '', step: int = 0,
                 state: dict | None = None):
        """Ask questions about `screen`; `state` adds named context (e.g. the utterance)."""
        encoded, size, padded = prepare_image(jpeg)
        ref = str(uuid.uuid4())
        body = {'model': self.model, 'state': {'screen': ref, **(state or {})},
                'assets': {ref: {'type':'image','format':'jpg',
                                 'data_base64':base64.b64encode(encoded).decode()}},
                'questions': questions}
        for attempt in range(1, ATTEMPTS + 1):
            started, success, error = time.monotonic(), False, ''
            try:
                self.cancel.check()
                response = self.cancel.run(lambda: self.client.post(
                    'https://api.typesafe.ai/v1/systemone', json=body))
                self.cancel.check()
                if response.status_code != 200:
                    error = f'HTTP {response.status_code}'
                    if response.status_code not in RETRY_STATUS or attempt == ATTEMPTS:
                        raise JevError(f'Jev HTTP {response.status_code}')
                else:
                    answers = response.json().get('answers')
                    if not isinstance(answers, dict) or any(k not in answers for k in questions):
                        raise JevError('Jev returned incomplete answers')
                    success = True
                    return answers, size, padded
            except httpx.TransportError as exc:  # a timeout or dropped connection: one retry
                error = type(exc).__name__
                if attempt == ATTEMPTS:
                    raise JevError(f'Jev transport failure ({error})') from None
            except httpx.HTTPError as exc:
                error = type(exc).__name__
                raise JevError(f'Jev transport failure ({error})') from None
            except BaseException as exc:
                error = error or type(exc).__name__
                raise
            finally:
                self._log(run_id, step, attempt, started, success, error)
            self.cancel.sleep(RETRY_DELAY_S)
        raise AssertionError('unreachable')

    def _log(self, run_id: str, step: int, attempt: int, started: float, success: bool, error: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as stream:
            stream.write(json.dumps({'run_id':run_id,'step':step,'role':'jev',
                'model':self.model,'attempt':attempt,'elapsed_s':time.monotonic()-started,
                'success':success,'error':error})+'\n')

    def locate(self, jpeg: bytes, target: str, *, run_id: str = '', step: int = 0):
        questions = {
            'present': {'type':'noul','instructions':f'Is the following target visible and enabled in `screen`? Target: {target}'},
            'unique': {'type':'noul','instructions':f'Is there exactly one unambiguous visible target matching this description in `screen`? Target: {target}'},
            'location': {'type':'box','instructions':f'Locate the interactive target in `screen`: {target}'},
        }
        answers, (w,h), (pw,ph) = self.evaluate(jpeg, questions, run_id=run_id, step=step)
        present = probability(answers['present'].get('noul'))
        unique = probability(answers['unique'].get('noul'))
        confidence = probability(answers['location'].get('confidence'))
        if min(present,unique) < self.min_presence or confidence < self.min_confidence:
            return None
        box = answers['location'].get('box', {})
        left,right,top,bottom = [probability(box.get(k)) for k in ('left','right','top','bottom')]
        if not left < right or not top < bottom:
            raise ValueError('Invalid Jev box ordering')
        x,y = (left+right)/2*pw/w, (top+bottom)/2*ph/h
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            return None  # padded region must never become a click
        return x,y

    def completion(self, jpeg: bytes, goal: str, executed: list[str], *, evidence: str = '',
                   run_id: str = '', step: int = 0) -> float:
        """Probability the whole goal is done, judged from the executed steps AND the
        screen. History-dependent goals ("the first video") are covered by the steps;
        skipped or unsubmitted steps are caught because they are absent from them.
        `evidence` (the planner's expected end state) only tells Jev what to look for:
        12/12 finished and 0/18 traps accepted, 12/15 and 0/15 held out (bench/arch/verify_study)."""
        state = {'goal': goal, 'executed_steps': executed}
        questions = {'complete': {'type': 'noul', 'instructions': COMPLETION_QUESTION}}
        if evidence:
            # The hint is used only when it would prove the whole goal: a plan whose evidence was just
            # "Blender is open" for "make me a house in blender" passed as done (laptop suite).
            state['expected_evidence'] = evidence
            questions['complete'] = {'type': 'noul', 'instructions': COMPLETION_QUESTION +
                                     ' Ignore `expected_evidence` for this question.'}
            questions['hinted'] = {'type': 'noul', 'instructions': COMPLETION_HINTED}
            questions['sufficient'] = {'type': 'noul', 'instructions': EVIDENCE_SUFFICIENT}
        answers, _, _ = self.evaluate(jpeg, questions, run_id=run_id, step=step, state=state)
        if evidence and probability(answers['sufficient'].get('noul')) >= 0.5:
            return probability(answers['hinted'].get('noul'))
        return probability(answers['complete'].get('noul'))

    def close(self):
        self.client.close()

    def assess(self, jpeg: bytes, checks: dict[str, str], targets: dict[str, str],
               *, run_id: str = '', step: int = 0, request: str = '', proposal=None, irreversible: bool = False):
        """Batch independent state checks and named targets against ONE fresh frame. With `irreversible`,
        values['irreversible'] scores whether the proposal would send, delete, pay or submit."""
        questions = {f'check_{key}': {'type':'noul', 'instructions':
            f'Based only on the visible `screen`, is this condition true? {condition}'}
            for key, condition in checks.items()}
        if proposal:
            questions['side_effects'] = {'type':'noul', 'instructions': SIDE_EFFECTS}
        if proposal and irreversible:
            questions['irreversible'] = {'type':'noul', 'instructions': STEP_IRREVERSIBLE}
        questions['route'] = {'type':'choice', 'instructions': {
            'question':'Judge ONLY the supplied next actions or verification checks, not the complexity of a larger task. Choose execute when this bounded next step is clear and feasible on this screen. Choose ask_flash for uncertainty, unclear intent, missing actions, or an unexpected screen needing a new plan. A verification-only step with checks and no actions is valid. Do not invent steps.',
            'request':request, 'actions':proposal, 'checks':checks, 'targets':targets},
            'criteria':{
                'execute':'The supplied next action or verification checks are clear, sufficient, and supported by the visible screen.',
                'ask_flash':'Need language-model planning, clarification, or recovery; uncertain or ambiguous target/conditions; or no executable proposal is supplied.'}}
        for key, target in targets.items():
            questions.update({
                f'{key}_present': {'type':'noul','instructions':f'Is this target visible and unobscured in `screen`, and enabled if it is a control? A neutral region may be used for pointer movement. Target: {target}'},
                f'{key}_unique': {'type':'noul','instructions':f'Is exactly one visible target in `screen` unambiguously identified by: {target}?'},
                f'{key}_clear': {'type':'noul','instructions':f'Does this instruction clearly identify the intended interactive target in `screen` without guessing missing user intent or resolving conflicting directions? Answer no if the description is vague or unclear. Instruction: {target}'},
                f'{key}_box': {'type':'box','instructions':f'The exact described target in `screen`. For an input field, locate the editable box itself, excluding its label. Target: {target}'},
            })
        state={'request':request,'actions':json.dumps(proposal)} if proposal else None
        answers, (w,h), (pw,ph) = self.evaluate(jpeg,questions,run_id=run_id,step=step,state=state)
        values={key:probability(answers[f'check_{key}'].get('noul')) for key in checks}
        route=answers['route']
        # Choice confidence and Box confidence have different semantics. Honor
        # the explicit routing choice; the independent evidence gates still apply.
        probability(route.get('confidence'))
        values['route_execute']=float(route.get('choice')=='execute')
        if proposal:  # a stage that closes or discards something nobody asked for never runs
            # Only a clear yes blocks: borderline verdicts vetoed dismissing a splash screen (0.5-0.7).
            values['safe']=float(probability(answers['side_effects'].get('noul'))<SIDE_EFFECT_GATE)
        if 'irreversible' in questions:
            values['irreversible']=probability(answers['irreversible'].get('noul'))
        self.last_scores={'route':route.get('choice'),'route_confidence':route.get('confidence')}
        points={}
        self.last_points={}  # every target's box center, before the presence/confidence gates
        for key in targets:
            a=answers[f'{key}_box']
            self.last_scores[key]={'presence':answers[f'{key}_present'].get('noul'),
                                  'unique':answers[f'{key}_unique'].get('noul'),
                                  'clear':answers[f'{key}_clear'].get('noul'),
                                  'box_confidence':a.get('confidence')}
            # Intent clarity is a routing decision, not another copy of the
            # localization probability cutoff. Keep its score for diagnostics.
            probability(answers[f'{key}_clear'].get('noul'))
            gated=min(probability(answers[f'{key}_present'].get('noul')),
                      probability(answers[f'{key}_unique'].get('noul'))) < self.min_presence \
                or probability(a.get('confidence')) < self.min_confidence
            box=a.get('box') or {}
            try:
                left,right,top,bottom=[probability(box.get(k)) for k in ('left','right','top','bottom')]
                ordered=left<right and top<bottom
            except ValueError:
                ordered=False
            # A malformed box means "not located": it ended a whole run when it raised (laptop suite).
            x,y=((left+right)/2*pw/w,(top+bottom)/2*ph/h) if ordered else (-1,-1)
            center=(x,y) if 0<=x<=1 and 0<=y<=1 else None
            self.last_points[key]=center
            points[key]=None if gated else center
        return values,points
