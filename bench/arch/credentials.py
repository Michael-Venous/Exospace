"""The credential guard: the agent never types a password, PIN, card number or code (issue #27).

A prompt rule alone can be ignored, so the executor checks every input that could type characters (`type`, and a
`key` of letters, digits or SPACE) against the field it goes into. The planner's own words name that field, so a
field it names without a listed word is not caught; the prompt rule still covers it. Usernames and email
addresses are not credentials: typing them is allowed, but a label that also names a secret ("email verification
code", "username or PIN") is refused.
"""
from __future__ import annotations

import re

from bench.arch.confirm import CLICKS
from brain.keys import MODIFIERS, canonical_combo

# Fields that hold a secret. "PIN" counts only in capitals or next to a word that makes it one ("pin code"),
# so "Pin tab", "pinned" and "Pinterest" pass.
SECRET = re.compile(r'\b(pass ?words?|passcodes?|passphrases?|pwd|security codes?|cvv2?|cvc|csc|card ?numbers?|'
                    r'credit ?cards?|debit ?cards?|one[- ]time (?:code|password|passcode|pin)|otp|2fa|mfa|two[- ]factor|'
                    r'verification codes?|authentication codes?|auth codes?|authenticator codes?|secrets?|'
                    r'pin ?(?:code|number|field|box|pad|entry)|(?:enter|your|sim|device) pin)\b', re.IGNORECASE)
CAPITAL_PIN = re.compile(r'\bPIN\b')
MOVES_FOCUS = frozenset({'TAB', 'ENTER'})  # a key with either lands the caret somewhere the planner never named
SHORTCUT_MODIFIERS = frozenset({'CTRL', 'GUI'})  # with these, letters are a shortcut, not typing

REFUSAL = ('Nothing was typed: that looks like a password, PIN, card number or code field, and you never type '
           'those. Use the device\'s password manager (its autofill shortcut, or click the field and pick the saved '
           'password), or a passkey, Windows Hello or Touch ID prompt; if none works, '
           '{"ask":"Please type your password, then say done."}')
NAME_THE_FIELD = ('Nothing was typed: the caret may have moved since you last clicked a field (Tab, Enter, a '
                  'selection or a drag). Send the text again with "into":"<the field>" so it goes where you mean.')
ONE_FIELD = ('Nothing was typed: a Tab or line break inside the text moves to another field. Type into one field '
             'at a time with "into".')

Field = str | None  # the field the caret is in: its label, "" when nothing was named yet, None when unknown


def is_secret_field(label: str) -> bool:
    """Whether a field the planner named holds a secret ("the Password box", "CVV", "Enter PIN")."""
    return bool(SECRET.search(label) or CAPITAL_PIN.search(label))


def focused_field(history: list[tuple[dict, str]]) -> Field:
    """Which field the caret is in, from this request's inputs so far as (action, result), oldest first."""
    return _scan(history)[0]


def _scan(history: list[tuple[dict, str]]) -> tuple[Field, int]:
    """The focused field, and how many character keys (see _types_characters) ran since a field was last named.

    A click, double_click or right_click that ran ("done") puts the caret in the field it named, as does a `type`
    with "into". Anything that may have moved it elsewhere makes it unknown (None), whatever its result, since a
    failed or held input may still have landed: Tab or Enter, a typed Tab or line break, a triple_click, a drag, a
    click that didn't resolve, and an "into" that didn't finish. Other inputs (scroll, hover, shortcuts) leave it.
    At the start of a request nothing is named (""): typing at the caret, as dictation does, is judged by nothing."""
    field: Field = ''
    run = 0
    for action, result in history:
        kind = next((k for k in CLICKS if k in action), None)
        if kind == 'triple_click' or 'drag' in action:
            field = None
        elif kind is not None:
            field, run = (str(action[kind]), 0) if result == 'done' else (None, run)
        elif 'type' in action:
            if action.get('into'):
                field, run = (str(action['into']), 0) if result == 'done' else (None, run)
            if _leaves_field(str(action['type'])):
                field = None
        elif 'key' in action:
            keys = _keys(action['key'])
            if MOVES_FOCUS & set(keys):
                field = None
            elif _types_characters(keys) and result == 'done':
                run += _characters(keys)
    return field, run


def refusal(action: dict, history: list[tuple[dict, str]]) -> str | None:
    """Why this input must not be sent (the error for the planner), or None to send it. `history` is as in _scan.

    A `type` is judged on its "into", else on the focused field; an unknown field asks for "into", and so does a
    Tab or line break before the text's last character (a final one, such as ENTER after a search, is fine).
    A key that types a letter, digit or SPACE is refused in a secret field, and asks for "into" where the caret may
    have moved somewhere unknown (after Tab, Enter, a drag...). Before any field is named, one such character runs
    (YouTube's "k", Gmail's "c" are shortcuts), counted per character, not per action, so ["A", "B"] is two; a
    second asks for "into", or a password could go in key by key. The trade-off: two shortcut keys in a row before
    anything is clicked need a click first. CTRL+V or GUI+V in a secret field is allowed: what it pastes is on the clipboard the
    user's password manager filled, not text the agent read."""
    if 'type' in action:
        field = str(action['into']) if action.get('into') else focused_field(history)
        if field is None:
            return NAME_THE_FIELD
        if is_secret_field(field):
            return REFUSAL
        return ONE_FIELD if _leaves_midway(str(action['type'])) else None
    if 'key' in action and _types_characters(_keys(action['key'])):
        field, run = _scan(history)
        if field and is_secret_field(field):
            return REFUSAL
        if field is None or (field == '' and run + _characters(_keys(action['key'])) > 1):
            return NAME_THE_FIELD
        return None
    return None


def _keys(value) -> list[str]:
    """A key action's keys in the HID's own spelling; none when the HID would refuse them (nothing is sent)."""
    try:
        return canonical_combo([str(k) for k in (value if isinstance(value, list) else [value])])
    except ValueError:
        return []


def _types_characters(keys: list[str]) -> bool:
    """A letter, digit or SPACE, alone or with SHIFT or ALT: it types into the field instead of running a shortcut."""
    return not SHORTCUT_MODIFIERS & set(keys) and any(len(k) == 1 or k == 'SPACE' for k in keys if k not in MODIFIERS)


def _characters(keys: list[str]) -> int:
    """How many characters a key action types (each letter, digit or SPACE in it)."""
    return sum(1 for k in keys if k not in MODIFIERS and (len(k) == 1 or k == 'SPACE'))


def _leaves_field(text: str) -> bool:
    """A typed Tab or line break (any str.splitlines separator: the HID types them as TAB or ENTER) moves the
    caret out of the field."""
    return bool(text) and ('\t' in text or text.splitlines() != [text])


def _leaves_midway(text: str) -> bool:
    """A Tab or line break before the text's end: the rest would go into another field."""
    body = text[:-2] if text.endswith('\r\n') else text[:-1] if _leaves_field(text[-1:]) else text
    return _leaves_field(body)
