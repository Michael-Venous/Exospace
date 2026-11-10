"""The credential guard: the agent never types a password, PIN, card number or code (issue #27).

A prompt rule alone can be ignored, so the executor checks every `type` against the field it goes into: the
`into` field, or else the last thing this request clicked. The planner's own words name that field, so a field
it names without a listed word is not caught; the prompt rule still covers it. Usernames and email addresses are
not credentials: typing them is allowed.
"""
from __future__ import annotations

import re

from bench.arch.confirm import CLICKS

# Fields that hold a secret. "PIN" counts only in capitals or next to a word that makes it one ("pin code"),
# so "Pin tab", "pinned" and "Pinterest" pass.
SECRET = re.compile(r'\b(pass ?words?|passcodes?|passphrases?|pwd|security codes?|cvv2?|cvc|csc|card ?numbers?|'
                    r'credit ?cards?|debit ?cards?|one[- ]time (?:code|password|passcode|pin)|otp|2fa|mfa|two[- ]factor|'
                    r'verification codes?|authentication codes?|auth codes?|authenticator codes?|secrets?|'
                    r'pin ?(?:code|number|field|box|pad|entry)|(?:enter|your|sim|device) pin)\b', re.IGNORECASE)
CAPITAL_PIN = re.compile(r'\bPIN\b')
# Fields that never do, even next to a secret's name ("the email box on the Reset password page").
NOT_SECRET = re.compile(r'\b(e-?mail|user ?name|user ?id|phone number|search)\b', re.IGNORECASE)

REFUSAL = ('Nothing was typed: that looks like a password, PIN, card number or code field, and you never type '
           'those. Use the device\'s password manager (its autofill shortcut, or click the field and pick the saved '
           'password), or a passkey, Windows Hello or Touch ID prompt; if none works, '
           '{"ask":"Please type your password, then say done."}')


def is_secret_field(label: str) -> bool:
    """Whether a field the planner named holds a secret ("the Password box", "CVV", "Enter PIN")."""
    if NOT_SECRET.search(label):
        return False
    return bool(SECRET.search(label) or CAPITAL_PIN.search(label))


def typing_target(action: dict, sent: list[dict]) -> str:
    """The field a `type` goes into: its `into`, else the last click among this request's sent inputs ("" if none)."""
    if action.get('into'):
        return str(action['into'])
    for earlier in reversed(sent):
        kind = next((k for k in CLICKS if k in earlier), None)
        if kind is not None:
            return str(earlier[kind])
    return ''


def refuses(action: dict, sent: list[dict]) -> bool:
    """Whether this input types into a secret field. Only `type` is judged: keys such as a password manager's
    autofill shortcut and clicks pass."""
    return 'type' in action and is_secret_field(typing_target(action, sent))
