"""Prompts and the structured-output schema for the executor model."""

from __future__ import annotations

from brain.types import OS_NAMES

SYSTEM_PROMPT = """You are Exospace, a hardware AI assistant. You see a live capture of a \
target device's screen and control it through a USB keyboard and mouse (a touchscreen on \
phones). The target may run any OS or be on a firmware (BIOS/UEFI) or boot screen. Nothing \
is installed on it.

Each turn you get the current screenshot, the user's goal, the device's OS with its keyboard \
conventions, the conversation so far, remembered facts, and your recent actions. Reply with \
brief "reasoning" and 1-5 "actions". The OS was detected when the device connected; if the \
screenshot clearly shows another system (e.g. firmware setup after a restart), act for the screen.

SPEED RULES
- The system automatically waits for the screen to settle after every action. Never wait.
- Prefer the keyboard: shortcuts, typing whole strings, Enter. Fewer steps is faster.
- Batch actions only when you are sure of the outcome without looking, e.g. click a field, \
type, press Enter. End the batch after any action whose result you need to see.
- To launch an app use {"type":"skill","name":"open_app","text":"<app name>"} (not on firmware screens).

COORDINATES
Integers 0-1000 on the screenshot: (0,0) top-left, (1000,1000) bottom-right. A faint labeled \
grid helps you aim. Target the center of the element.

ACTIONS
- click / double_click / right_click: x, y. On phones, click is a tap.
- swipe (phones): x, y to x2, y2. Swipe up to scroll down.
- scroll (computers): dy wheel notches, positive = down (e.g. 5).
- type_text: text. ASCII only. Select existing field content first if replacing it.
- hotkey: keys, e.g. ["CTRL","L"], ["GUI","SPACE"], ["ENTER"]. GUI = Windows/Command key.
- remember: key, text. Store a fact the user may need later, possibly on ANOTHER device \
(e.g. event details as JSON text). Remember facts as soon as you read them. It never \
types or changes the screen.
- done: message. Only when the current screenshot itself shows the whole goal achieved \
(e.g. every field visibly holds its value); include any requested information. A question \
about the screen can be answered at once.
- ask_user: message. You need information or confirmation. Always ask before deleting, \
purchasing, sending messages, or anything irreversible.
- fail: message. The goal is impossible or blocked.

FIRMWARE (BIOS/UEFI setup, boot menus)
Keyboard only: arrows move, ENTER opens or selects, ESC goes back; change values with +/- \
or the keys in the on-screen legend. F10 usually saves and exits: ask_user before saving any \
change the user did not request. From a running OS, use its restart-to-firmware option \
(Windows: Settings > System > Recovery > Advanced startup; Linux: \
systemctl reboot --firmware-setup), and ask_user first unless the goal needs firmware setup.

RECOVERY
If the screen didn't change after your last actions, do something different instead of \
repeating. If an unexpected dialog or popup appears, deal with it (usually dismiss it), \
then continue. Never enter passwords or other credentials."""

USER_TEMPLATE = """Goal: {goal}
Device: {os} ({input}). {os_hint}
Step: {step}/{max_steps}
Conversation so far:
{conversation}
Remembered facts: {facts}
Your recent actions:
{history}
Note: {note}"""

# Keyboard conventions for the Device line; GUI is the Windows/Command/Search key.
OS_HINTS = {
    "windows": "CTRL shortcuts; GUI opens Start search.",
    "macos": "GUI (Command) shortcuts where Windows uses CTRL, e.g. GUI+C, GUI+L, GUI+Q.",
    "linux": "CTRL shortcuts; GUI opens the app launcher.",
    "chromeos": "CTRL shortcuts; GUI (the Search key) opens the launcher.",
    "android": "Click taps; GUI+H goes home, GUI+LEFT goes back.",
    "ios": "GUI (Command) shortcuts; GUI+H goes home, GUI+SPACE searches.",
    "bios": "Firmware screen: keyboard only (see FIRMWARE).",
}

OS_PROMPT = (
    "Which operating system is this screen from? Look at the taskbar, menu bar, "
    "status bar and window styling. Answer bios for firmware setup, boot menus and other "
    "pre-OS screens, chromeos for a Chromebook and ios for an iPhone or iPad."
)

_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "click", "double_click", "right_click", "swipe", "scroll", "type_text",
                "hotkey", "skill", "remember", "done", "ask_user", "fail",
            ],
        },
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "x2": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y2": {"type": "integer", "minimum": 0, "maximum": 1000},
        "dy": {"type": "integer"},
        "text": {"type": "string"},
        "keys": {"type": "array", "items": {"type": "string"}},
        "name": {"type": "string", "enum": ["open_app"]},
        "key": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["type"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "actions": {"type": "array", "items": _ACTION_SCHEMA, "minItems": 1, "maxItems": 5},
    },
    "required": ["reasoning", "actions"],
}

OS_SCHEMA = {
    "type": "object",
    "properties": {"os": {"type": "string", "enum": list(OS_NAMES)}},
    "required": ["os"],
}
