"""Study 5: which completion check accepts finished tasks and rejects traps?

Checkers (all Jev, one call each):
  wording   runtime: original goal + proposed evidence judged from the final screen (gate 0.85)
  concrete  the planner's concrete completion text judged from the screen (gate 0.5)
  ledger    goal + list of actually executed steps + screen (gate 0.5)
  both      concrete AND ledger
Traps include a skipped step whose weak completion text looks satisfied (Codex's
false positive), an unsubmitted search, an unsent reply, and a click with no effect.
"""

from __future__ import annotations

import base64
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.arch.workstation import HTML as WORKSTATION, at
from brain.agent.jev import JevVision, prepare_image
from brain.cancel import CancelToken
from brain.config import Config
from brain.sequence_benchmark import CASES as SEQ

OUT = Path("work/arch/verify")
CATALOG_GOAL = "Open the catalog, then click the first result. Verify that its guide opened."
REPLY = "I'll have the report to you by 5pm."

# name -> (html, js to run, viewport)
SCREENS = {
    "catalog_open": (SEQ["two_clicks"][0], "document.getElementById('start').hidden=true;"
                     "document.getElementById('results').hidden=false"),
    "guide_opened": (SEQ["two_clicks"][0], "document.getElementById('start').hidden=true;"
                     "document.getElementById('status').textContent='Alpha guide opened successfully'"),
    "watch_lofi": (WORKSTATION, f"load({json.dumps(at('watch:lofi', ['inbox', 'youtube'], watched='lofi'))})"),
    "results_lofi": (WORKSTATION, f"load({json.dumps(at('results:lofi', ['inbox', 'youtube'], query='lofi'))})"),
    "youtube_typed": (WORKSTATION, f"load({json.dumps(at('youtube', ['inbox']))});document.getElementById('q').value='lofi'"),
    "compose_draft": (WORKSTATION, f"load({json.dumps(at('compose', ['inbox', 'email:john'], draft={'to': 'john', 'body': REPLY}))})"),
    "inbox": (WORKSTATION, f"load({json.dumps(at('inbox', []))})"),
}

# (goal, screen, executed steps, planner's completion text, truly done?)
CASES = [
    (CATALOG_GOAL, "catalog_open", ["clicked 'Open catalog'"],
     "The catalog is open and the first result's guide is visible.", False),
    (CATALOG_GOAL, "guide_opened", ["clicked 'Open catalog'", "clicked the first result, 'Alpha guide'"],
     "The status says 'Alpha guide opened successfully'.", True),
    ("play the first video", "watch_lofi", ["clicked the first video card in the grid, 'lofi hip hop radio - beats to relax/study to'"],
     "The lofi hip hop radio video is now playing.", True),
    ("play the pasta video", "watch_lofi", ["clicked the video card 'lofi hip hop radio - beats to relax/study to'"],
     "A video is now playing.", False),
    ("search for lofi", "results_lofi", ["clicked the Search box", "typed 'lofi'", "pressed ENTER"],
     "Search results for 'lofi' are shown.", True),
    ("search for lofi", "youtube_typed", ["clicked the Search box", "typed 'lofi'"],
     "'lofi' is entered in the YouTube search box.", False),
    ("draft a reply to john saying I'll have the report by 5pm, but don't send it yet", "compose_draft",
     ["clicked Reply on John Miller's email", f"typed '{REPLY}'"],
     "A reply to John Miller contains 'I'll have the report to you by 5pm.'", True),
    ("reply to john saying I'll have the report by 5pm and send it", "compose_draft",
     ["clicked Reply on John Miller's email", f"typed '{REPLY}'"],
     "A reply to John Miller with the 5pm message is drafted.", False),
    ("open janet's email", "inbox", ["clicked the Janet Park email in the list"],
     "Janet Park's email is selected.", False),
    ("open janet's email", "inbox", ["clicked the Janet Park email in the list"],
     "Janet Park's email 'Q3 report numbers + notes' is open in the reading pane.", False),
]


def render(screens: dict[str, tuple[str, str]] = SCREENS) -> dict[str, bytes]:
    from playwright.sync_api import sync_playwright

    shots = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        for name, (html, js) in screens.items():
            page.set_content(html)
            page.evaluate(f"() => {{ {js} }}")
            shots[name] = page.screenshot(type="jpeg", quality=90)
        browser.close()
    return shots


def ask(jev: JevVision, jpeg: bytes, extra_state: dict, instructions: str) -> tuple[float, float]:
    encoded, _, _ = prepare_image(jpeg)
    ref = str(uuid.uuid4())
    body = {"model": jev.model, "state": {"screen": ref, **extra_state},
            "assets": {ref: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(encoded).decode()}},
            "questions": {"q": {"type": "noul", "instructions": instructions}}}
    started = time.monotonic()
    response = jev.client.post("https://api.typesafe.ai/v1/systemone", json=body)
    response.raise_for_status()
    return float(response.json()["answers"]["q"]["noul"]), time.monotonic() - started


def check(jev: JevVision, jpeg: bytes, kind: str, goal: str, steps: list[str], completion: str) -> tuple[float, float]:
    if kind == "wording":
        return ask(jev, jpeg, {}, "Based only on the visible `screen`, is this condition true? "
                   "The ENTIRE original user request is visibly complete, including every required action and result. "
                   "An available link/button or preview is not evidence that it was opened or executed. "
                   f"Original user request: {goal}\nProposed completion evidence: {completion}")
    if kind == "concrete":
        return ask(jev, jpeg, {}, f"Based only on the visible `screen`, is this condition true? The screen shows this state: {completion}")
    return ask(jev, jpeg, {"goal": goal, "executed_steps": steps},
               "Considering the `executed_steps` (what was actually done) together with the current `screen`, "
               "has the user's ENTIRE `goal` been accomplished? Every part of the goal must be covered by an "
               "executed step or be visibly true on the screen. A visible link, button or preview is not evidence "
               "that it was used; text typed but not submitted is not a completed search or send.")


GATES = {"wording": 0.85, "concrete": 0.5, "ledger": 0.5}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", action="store_true", help="use the studio held-out set")
    args = parser.parse_args()
    global OUT, CASES
    if args.heldout:
        OUT, CASES = OUT.with_name("verify_heldout"), HELDOUT
    cfg = Config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    shots = render(heldout_screens() if args.heldout else SCREENS)
    for name, jpeg in shots.items():
        (OUT / f"{name}.jpg").write_bytes(jpeg)
    jev = JevVision(cfg.typesafe_api_key, CancelToken(), OUT / "calls.jsonl", cfg.jev_model, 30)
    jobs = [(i, kind, r) for i in range(len(CASES)) for kind in GATES for r in range(3)]

    def run(job):
        i, kind, _ = job
        goal, screen, steps, completion, _ = CASES[i]
        return job, check(jev, shots[screen], kind, goal, steps, completion)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, jobs))
    jev.close()
    scores: dict[tuple[int, str], list[float]] = {}
    latency: dict[str, list[float]] = {k: [] for k in GATES}
    for (i, kind, _), (score, elapsed) in results:
        scores.setdefault((i, kind), []).append(score)
        latency[kind].append(elapsed)
    rows = []
    for i, (goal, screen, steps, completion, done) in enumerate(CASES):
        passed = {k: [s >= GATES[k] for s in scores[(i, k)]] for k in GATES}
        passed["both"] = [a and b for a, b in zip(passed["concrete"], passed["ledger"])]
        rows.append({"goal": goal, "screen": screen, "done": done,
                     "scores": {k: [round(s, 2) for s in scores[(i, k)]] for k in GATES}, "passed": passed})
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    for kind in [*GATES, "both"]:
        true_accept = sum(sum(r["passed"][kind]) for r in rows if r["done"])
        false_accept = sum(sum(r["passed"][kind]) for r in rows if not r["done"])
        n_done = 3 * sum(r["done"] for r in rows)
        n_trap = 3 * sum(not r["done"] for r in rows)
        lat = f" p50 {statistics.median(latency[kind]):.2f}s" if kind in latency else ""
        print(f"{kind:9} accepts finished {true_accept}/{n_done}   accepts traps (false done) {false_accept}/{n_trap}{lat}")
    for r in rows:
        print(("DONE " if r["done"] else "TRAP ") + f"{r['goal'][:55]:55} {r['screen']:14} {r['scores']}")



# --- Held-out set on the studio fixture (nothing tuned on it) -------------------------
def _studio(state: dict, js: str = "") -> tuple[str, str]:
    from bench.arch.studio import HTML as STUDIO

    return STUDIO, f"load({json.dumps(state)});{js}"


def heldout_screens() -> dict[str, tuple[str, str]]:
    from bench.arch.studio import at

    note = "buy keyboard later"
    return {
        "s_taxes_created": _studio(at(folder="taxes", folders=["documents", "downloads", "pictures", "taxes"])),
        "s_taxes_typed": _studio(at(folder="downloads", dialog="new"), "document.getElementById('nf').value='taxes'"),
        "s_note_saved": _studio(at(app="notes", notes=note, saved=note)),
        "s_note_unsaved": _studio(at(app="notes", notes=note, saved="")),
        "s_kb_in_cart": _studio(at(app="shop", product="keyboard", cart=["keyboard"])),
        "s_kb_not_added": _studio(at(app="shop", product="keyboard")),
        "s_pictures": _studio(at(folder="pictures")),
        "s_downloads": _studio(at(folder="downloads")),
        "s_delete_dialog": _studio(at(folder="downloads", dialog="delete")),
    }


HELDOUT = [
    ("make a new folder called taxes", "s_taxes_created", ["clicked '+ New folder'", "typed 'taxes'", "clicked 'Create'"],
     "A folder named 'taxes' appears in the folder list.", True),
    ("make a new folder called taxes", "s_taxes_typed", ["clicked '+ New folder'", "typed 'taxes'"],
     "The name 'taxes' is entered for the new folder.", False),
    ("open notes, write 'buy keyboard later' and save it", "s_note_saved",
     ["clicked 'Notes' in the taskbar", "clicked the note text area", "typed 'buy keyboard later'", "clicked 'Save'"],
     "The note contains 'buy keyboard later'.", True),
    ("open notes, write 'buy keyboard later' and save it", "s_note_unsaved",
     ["clicked 'Notes' in the taskbar", "clicked the note text area", "typed 'buy keyboard later'"],
     "The note contains 'buy keyboard later'.", False),
    ("add the mechanical keyboard to my cart", "s_kb_in_cart", ["clicked 'Mechanical keyboard'", "clicked 'Add to cart'"],
     "The Mechanical keyboard page shows it is in the cart.", True),
    ("add the mechanical keyboard to my cart", "s_kb_not_added", ["clicked 'Mechanical keyboard'"],
     "The Mechanical keyboard product page is shown.", False),
    ("open the pictures folder", "s_pictures", ["clicked 'Pictures' in the sidebar"],
     "The Pictures folder is open.", True),
    ("open the pictures folder", "s_downloads", ["clicked 'Pictures' in the sidebar"],
     "A folder is open in the file manager.", False),
    ("remove the keyboard from my cart", "s_kb_not_added", ["clicked 'Remove from cart'"],
     "The Mechanical keyboard page offers 'Add to cart' and the cart shows 0 items.", True),
    ("delete the downloads folder", "s_delete_dialog", ["clicked 'Delete folder'"],
     "A confirmation to delete the downloads folder is shown.", False),
]


if __name__ == "__main__":
    main()
