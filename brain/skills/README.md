# Skills

**Owner:** Agent

Deterministic macros, invoked as `{"type":"skill","name":...,"args":{...}}` (PLAN.md §5.3). A skill saves model
calls where the OS gives a reliable keyboard path; anything else stays on screen actions.

| Skill | What it does |
|---|---|
| `open_app` | Opens the OS search (GUI, or GUI+SPACE on macOS/iOS), types the name, presses Enter; unavailable on firmware and unknown targets |
| `hover` | Moves the pointer to `x`,`y` without a button (mouse targets only) |
| `wait` | Blocks until the screen changes, at most 10 s; sends no input (direct executor only) |
| `boot_keys` | Arms 1–3 firmware keys (ESC, DELETE, F1–F12) that the controller taps through the next restart (experimental) |
