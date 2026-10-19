# Skills

**Owner:** Agent

Deterministic macros, invoked as `{"type":"skill","name":...,"args":{...}}` (PLAN.md §5.3). A skill saves model
calls where the OS gives a reliable keyboard path; anything else stays on screen actions.

| Skill | What it does |
|---|---|
| `open_app` | Opens the desktop launcher search (GUI, or GUI+SPACE on macOS), types the name, presses Enter; on phones and unknown targets the model uses the screen instead |
| `hover` | Moves the pointer to `x`,`y` without a button (mouse targets only) |
| `select` | Triple-clicks `x`,`y`, selecting the line or paragraph there, with no settle between the clicks (mouse targets only) |
| `wait` | Blocks until the screen changes, at most 10 s; sends no input (every executor; race plans it as its own stage) |
