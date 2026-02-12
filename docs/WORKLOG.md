# Work log

Short, append-only notes intended to preserve continuity across short-lived builder runs.

Guidelines:
- One dated bullet per meaningful action.
- Record failures (CI link, error snippet) and the next concrete step.
- Keep this file small.

- 2026-02-12: Enabled GitHub Pages publishing on every push to `main` (kept tag publishing too); ran `ruff check .` + `pytest` locally (all pass).
- 2026-02-12: Switched Pages workflow back to tag-only publishing (removed `push` to `main` trigger); ran `ruff check .` + `pytest` (all pass).
