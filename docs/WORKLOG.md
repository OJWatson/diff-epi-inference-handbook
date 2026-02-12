# Work log

Short, append-only notes intended to preserve continuity across short-lived builder runs.

Guidelines:
- One dated bullet per meaningful action.
- Record failures (CI link, error snippet) and the next concrete step.
- Keep this file small.

- 2026-02-12: Enabled GitHub Pages publishing on every push to `main` (kept tag publishing too); ran `ruff check .` + `pytest` locally (all pass).
- 2026-02-12: Switched Pages workflow back to tag-only publishing (removed `push` to `main` trigger); ran `ruff check .` + `pytest` (all pass).
- 2026-02-12: Verified M1 DoD in `docs/STATUS.md` (added checklist; marked M1 complete; set next step toward M2).
- 2026-02-12: Started M2 (state=running); added `book/classical-baselines.qmd` chapter outline and linked it in `book/_quarto.yml`; ran `ruff check .` + `pytest` (all pass).
- 2026-02-12: Implemented minimal random-walk Metropolis–Hastings sampler (`src/diff_epi_inference/mcmc/mh.py`) + standard-normal smoke test; ran `ruff check .` + `pytest` (all pass).
- 2026-02-12: Added a minimal MH usage demo to `book/classical-baselines.qmd` (1D standard normal; reports acceptance rate/mean/std); ran `ruff check .` + `pytest` (all pass) and pushed.
