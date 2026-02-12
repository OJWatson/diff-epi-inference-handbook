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
- 2026-02-12: Added MH baseline demo inferring only `beta` (via `log_beta`) in the SEIR running example likelihood in `book/classical-baselines.qmd`; ran `ruff check .` + `pytest` (all pass) and pushed.
- 2026-02-12: Added a minimal HMC sampler baseline (`src/diff_epi_inference/mcmc/hmc.py`) using finite-difference gradients + smoke test; ran `ruff check .` + `pytest` (all pass) and pushed.
- 2026-02-12: Wired the minimal HMC baseline into the SEIR `beta`-only likelihood demo in `book/classical-baselines.qmd` and documented finite-difference limitations vs autodiff+NUTS; ran `ruff check .` + `pytest` (all pass) and pushed.
- 2026-02-12: Added minimal posterior predictive checks (simulate `y_rep` from posterior `beta` draws; overlay 90% bands vs `y_obs`) for both MH and HMC demos in `book/classical-baselines.qmd`; ran `ruff check .` + `pytest` (all pass) and pushed.
