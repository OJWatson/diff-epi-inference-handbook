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
- 2026-02-12: Added MH calibration smoke test to `book/classical-baselines.qmd` (20 synthetic datasets across a `beta_true` grid; reports approximate 90% interval coverage); ran `ruff check .` + `pytest` (all pass) and pushed.
- 2026-02-12: Added matching HMC calibration smoke test to `book/classical-baselines.qmd` (same ~20 datasets; reports approximate 90% interval coverage); ran `ruff check .` + `pytest` (all pass) and pushed.
- 2026-02-12: Added optional BlackJAX NUTS wrapper (`src/diff_epi_inference/mcmc/nuts_blackjax.py`) + skipped smoke test + tiny guarded demo snippet; ran `ruff check .` + `pytest` (all pass; NUTS test skipped without deps) and pushed.
- 2026-02-12: Added an optional BlackJAX NUTS `beta`-only SEIR posterior demo (JAX reimplementation of the minimal model) + PPC in `book/classical-baselines.qmd`; ran `ruff check .` + `pytest` (all pass; NUTS optional) and pushed.
- 2026-02-12: Added optional BlackJAX NUTS SEIR `beta`-only calibration/coverage smoke test (`tests/test_blackjax_nuts_seir_calibration_optional.py`); ran `ruff check .` + `pytest` (all pass; optional tests skipped without deps).
- 2026-02-12: Fixed CI ruff failure (UP027) in optional BlackJAX calibration test; ran `ruff check .` + `pytest`, pushed; CI: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21945763287
- 2026-02-12: Added a short note in `book/classical-baselines.qmd` pointing to the optional BlackJAX NUTS SEIR calibration/coverage smoke test; ran `ruff check .` + `pytest` (all pass; optional tests skipped without deps) and pushed.
- 2026-02-12: Refactored the duplicated JAX SEIR beta-only log-posterior (book + optional BlackJAX calibration test) into `src/diff_epi_inference/models/seir_jax_beta_only.py`; ran `ruff` + `pytest` (pass) and will push.

- 2026-02-12: Extracted NumPy SEIR beta-only log posterior into  and updated  to use it; ran  + ============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/kana/git/diff-epi-inference-handbook
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.12.1
collected 17 items

tests/test_blackjax_nuts_optional.py s                                   [  5- 2026-02-12: Extracted NumPy SEIR beta-only log posterior into `make_log_post_logbeta_numpy` and updated `book/classical-baselines.qmd` to use it; ran `python3 -m ruff check .` + `python3 -m pytest` (all pass) and pushed.
