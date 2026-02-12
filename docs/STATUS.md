```pm-status
milestone: M2
state: running
headSha: f3a575ece7a936ab5785660924d3e43a66cf8989
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions
updatedAtUtc: 2026-02-12T09:58:45Z
nextStep: Add a minimal **posterior predictive check (PPC)** to the `beta`-only MH/HMC demos in `book/classical-baselines.qmd` (simulate trajectories from posterior draws and overlay against `y_obs`).
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## M1 DoD checklist (verified)
- ✅ Differentiable ODE SEIR + observation model.
- ✅ Non-differentiable stochastic variant paired to the same observation model.
- ✅ Standard dataset format + plotting utils + tests.

## Completed work (M1)
- ✅ Tiny paired example added: deterministic + stochastic SEIR feed the same observation model and return `TimeSeriesDataset` objects.
- ✅ Added `scripts/make_synthetic_dataset.py` to write paired synthetic SEIR datasets to disk via `save_timeseries_npz`; documented `.npz` layout in `docs/DATASETS.md`.
  - ✅ Repo hygiene: ignore generated datasets under `data/` (keep `data/README.md` committed).
- ✅ GitHub Pages decision: add a Pages publish workflow (`.github/workflows/pages.yml`) that deploys the rendered HTML to `gh-pages`.
  - ✅ Auto-publish mode: enabled on version tags (`v*`) and added `docs/PAGES.md` with setup notes.
  - ✅ Decision: keep publishing **tag-only** (removed `push` to `main` trigger).
- ✅ Reproducibility pinning: added `constraints-dev.txt` and use it in CI installs.
