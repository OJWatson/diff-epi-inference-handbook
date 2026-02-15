```pm-status
milestone: M6
state: running
headSha: (pending)
ciRunUrl: (see GitHub Actions)
updatedAtUtc: 2026-02-15T14:33:00Z
nextStep: M6.0 — Add two new Quarto chapters: autodiff basics + differentiability axis (with runnable demos).
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.
- **M2 complete**: `book/classical-baselines.qmd` includes MH + HMC baselines (plus optional NUTS via BlackJAX), PPC overlays, and lightweight calibration/coverage smoke tests.
- **M3 complete**: `book/likelihood-free-baselines.qmd` includes ABC rejection + SMC-ABC + synthetic likelihood baselines, each with at least one minimal diagnostic.

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
