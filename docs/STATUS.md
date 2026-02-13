```pm-status
milestone: M3
state: complete
headSha: 432644561751c4f13faac050f857ac75929e37f9
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21974819260
updatedAtUtc: 2026-02-13T05:24:07Z
nextStep: Start M4 by adding `book/modern-sbi.qmd` (chapter outline) and linking it from `book/_quarto.yml`.
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.
- **M2 complete**: `book/classical-baselines.qmd` includes MH + HMC baselines (plus optional NUTS via BlackJAX), PPC overlays, and lightweight calibration/coverage smoke tests.

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
