```pm-status
milestone: M2
state: running
headSha: d8fe6518a74f30f14c761aa3073dbb7ddbef2b4c
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions
updatedAtUtc: 2026-02-12T19:27:16Z
nextStep: Confirm the GitHub Actions run for `d8fe651` is green, then (optionally) document in `docs/BUILD.md` that local Quarto downloads create a `.tools/` directory which is intentionally ignored/excluded from lint.
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
