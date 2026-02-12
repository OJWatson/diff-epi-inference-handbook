```pm-status
milestone: M1
state: running
headSha: d1e3dbdae8adfe3bfd2bde30a2f79da9f4401eff
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21936886101
updatedAtUtc: 2026-02-12T07:05:22Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- ✅ Tiny paired example added: deterministic + stochastic SEIR feed the same observation model and return `TimeSeriesDataset` objects.
- ✅ Added `scripts/make_synthetic_dataset.py` to write paired synthetic SEIR datasets to disk via `save_timeseries_npz`; documented `.npz` layout in `docs/DATASETS.md`.
  - ✅ Repo hygiene: ignore generated datasets under `data/` (keep `data/README.md` committed).
- ✅ GitHub Pages decision: add a Pages publish workflow (`.github/workflows/pages.yml`) that deploys the rendered HTML to `gh-pages`.
  - ✅ Auto-publish mode: enabled on version tags (`v*`) and added `docs/PAGES.md` with setup notes.
  - Next: decide whether to enable publishing on every `push` to `main`.
- ✅ Reproducibility pinning: added `constraints-dev.txt` and use it in CI installs.
- Next: decide whether to enable Pages publishing on every `push` to `main`, or keep tag-only publishing.
- Consider further reproducibility pinning (e.g. a full lockfile for Python deps).
