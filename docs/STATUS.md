```pm-status
milestone: M1
state: running
headSha: 5c6d4ee52903449dd0010ae7bf0ed978d7ea4974
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21936176632
updatedAtUtc: 2026-02-12T06:53:30Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- ✅ Tiny paired example added: deterministic + stochastic SEIR feed the same observation model and return `TimeSeriesDataset` objects.
- Next: add a tiny `make_synthetic_dataset.py` that writes the paired datasets to disk via `save_timeseries_npz`, and document the on-disk layout.
- ✅ GitHub Pages decision: add a manual Pages publish workflow (`.github/workflows/pages.yml`) that deploys the rendered HTML to `gh-pages`.
  - Next: decide whether to enable automatic publishing on `push` to `main` (and configure repo Pages settings).
- ✅ Reproducibility pinning: added `constraints-dev.txt` and use it in CI installs.
- Consider further reproducibility pinning (e.g. a full lockfile for Python deps).
