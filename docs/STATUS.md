```pm-status
milestone: M1
state: running
headSha: b2c210a1669b68674674b2a672d867449ff492fe
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21936176632
updatedAtUtc: 2026-02-12T06:46:08Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- ✅ Tiny paired example added: deterministic + stochastic SEIR feed the same observation model and return `TimeSeriesDataset` objects.
- Next: add a tiny `make_synthetic_dataset.py` that writes the paired datasets to disk via `save_timeseries_npz`, and document the on-disk layout.
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- ✅ Reproducibility pinning: added `constraints-dev.txt` and use it in CI installs.
- Consider further reproducibility pinning (e.g. a full lockfile for Python deps).
