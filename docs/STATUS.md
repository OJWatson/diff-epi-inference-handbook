```pm-status
milestone: M1
state: running
headSha: 4e6b4f7f8c21102bddf8753e4b4bab99f278c7fb
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21934886995
updatedAtUtc: 2026-02-12T05:34:47Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Add a tiny paired example: simulate *both* deterministic and stochastic SEIR, then feed each through the same observation model and store as `TimeSeriesDataset`.
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
