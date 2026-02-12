```pm-status
milestone: M1
state: running
headSha: e427c5535a693ac085c0e737308ffdf058a5d0d3
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21934221294
updatedAtUtc: 2026-02-12T05:02:26Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Next: consider a more realistic reporting process (weekday effects, censoring, time-varying rho) and start scaffolding the stochastic SEIR variant.
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
