```pm-status
milestone: M1
state: running
headSha: b06005048e8499df41e7bd95f37efd387577eed4
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21923439390
updatedAtUtc: 2026-02-11T21:20:21Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Extend the observation model beyond the minimal `rho * incidence` (e.g. delay, under-reporting + overdispersion).
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
