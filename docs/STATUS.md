```pm-status
milestone: M1
state: running
headSha: b3e2747eac58c7bfb095b827de04b961cba250cd
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21924305140
updatedAtUtc: 2026-02-11T21:48:13Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Next: consider a more realistic reporting process (weekday effects, censoring, time-varying rho) and a dataset format.
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
