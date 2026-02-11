```pm-status
milestone: M1
state: running
headSha: aa3b0b23203342276fcc495c44764865b1b3593f
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21922989159
updatedAtUtc: 2026-02-11T21:06:05Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Define the running-example v1 scope (SEIR + reporting / observation model).
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
