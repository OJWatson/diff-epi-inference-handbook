```pm-status
milestone: M1
state: running
headSha: c3c3d698ddf10824808186479e0c04e3b10389cb
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21922093223
updatedAtUtc: 2026-02-11T20:36:40Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Define the running-example v1 scope (SEIR + reporting / observation model).
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
