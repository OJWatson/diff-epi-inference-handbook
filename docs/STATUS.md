```pm-status
milestone: M0
state: running
headSha: c3c3d698ddf10824808186479e0c04e3b10389cb
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21922093223
updatedAtUtc: 2026-02-11T20:36:40Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
