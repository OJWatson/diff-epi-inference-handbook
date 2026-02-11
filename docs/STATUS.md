```pm-status
milestone: M0
state: running
headSha: 7f995bf2c3ad205890cb8686f288f5ab974c3660
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21922042579
updatedAtUtc: 2026-02-11T20:35:10Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
