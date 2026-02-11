```pm-status
milestone: M0
state: running
headSha: f0038f64f04fd492f4b918b5cbab6b88be9240dd
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21921381985
updatedAtUtc: 2026-02-11T20:14:28Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
