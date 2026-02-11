```pm-status
milestone: M0
state: running
headSha: 4b86c83dc0deb887fb6f7950340f4c0df0fd43ed
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21921994649
updatedAtUtc: 2026-02-11T20:33:28Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
