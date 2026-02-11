```pm-status
milestone: M0
state: running
headSha: 084561af779cce2dea0c1c4c392a543aa94e14ea
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21921433320
updatedAtUtc: 2026-02-11T20:16:10Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
