```pm-status
milestone: M0
state: running
headSha: 83b36c3238c087f46c9693101917771e023d0e79
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21920296216
updatedAtUtc: 2026-02-11T19:42:26Z
```

## Next steps
- GitHub Pages publishing is deferred to **M1** (M0 focuses on a CI-gated, reproducible local+CI build).
- If/when we add Pages in M1: add a `deploy-pages` workflow that publishes `book/_book/` on pushes to `main`.
- Consider further pinning for reproducibility (e.g. a lockfile for Python deps).
