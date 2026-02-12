```pm-status
milestone: M1
state: running
headSha: 2fac0e4f0bad657ccaf6bab8d024137a08e31774
ciRunUrl: https://github.com/OJWatson/diff-epi-inference-handbook/actions/runs/21936176632
updatedAtUtc: 2026-02-12T06:34:00Z
```

## Notes
- **M0 complete**: CI builds (lint+tests, HTML render, optional PDF render) are green; running-example executes; build docs exist; Quarto version is pinned in CI.

## Next steps (M1)
- Next: wire the paired datasets into a minimal `data/` layout (or add a `make_synthetic_dataset.py` script) and document the format in a short chapter stub.
- Decide whether to publish the rendered site via GitHub Pages (deferred to M1).
- Consider further reproducibility pinning (e.g. a lockfile for Python deps).
