# ROADMAP (M0-M8)

This file is the lightweight milestone tracker for the handbook implementation.
It reflects the current `main` branch state and is kept aligned with:

- `docs/diff_epi.pdf`
- `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`
- `README.md`

## Status Snapshot (2026-02-17)

| Milestone | Status | Evidence (chapter / module / tests) |
| --- | --- | --- |
| M0 Bootstrap | Complete | `README.md`, `scripts/ci.sh`, `scripts/test.sh`, `.github/workflows/ci.yml` |
| M1 Running example | Complete | `book/running-example.qmd`, `src/diff_epi_inference/seir.py`, `tests/test_running_example_smoke.py` |
| M2 Classical baselines | Complete | `book/classical-baselines.qmd`, `src/diff_epi_inference/mcmc/`, `tests/test_mh_smoke.py`, `tests/test_hmc_smoke.py` |
| M3 Likelihood-free baselines | Complete | `book/likelihood-free-baselines.qmd`, `src/diff_epi_inference/abc/`, `tests/test_abc_rejection.py`, `tests/test_smc_abc.py` |
| M4 Modern SBI | Complete | `book/modern-sbi.qmd`, `src/diff_epi_inference/flows/conditional_affine.py`, `tests/test_modern_sbi_end_to_end_golden.py` |
| M5 Variational inference + hybrids | Complete | `book/variational-inference.qmd`, `src/diff_epi_inference/vi/meanfield_gaussian_jax.py`, `tests/test_vi_meanfield_seir_beta_only_optional.py` |
| M6 Autodiff + differentiability | Complete | `book/autodiff-basics.qmd`, `book/differentiability-axis.qmd`, `src/diff_epi_inference/autodiff/`, `tests/test_differentiability_demos.py` |
| M7 Validation + robustness | Complete | `book/validation-and-robustness.qmd`, calibration smoke coverage in `tests/` |
| M8 Polish + publication | Complete | `.github/workflows/pages.yml`, `docs/BUILD.md`, Quarto CI profile in `book/_quarto-ci.yml` |

## Working Rule

- Keep features intentionally minimal and reproducible.
- Prefer tested helper functions in `src/` over large in-chapter code blocks.
- Treat CI `build-html` as the authoritative executable documentation render.
