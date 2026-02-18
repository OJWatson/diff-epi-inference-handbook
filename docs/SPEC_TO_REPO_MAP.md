# Spec-to-Repo Map

This document maps the canonical product brief/spec inputs to concrete book chapters,
package modules, and tests.

## Canonical inputs

- `docs/diff_epi.pdf`
- `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`
- `README.md`

## Coverage matrix

| Spec theme | Primary chapter(s) | Package modules | Main tests | Status |
|---|---|---|---|---|
| Deterministic/stochastic SEIR simulation | `book/running-example.qmd` | `src/diff_epi_inference/seir.py`, `src/diff_epi_inference/pipeline.py` | `tests/test_seir_smoke.py`, `tests/test_seir_stochastic_smoke.py`, `tests/test_running_example_smoke.py` | Covered |
| Observation model (reporting, delay, over-dispersion) | `book/running-example.qmd`, `book/classical-baselines.qmd` | `src/diff_epi_inference/observation.py` | `tests/test_observation_model.py`, `tests/test_observation_delay_nbinom.py` | Covered |
| Dataset schema and IO contract | `book/running-example.qmd` | `src/diff_epi_inference/dataset.py`, `src/diff_epi_inference/io.py` | `tests/test_dataset_format.py`, `tests/test_io_roundtrip.py` | Covered |
| Classical MCMC baselines (MH/HMC/NUTS) | `book/classical-baselines.qmd` | `src/diff_epi_inference/mcmc/`, `src/diff_epi_inference/models/seir_numpy_beta_only.py`, `src/diff_epi_inference/models/seir_jax_beta_only.py` | `tests/test_mh_smoke.py`, `tests/test_hmc_smoke.py`, `tests/test_blackjax_nuts_optional.py`, `tests/test_blackjax_nuts_seir_calibration_optional.py` | Covered |
| Likelihood-free baselines (ABC/SMC/synthetic likelihood) | `book/likelihood-free-baselines.qmd` | `src/diff_epi_inference/abc/`, `src/diff_epi_inference/synthetic_likelihood.py` | `tests/test_abc_rejection.py`, `tests/test_smc_abc.py`, `tests/test_smc_resampling.py`, `tests/test_synthetic_likelihood_utils.py` | Covered |
| Variational inference | `book/variational-inference.qmd` | `src/diff_epi_inference/vi/meanfield_gaussian_jax.py` | `tests/test_vi_meanfield_seir_beta_only_optional.py` | Covered |
| Differentiability and gradient estimators | `book/autodiff-basics.qmd`, `book/differentiability-axis.qmd`, `book/gradient-estimators.qmd`, `book/toy-diff-epi-case-study.qmd` | `src/diff_epi_inference/autodiff/`, `src/diff_epi_inference/differentiability/`, `src/diff_epi_inference/gradients/`, `src/diff_epi_inference/examples/toy_diff_epi_jax.py` | `tests/test_autodiff_demos_optional.py`, `tests/test_differentiability_demos.py`, `tests/test_reinforce.py`, `tests/test_gumbel_softmax.py`, `tests/test_toy_diff_epi_case_study_optional.py` | Covered |
| Modern SBI (NPE/NLE/NRE framing, conditional flow baseline) | `book/modern-sbi.qmd` | `src/diff_epi_inference/flows/conditional_affine.py`, `src/diff_epi_inference/pipeline.py` | `tests/test_conditional_affine_flow.py`, `tests/test_modern_sbi_end_to_end_golden.py` | Covered |
| Validation workflow (PPC/SBC/sensitivity) | `book/validation-and-robustness.qmd` | `src/diff_epi_inference/plotting/chapter.py`, `src/diff_epi_inference/plotting/timeseries.py` | `tests/test_plotting_chapter_story_helpers.py`, `tests/test_plotting_chapter_helpers.py` | Covered |
| Book publishing and CI | `book/_quarto.yml`, `docs/BUILD.md` | N/A | CI workflow checks in `.github/workflows/ci.yml`, publishing in `.github/workflows/pages.yml` | Covered |

## Schema contract (dataset)

- `t` is finite, strictly increasing, and 1D.
- `y` is 1D, non-negative, and finite when float.
- `len(t) == len(y) >= 1`.
- Contract owner: `src/diff_epi_inference/dataset.py`.

## Drift register

| Item | Current state | Owner | Recovery task |
|---|---|---|---|
| `docs/diff_epi.pdf` chapter plan vs current book structure | Diverged | Repo maintainer | Keep this map as source of truth; refresh or replace PDF with chapter structure matching `book/_quarto.yml` at next spec update. |
| Citation support in the book | Recovered | Repo maintainer | Maintain `book/references.bib` and citation markers in narrative chapters; keep bibliography wired in `book/_quarto.yml`. |
| Roadmap-style chapter framing | Recovered | Repo maintainer | Preserve completed-reader framing; `scripts/check_book_quality.sh` now fails CI for roadmap-style section markers. |
| Excessive in-chapter code cell length | Recovered (guarded) | Repo maintainer | Keep cells under quality threshold; CI guard in `scripts/check_book_quality.sh`. |
