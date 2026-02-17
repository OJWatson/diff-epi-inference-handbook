# Spec-to-Repo Map

This map ties the canonical specs to concrete implementation and tests.

Canonical spec inputs:

- `docs/diff_epi.pdf`
- `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`
- `README.md`

## Core Model + Observation Pipeline (M1)

- SEIR simulators: `src/diff_epi_inference/seir.py`
- Observation/reporting helpers: `src/diff_epi_inference/observation.py`
- End-to-end reporting pipelines: `src/diff_epi_inference/pipeline.py`
- Tests: `tests/test_seir_smoke.py`, `tests/test_seir_stochastic_smoke.py`, `tests/test_observation_model.py`, `tests/test_paired_reporting_pipeline.py`

## Dataset Schema + IO

- Schema object: `src/diff_epi_inference/dataset.py`
- IO contract (`.npz`): `src/diff_epi_inference/io.py`
- Schema invariants:
  - `t` is finite, strictly increasing, 1D
  - `y` is 1D, non-negative, finite when float
  - `len(t) == len(y) >= 1`
- Tests: `tests/test_dataset_format.py`, `tests/test_io_roundtrip.py`

## Inference Baselines

- MCMC (MH/HMC/NUTS): `src/diff_epi_inference/mcmc/`
- Likelihood-free (ABC/SMC/synthetic likelihood): `src/diff_epi_inference/abc/`, `src/diff_epi_inference/synthetic_likelihood.py`
- Tests: `tests/test_mh_smoke.py`, `tests/test_hmc_smoke.py`, `tests/test_blackjax_nuts_optional.py`, `tests/test_abc_rejection.py`, `tests/test_smc_abc.py`, `tests/test_synthetic_likelihood_utils.py`

## Modern SBI / VI / Differentiability

- Conditional density helper (minimal flow baseline): `src/diff_epi_inference/flows/conditional_affine.py`
- VI helper: `src/diff_epi_inference/vi/meanfield_gaussian_jax.py`
- Differentiability + autodiff demos: `src/diff_epi_inference/differentiability/`, `src/diff_epi_inference/autodiff/`
- Tests: `tests/test_conditional_affine_flow.py`, `tests/test_modern_sbi_end_to_end_golden.py`, `tests/test_vi_meanfield_seir_beta_only_optional.py`, `tests/test_differentiability_demos.py`, `tests/test_autodiff_demos_optional.py`

## Book + Deployment

- Book config: `book/_quarto.yml`, `book/_quarto-ci.yml`
- CI and Pages workflows: `.github/workflows/ci.yml`, `.github/workflows/pages.yml`
- Build/run docs: `docs/BUILD.md`, `README.md`, `AGENTS.md`
