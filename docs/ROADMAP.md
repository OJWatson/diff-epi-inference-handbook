# Roadmap

Project: **Differentiable Epidemiology**

Subtitle: *Neural and Simulation‑Based Inference for Infectious Diseases*

Repo: `diff-epi-inference-handbook`

## M0 — Bootstrap (book skeleton + reproducible build)
- DoD:
  - Website builds (HTML) and book builds (PDF) from one source.
  - CI runs lint + tests + book build (HTML at minimum; PDF if feasible).
  - A minimal running example page executes end-to-end.

## M1 — Running example v1 (SEIR + reporting)
- DoD:
  - Differentiable ODE SEIR + observation model.
  - Non-differentiable stochastic variant (paired to the same conceptual model).
  - Standard dataset format + plotting utils + tests.

## M2 — Classical baselines (amortisation-free anchors)
- DoD:
  - MH MCMC baseline.
  - HMC/NUTS on differentiable model.
  - Diagnostics: posterior predictive + basic calibration checks.

## M3 — Likelihood-free baselines (ABC + synthetic likelihood)
- DoD:
  - ABC rejection + (optional) SMC-ABC with careful summaries.
  - Synthetic likelihood / density estimation on summaries.

## M4 — Modern SBI (flows + NPE/NLE/NRE)
- DoD:
  - Normalising flow tutorial + conditional flows.
  - NPE pipeline with calibration diagnostics; amortised vs local comparison.

## M5 — VI/GVI + hybrids
- DoD:
  - VI/GVI chapter (incl flows as variational family).
  - Hybrid example (learned proposal / flow-assisted sampler).

## M6 — Autodiff + differentiability axis
- DoD:
  - Forward vs reverse mode AD chapter with runnable demos.
  - Differentiable vs non-diff simulator decision guidance.

## M7 — Validation + robustness + “cookbook”
- DoD:
  - SBC/posterior checks chapter.
  - Failure modes + method-selection guide.

## M8 — Polish + publication
- DoD:
  - Glossary, references, indexing, stable releases.
  - Public site + versioned PDF releases.
