# Differentiable Epidemiology Inference Handbook – Development Plan (gap-to-goal)

This plan is written after reviewing the current repository contents in `diff-epi-inference-handbook-main.zip` and aligning it with:
1) the existing milestone roadmap (M0–M8) in `ROADMAP.md`; and
2) the broader “original project plan” goal: an exhaustive, beginner-accessible (non-ML background) handbook explaining SBI, surrogates, autodiff, differentiability, and modern inference tricks, with infectious-disease-flavoured examples, rendered as both a website and a PDF book.

The repo is already in good shape structurally: Quarto book, executable examples, a minimal companion Python package, a healthy set of unit tests, CI that builds HTML (and optionally PDF), and tag-triggered GitHub Pages publishing. The biggest remaining work is content expansion and implementing the “modern SBI” and “autodiff/differentiability” material in a way that stays runnable and reproducible.

---

## 0. Current snapshot (what is already implemented)

### A. Build and repo hygiene
- Quarto book lives in `book/` and renders to HTML and PDF from one source.
- CI has three jobs: lint+tests, HTML render, and best-effort PDF render.
- GitHub Pages publishing is configured via a workflow that renders HTML and pushes `book/_book` to `gh-pages` on version tags.

### B. Companion Python package (dependency-light)
- `diff_epi_inference` is NumPy-first, with optional extras for JAX/BlackJAX.
- Core pieces are in place:
  - SEIR deterministic Euler simulator
  - Stochastic tau-leaping simulator
  - Observation / reporting model (delay + NegBin noise)
  - A minimal dataset format and IO (`.npz`)
  - MH and an educational HMC implementation (finite differences)
  - Optional BlackJAX NUTS baseline (guarded)
  - ABC rejection, SMC-ABC, synthetic likelihood utilities
  - Plotting helpers (traces, ACF, time series)

### C. Book content (current chapter set)
- `running-example.qmd`: deterministic and stochastic SEIR + observation pipeline
- `classical-baselines.qmd`: MH, HMC, optional NUTS, PPC overlays, calibration smoke tests
- `likelihood-free-baselines.qmd`: ABC rejection, SMC-ABC, synthetic likelihood
- `modern-sbi.qmd`: flow tutorial (minimal), and a beta-only “NPE loop” using a conditional Gaussian (not yet a conditional flow)

---

## 1. Design review: what to keep, what to change

### Keep (these are good design choices)
1) **Single-source Quarto book**. This matches your website + printable PDF goal.
2) **Dependency-light core package**. This is excellent for pedagogical reproducibility.
3) **Tests-first infrastructure**. The repo already has enough tests to prevent regression.
4) **Tag-only publishing**. This prevents accidental public site updates.

### Change or decide explicitly (to avoid future pain)
1) **Advanced chapters will need autodiff**.
   - Training a conditional flow for NPE (and doing VI/GVI properly) is painful without autodiff.
   - Recommendation: adopt a clear “two-tier dependency policy”:
     - Tier 1 (core): NumPy-only, everything in M1–M3 works.
     - Tier 2 (advanced): JAX-based chapters (M4–M6) enabled via an extra (e.g. `.[jax]` or `.[ml]`) and included in the book build environment.

2) **Book narrative structure needs “foundations” pages**.
   - The current book jumps quickly into code.
   - Your stated end goal is “start from basics for my level”, so the book should begin with short but explicit prerequisite chapters (Bayesian recap, likelihood vs simulator, what “gradient-based” really means).
   - Recommendation: add a “Part 0: Orientation and foundations” before the running example.

3) **Runtime budget discipline**.
   - SBI training loops can blow up CI runtime quickly.
   - Recommendation:
     - Keep all “book execution” examples small and deterministic.
     - Provide “scale-up notes” rather than scale-up defaults.
     - Add a “fast mode” flag for code cells where possible (small epochs, small simulation counts).

4) **Avoid scattering large amounts of algorithm code inside QMD files**.
   - Keep QMD pages narrative-heavy.
   - Put reusable code into `src/diff_epi_inference/...` with tests, then call it from QMD.

---

## 2. Roadmap alignment (M0–M8)

The existing repo roadmap defines milestones M0–M8. The repo is effectively at M4 (in progress). M0–M2 are complete, M3 is effectively implemented but needs a status tick, and M4 needs core work (conditional flow NPE).

Below, “DONE” means: code, tests, and book content exist and render end-to-end.

- M0 (Bootstrap): DONE
- M1 (Running example v1): DONE
- M2 (Classical baselines): DONE
- M3 (Likelihood-free baselines): MOSTLY DONE (chapter + code exist; finish DoD verification + status update)
- M4 (Modern SBI): IN PROGRESS (flow tutorial exists; NPE is currently conditional Gaussian; needs conditional flow + diagnostics)
- M5 (VI/GVI + hybrids): NOT STARTED
- M6 (Autodiff + differentiability): NOT STARTED
- M7 (Validation + robustness + cookbook): NOT STARTED (some pieces exist as “smoke checks” already)
- M8 (Polish + publication): NOT STARTED

---

## 3. Detailed milestone-by-milestone development plan

### M3 – Likelihood-free baselines (closure work)
Goal (per roadmap): ABC rejection + optional SMC-ABC + synthetic likelihood baselines.

What exists already:
- ABC rejection demo
- SMC-ABC demo
- Synthetic likelihood demo
- Supporting code: `diff_epi_inference.abc`, `diff_epi_inference.synthetic_likelihood`
- Tests for ABC and SMC exist

Remaining tasks (to mark M3 complete cleanly):

M3.1 – DoD verification checklist
- Confirm:
  - `book/likelihood-free-baselines.qmd` executes without manual intervention.
  - ABC + SMC-ABC + synthetic likelihood sections run in reasonable time on CI.
  - The chapter produces at least one diagnostic plot or summary for each method.

DoD:
- CI “build-html” job passes on `main` with this chapter included.

M3.2 – Add one minimal diagnostic per method (if currently missing)
- ABC: plot accepted beta histogram + mark beta_true
- SMC-ABC: show final-round weighted histogram + ESS readout
- Synthetic likelihood: trace plot + posterior predictive overlay (even if very small)

Files:
- `book/likelihood-free-baselines.qmd`
- optionally small helper functions in `diff_epi_inference.plotting`

DoD:
- Chapter outputs show something diagnostic beyond just printed numbers.

M3.3 – Update `docs/STATUS.md` to mark M3 complete and set milestone to M4
- Update milestone state and “nextStep”.

DoD:
- Status block accurately reflects repo stage.

---

### M4 – Modern SBI (flows + NPE/NLE/NRE)

Roadmap DoD: normalising flow tutorial + conditional flows + NPE pipeline with calibration diagnostics; amortised vs local comparison.

This is the largest “methods” milestone remaining.

#### M4.0 – Decide dependency strategy for modern SBI (blocking decision)
Problem:
- A true NPE conditional flow needs gradients and stable optimisation.
- Pure NumPy implementation is possible but will be time-consuming and brittle.

Recommendation:
- Introduce an extra dependency group, for example:
  - `ml = ["jax[cpu]>=0.4.28", "optax>=0.2.3"]`
  - Keep `nuts` for `blackjax` if desired, or merge.
- Update CI “build-html” to install `.[dev,ml]` once M4 contains code that requires JAX.
  - Alternatively: keep CI as-is and guard advanced code blocks, but then the published site will not execute those parts unless you commit freezes, which is not currently done.

Files:
- `pyproject.toml` (new optional dependency group)
- `constraints-dev.txt` (pin versions for reproducibility)
- `.github/workflows/ci.yml` (install extra group for HTML build)
- `docs/BUILD.md` (explain optional extras and which chapters need them)

DoD:
- A fresh CI run builds the HTML book including M4 content.

#### M4.1 – Implement a minimal conditional flow in the package
Target: something RealNVP-like for a low-dimensional theta (2D or 3D), conditioned on a summary vector.

Key design choice:
- For beta-only (1D theta), most common coupling flows are overkill and not very expressive unless you implement 1D monotone spline flows.
- Recommendation: expand theta to 2D for the first conditional-flow NPE demo (e.g. infer log(beta) and log(reporting_rate), or log(beta) and log(dispersion)) so that coupling layers are meaningful.

Implementation tasks:
- Create `src/diff_epi_inference/flows/` package.
- Implement:
  - Base Gaussian logpdf and sampler
  - A coupling layer (split x into x_a, x_b):
    - x_b' = x_b * exp(s(x_a, c)) + t(x_a, c)
    - where c is condition (summary vector)
  - A stack of coupling layers + permutations
  - Log-det Jacobian computation
  - Forward and inverse maps
- Implement a tiny MLP conditioner in JAX:
  - `mlp(params, x)` with tanh or relu
  - parameter initialisation helpers
- Keep everything small and explicit (no Flax/Haiku unless you decide you want them).

Tests:
- Invertibility test: inverse(forward(z)) recovers z to tolerance
- Logdet finite test
- Shapes and dtype validation
- Smoke training test (tiny dataset, a few optimisation steps) that confirms loss decreases

DoD:
- `pytest` covers the flow module basics and passes in CI.
- A simple “fit to toy data” example runs in under a minute.

#### M4.2 – Replace the conditional Gaussian NPE demo with conditional-flow NPE
Book tasks:
- Update `book/modern-sbi.qmd`:
  - Keep the conditional Gaussian demo as “baseline NPE without deep learning” (useful pedagogically).
  - Add a new section: “NPE with a conditional flow (JAX)”.
- Define a consistent NPE loop:
  1) sample theta from prior
  2) simulate y from stochastic pipeline
  3) compute summary s(y)
  4) train conditional flow to model q(theta | s)
  5) evaluate on one observed dataset and sample posterior

Code tasks:
- Add `diff_epi_inference/sbi/npe.py` (or similar) containing:
  - `simulate_dataset(theta, rng) -> y`
  - `summary_fn(y) -> s`
  - `sample_prior(rng) -> theta`
  - `train_npe_flow(...) -> trained_model`
  - `sample_posterior(...) -> theta_samples`
- Keep the book calling these functions to avoid giant code blocks.

Tests:
- Smoke test: train for N steps on small sims and produce finite posterior samples
- Reproducibility test with fixed seeds

DoD:
- Running `quarto render book --to html` executes the conditional-flow NPE section end-to-end.

#### M4.3 – Add a calibration diagnostic for NPE (SBC-style smoke test)
Minimum viable diagnostic:
- A small simulation-based calibration loop:
  - sample theta_true ~ prior
  - simulate y
  - train or use amortised model
  - compute rank of theta_true within posterior samples
  - repeat for ~20 trials, check rank histogram is not pathological

Important practical constraint:
- Full amortised SBC is expensive if you retrain the flow per dataset.
- For a smoke test, use either:
  - amortised training once, then evaluate multiple y’s; or
  - local training but very small epochs and small trials.

Implementation tasks:
- Add `diff_epi_inference/diagnostics/sbc.py` with:
  - `rank_statistic(...)`
  - `sbc_smoke_test(...)`
- Add minimal plots in the book.

DoD:
- Book shows at least one calibration plot and explains how to interpret it.

#### M4.4 – Amortised vs local training comparison
Goal:
- Make “amortisation” concrete.

Implementation approach:
- Amortised:
  - train on a simulation bank (theta_i, s_i) once
  - evaluate on one y_obs
- Local:
  - train (or fine-tune) using simulations targeted near the observation
  - simplest: start from amortised weights then fine-tune on extra sims around posterior mean

Book deliverable:
- A table or plot comparing:
  - compute budget (number of sims)
  - posterior width / bias
  - PPC results

DoD:
- Reader can see a genuine trade-off, not just narrative.

#### M4.5 – Add brief NLE and NRE sections (even if code is minimal)
You do not need to fully implement NLE/NRE to satisfy understanding goals, but you should include:
- Conceptual explanation
- Minimal toy example where possible

Tasks:
- NLE: fit q(y|theta) on summaries via Gaussian regression, then do MCMC using that surrogate likelihood (you already have synthetic likelihood; extend this as “learned likelihood”).
- NRE: classify joint vs marginal pairs and recover log ratio; likely keep as conceptual unless you want to implement.

DoD:
- Modern SBI chapter covers NPE, NLE, NRE at least at an intuitive and implementable level.

---

### M5 – VI/GVI + hybrids

Roadmap DoD: VI/GVI chapter (flows as variational family) + hybrid example (learned proposal or flow-assisted sampler).

#### M5.0 – Add a new Quarto chapter: `book/variational-inference.qmd`
Content tasks:
- Explain:
  - what VI is optimising (ELBO)
  - why it is different from sampling
  - amortised vs non-amortised VI
- Start with a toy Gaussian model where VI is exact.

DoD:
- Chapter renders and includes at least one runnable toy VI example.

#### M5.1 – Implement mean-field Gaussian VI for the SEIR beta-only model (JAX)
Implementation tasks:
- Add `diff_epi_inference/vi/meanfield.py`:
  - parameterisation: q(logbeta) = Normal(mu, sigma)
  - reparameterisation sampling
  - ELBO estimate
  - gradient steps (Adam or SGD)

Book tasks:
- Demonstrate VI posterior vs MH/HMC posterior on the same synthetic dataset.
- Add PPC overlay from VI posterior.

Tests:
- Smoke test: ELBO increases (or negative loss decreases) for a few steps.
- Output finite and stable.

DoD:
- VI produces a plausible posterior and is compared to baselines.

#### M5.2 – Flow-based VI (reuse the conditional flow machinery)
Implementation tasks:
- Adapt flow to be an unconditional variational family q_phi(theta) (condition is absent or fixed).
- Optimise flow parameters to maximise ELBO.

Book tasks:
- Show where flow VI helps (non-Gaussian posterior).
- Use a 2D parameter example if needed.

DoD:
- At least one case where mean-field VI is visibly insufficient and flow-VI improves.

#### M5.3 – GVI framing and “gradient-based inference” explanation
This is a pedagogical bridge to the autodiff chapter:
- Explain gradients as local sensitivity, not a new philosophy of modelling.
- Clarify what “gradient-based inference” includes:
  - HMC
  - VI
  - gradient-informed proposals

Deliverable:
- A short section that maps your key terms to the pipeline.

DoD:
- Readers can distinguish “optimisation”, “sampling”, and “amortised learning”.

#### M5.4 – Hybrid example: flow-assisted sampler
Choose one hybrid that is both pedagogically clear and implementable:
Option A: Independence MH with a trained flow proposal
- Fit a flow to approximate posterior
- Use it as a proposal in MH
- Correct with MH acceptance ratio

Option B: Flow-initialised HMC
- Use VI / NPE to find good initial states and mass matrix heuristics

Implementation tasks:
- Add `diff_epi_inference/mcmc/independence_mh.py` or similar.
- Reuse the flow’s `log_prob` and `sample` methods.

DoD:
- Hybrid sampler demonstrates a measurable improvement (mixing or compute) over a naive baseline.

---

### M6 – Autodiff + differentiability axis

Roadmap DoD: forward vs reverse mode AD chapter with runnable demos; differentiable vs non-diff simulator decision guidance.

This milestone is essential for your stated goal of “not thinking in gradients but understanding what they are doing”.

#### M6.0 – Add two new Quarto chapters (minimum)
1) `book/autodiff-basics.qmd`
2) `book/differentiability-axis.qmd`

Update `_quarto.yml` to include these chapters, ideally as a new “Part” in the book.

DoD:
- Both chapters render and include at least one runnable demo each.

#### M6.1 – Autodiff basics chapter tasks
Content:
- What AD is (vs symbolic vs finite differences)
- Forward mode vs reverse mode:
  - JVP (Jacobian-vector product) intuition
  - VJP (vector-Jacobian product) intuition
- Backprop as reverse-mode AD
- Why deep learning uses reverse mode

Runnable demos:
- A tiny scalar function and its gradient
- A vector-valued function where forward mode is efficient
- A scalar loss with many parameters where reverse mode wins

Implementation support:
- Add `diff_epi_inference/autodiff/demos.py` with small JAX snippets that the book imports.

DoD:
- The demos run in a few seconds and produce clear printed results and a small plot.

#### M6.2 – Differentiable vs non-differentiable models chapter tasks
Content:
- A taxonomy:
  - differentiable deterministic simulator (ODE)
  - stochastic but reparameterisable simulator
  - discrete-event simulator (ABM) with non-diff control flow
- Practical guidance:
  - when to use HMC/VI vs SBI vs ABC
  - when surrogates are safer than differentiating the simulator

Runnable demos:
- Show finite difference instability on a discontinuous function
- Show JAX gradient on a smooth function
- Show that tau-leaping SEIR is non-differentiable as implemented

Deliverable:
- A decision tree figure (Quarto diagram or simple matplotlib plot) summarising method choice.

DoD:
- Chapter ends with a “method selection checklist” for the reader.

#### M6.3 – Gradient estimators and relaxations (needed for Quera-Bofarull-style work)
This is likely required to understand differentiable ABMs and “tricks”.

Content sections:
- Finite differences (and why they are fragile)
- Score-function estimator (REINFORCE): unbiased, high variance
- Reparameterisation trick: low variance, requires reparameterisable noise
- Straight-through estimators: biased but practical
- Gumbel-Softmax / Concrete relaxation for categorical choices

Runnable demos:
- A toy Bernoulli “infection event”:
  - compare finite diff vs REINFORCE vs a soft relaxation
- A toy categorical choice with Gumbel-Softmax

Implementation support:
- Add `diff_epi_inference/grad_estimators/` with minimal functions used in demos.

DoD:
- Reader sees the bias/variance trade-off empirically in one plot.

#### M6.4 – “Differentiable epidemiology” mini-case-study (toy)
Goal:
- Demonstrate what “making an epidemic simulator differentiable” means in practice, without building a full ABM.

Candidate toy:
- Replace discrete infection events with expected transitions (mean-field approximation)
- Or implement a “soft” stochastic transition with reparameterised noise

Deliverables:
- A short case study page: “From discrete events to differentiable relaxations”
- One optimisation task: fit beta by gradient descent on this relaxed simulator

DoD:
- The toy example is runnable and conceptually aligned with the differentiability literature.

---

### M7 – Validation + robustness + cookbook

Roadmap DoD: SBC/posterior checks chapter; failure modes + method-selection guide.

This should consolidate diagnostics used earlier and make them systematic.

#### M7.0 – Add a new Quarto chapter: `book/validation-and-robustness.qmd`
Sections:
- Posterior predictive checks (PPC): what they do and do not tell you
- Simulation-based calibration (SBC): rank histograms, coverage
- Sensitivity:
  - prior sensitivity
  - summary selection sensitivity (for ABC/SBI)
  - simulator discrepancy and model misspecification

Runnable content:
- Reuse the existing calibration smoke tests from M2, but present them as a general diagnostic pattern.
- Add an SBC smoke test for NPE/VI once implemented.

DoD:
- A reader can apply a repeatable validation workflow to any new method chapter.

#### M7.1 – Implement reusable diagnostic helpers in the package
Add `diff_epi_inference/diagnostics/`:
- `ppc.py`: simulate replicated datasets and compute envelope bands
- `coverage.py`: compute empirical coverage over simulated datasets
- `sbc.py`: rank statistic utilities (if not already done in M4)

Tests:
- input validation tests
- deterministic reproducibility tests with seeded RNG

DoD:
- Diagnostics are used consistently across MH/HMC/ABC/NPE/VI chapters.

#### M7.2 – Failure modes chapter section (high value)
Write a “failure modes” section covering:
- non-identifiability and weakly identified parameters
- posterior multimodality
- simulator mismatch (structural model error)
- amortisation bias
- overconfident neural posteriors
- failure to cover due to summary choice

DoD:
- Includes at least one intentionally “broken” example showing a failure mode and its symptom.

#### M7.3 – Method-selection guide (“cookbook”)
Deliverable:
- A single page that answers: “Given my simulator and data, what should I try first?”
Inputs:
- differentiable vs not
- likelihood available vs not
- dimension of theta
- compute budget
- need for amortisation

Output:
- recommended baseline path:
  - Start with MH/HMC if possible
  - Else ABC baseline
  - Then NPE/NLE
  - Then VI/hybrids

DoD:
- A simple decision flowchart and a short checklist.

---

### M8 – Polish + publication

Roadmap DoD: glossary, references, indexing, stable releases, public site + versioned PDF releases.

This is mostly book-quality and release engineering work.

#### M8.0 – Add a glossary and notation index
Deliverables:
- `book/glossary.qmd` with:
  - acronyms (SBI, NPE, NLE, NRE, VI, GVI, HMC, NUTS, PF, SMC, ABC)
  - consistent notation table (theta, y, x, s(y), etc)
- Ensure chapters link to glossary entries.

DoD:
- Every new acronym introduced in text appears in the glossary.

#### M8.1 – Add citation and bibliography system
Deliverables:
- `book/references.bib`
- Quarto configuration for citations
- Replace informal mentions with citations (especially for: ABC, NPE, flows, BlackJAX, flow matching, REINFORCE, Gumbel-Softmax)

DoD:
- PDF and HTML have a references section and in-text citations render correctly.

#### M8.2 – PDF quality pass
Deliverables:
- Improve PDF styling:
  - code block formatting
  - figure placement
  - consistent fonts
  - chapter/section numbering
- Decide whether to use Quarto’s default LaTeX template or a custom template.

DoD:
- PDF is readable and looks book-like, not like a raw notebook dump.

#### M8.3 – Release automation
Goal:
- Tags produce:
  - GitHub Pages site update
  - a built PDF artefact attached to the GitHub Release (or stored as an Actions artefact)

Tasks:
- Update CI so PDF build is not “continue-on-error” for tags/releases.
- Add a workflow to upload the PDF.

DoD:
- `vX.Y.Z` tag leads to a site and a downloadable PDF without manual steps.

#### M8.4 – Changelog and versioning
Deliverables:
- `CHANGELOG.md`
- Decide semantic versioning rules (even if pre-1.0)

DoD:
- Every release tag has an entry and links to the rendered site.

---

## 4. Additional content needed to match the broader “original plan” (recommended extensions)

The milestone roadmap is strong for the core inference pipeline, but your stated goal includes additional concepts that are not explicit in M0–M8 (or only implicit). I recommend treating these as “extension milestones” once M4–M6 are stable.

### E1 – Surrogates and emulators (forward models)
Why:
- This is central for understanding modern epidemic inference workflows and the “surrogates” literature.

Deliverables:
- New chapter: `book/surrogates.qmd`
- Code: `diff_epi_inference/surrogates/`
  - simple regression surrogate for summary statistics
  - uncertainty-aware surrogate (e.g. Gaussian likelihood on residuals, or ensemble)
- Demonstrate:
  - how surrogates speed up MCMC/VI
  - where surrogates break (extrapolation)

DoD:
- A reader can build and validate a surrogate on the running example.

### E2 – Particle filters and state-space models
Why:
- You explicitly listed particle filters, and they are essential for sequential inference.

Deliverables:
- New chapter: `book/particle-filters.qmd`
- Minimal state-space model example (not necessarily SEIR first):
  - linear Gaussian SSM (Kalman as reference)
  - bootstrap particle filter
- Optional extension:
  - particle marginal MH (PMMH) demonstration
  - link to SMC-ABC as a conceptual cousin

Code:
- `diff_epi_inference/smc/particle_filter.py` etc.

DoD:
- At least one working PF demo with clear plots and explanation of degeneracy/resampling.

### E3 – Flow matching / diffusion-style SBI (optional advanced)
Why:
- You listed flow matching, and it is a modern direction in conditional generative modelling.

Deliverables:
- New chapter: `book/flow-matching.qmd`
- Keep it minimal:
  - conceptual explanation
  - a toy example (2D) where flow matching trains a continuous flow field

Dependency warning:
- Likely requires more tooling; keep it clearly “optional advanced”.

DoD:
- Reader has an implementable mental model of flow matching, even if they do not run it at scale.

### E4 – “Reading guide” to specific author lines of work
Goal:
- Make it easy to read Quera-Bofarull, Dyer, Semenova papers by mapping paper concepts to handbook sections.

Deliverables:
- New chapter: `book/reading-guide.qmd`
- For each author/topic line:
  - “What they assume you know”
  - “Where to learn that in this book”
  - “Core idea in 5 bullets”
  - “Common confusions”
  - “Minimal reproduction exercise” (where feasible)

DoD:
- You can pick a paper and follow an explicit trail of prerequisites in the book.

---

## 5. Practical issue breakdown (how to turn this into GitHub issues)

Recommended labels:
- `milestone:M4`, `milestone:M5`, ...
- `type:book`, `type:code`, `type:tests`, `type:ci`
- `priority:P0` (blocking), `P1` (high), `P2` (nice-to-have)

Suggested first batch of issues (high priority):
1) M3.1–M3.3 (close M3 properly)
2) M4.0 dependency strategy decision
3) M4.1 flow module skeleton + invertibility tests
4) M4.2 conditional-flow NPE demo wired into book
5) M6.0 add autodiff chapter skeleton (even before it is fully filled)

---

## 6. Acceptance criteria summary (quick checklist)

By the time M6 is complete, you should have:
- A book that begins with foundations suitable for a non-ML Bayesian modeller.
- A single running example used consistently across:
  - MH/HMC/NUTS
  - ABC/SMC-ABC/synthetic likelihood
  - NPE with conditional flows
  - VI/GVI
- A coherent explanation of gradients and autodiff, with runnable demos.
- A diagnostics framework (PPC, calibration, SBC smoke tests) used everywhere.

And by M8:
- A polished, citable, versioned handbook with a stable site and PDF artefacts per release.

