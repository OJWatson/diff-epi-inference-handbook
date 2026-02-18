# Differentiable Epidemiology Inference Handbook — Project Brief

This document describes the current design intent for the handbook and companion codebase.
It is written as a stable product brief rather than an internal roadmap.

## Purpose

The repository provides:

- a readable, executable handbook built with Quarto;
- a lightweight Python package containing reusable inference/modeling utilities;
- reproducible CI and publishing workflows for HTML documentation.

The emphasis is clarity and reproducibility over maximal feature count.

## Design principles

1. Keep examples small enough to run in CI.
2. Keep the default installation lightweight.
3. Prefer reusable, tested Python helpers over large in-chapter code blocks.
4. Keep chapter narrative and mathematics central; code should support, not dominate.
5. Maintain deterministic rendering paths for publication.

## Repository structure

- `book/`: Quarto source for the handbook website/book.
- `src/diff_epi_inference/`: companion Python package used by chapters and tests.
- `tests/`: regression and smoke tests for package behavior.
- `scripts/`: local wrappers for lint/test acceptance commands.
- `.github/workflows/`: CI and GitHub Pages publishing workflows.
- `docs/`: build instructions and project reference material.

## Technical scope

The package currently includes:

- deterministic and stochastic SEIR simulation helpers;
- observation/reporting model utilities (delay and over-dispersion);
- baseline inference methods (MCMC and likelihood-free utilities);
- modern simulation-based inference examples;
- plotting helpers used across chapters;
- optional JAX-based differentiability and VI demonstrations.

## Runtime and dependency strategy

- Core usage is NumPy-first.
- Optional extras enable advanced chapters (e.g. JAX/BlackJAX and modern SBI stack).
- CI executes HTML rendering with the advanced extras required by executable chapters.
- PDF build is best-effort in CI and should not block core quality gates.

## Publication model

- Main branch CI validates lint, tests, and executable HTML rendering.
- Pushes to `main` (and optional release tags) publish the rendered site to GitHub Pages.
- The live site should always correspond to a reproducible render from versioned sources.
