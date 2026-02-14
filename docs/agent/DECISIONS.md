# DECISIONS

- 2026-02-14 (M4.0): Modern SBI dependencies are optional extras.
  - Core install stays NumPy-only (`project.dependencies` minimal) so the book remains lightweight.
  - JAX is exposed as `.[jax]`.
  - Existing NUTS/BlackJAX path is `.[nuts]` (and is exercised in CI via a dedicated job).
  - The M4+ "modern SBI" stack is `.[modern-sbi]` (JAX + equinox/optax/distrax).
  - CI uses `uv` for installs/runs (matches local acceptance commands).

- 2026-02-13: Restarted agent continuity docs under docs/agent/.
