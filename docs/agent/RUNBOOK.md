# RUNBOOK

## Acceptance gates

Run from repo root:

- Unit tests:
  - `uv run pytest -q`
- Render book (HTML):
  - `quarto render book --to html`

## M3.1 — Likelihood-free baselines DoD verification

Checklist: `docs/agent/M3.1_DOD_CHECKLIST.md`

## Notes

- Prefer small, deterministic examples in rendered chapters.
- Keep changes atomic per milestone where possible.
