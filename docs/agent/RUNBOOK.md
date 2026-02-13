# RUNBOOK

## Acceptance gates

Run from repo root:

- Unit tests:
  - `uv run pytest -q`
- Render book (HTML):
  - `quarto render book --to html`

## Notes

- Prefer small, deterministic examples in rendered chapters.
- Keep changes atomic per milestone where possible.
