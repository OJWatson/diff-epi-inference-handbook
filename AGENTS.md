# AGENTS

## Spec locations

- `docs/diff_epi.pdf`
- `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`
- `README.md`

## Local acceptance commands

Run in this order:

1. `./scripts/ci.sh`
2. `./scripts/test.sh`
3. `make test`

Documented commands from README/docs:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -e '.[dev]'`
- `ruff check .`
- `pytest`
- `quarto render book --to html`

Sandbox reconciliation:

- If Quarto execution fails due restricted local runtime/socket permissions, run:
  - `XDG_CACHE_HOME=/tmp/xdg-cache quarto render book --to html --no-execute`
- Treat CI `build-html` as the executable render authority for fully executed notebooks.

## Branch and commit policy

- Branch: `main` (for this realignment pass)
- Commit format: `[M0.REALIGN.END] <imperative summary>`
- No PRs
- No force-push

## Done criteria for next milestone

- Spec-to-repo mapping is updated and current.
- Drift items marked `Diverged` have an owner and concrete recovery task.
- Local acceptance commands are runnable or explicitly reconciled in docs.
