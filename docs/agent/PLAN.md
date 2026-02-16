# Recovery Plan (M0.REALIGN.END)

## Now

- Stabilize docs topology on `main`: keep canonical spec files at `docs/diff_epi.pdf`, `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`, and `README.md`.
- Remove stale/broken doc references and restore required build instructions (`docs/BUILD.md`) so README links resolve.
- Make local acceptance gates runnable in-order with repo-local wrappers: `./scripts/ci.sh`, `./scripts/test.sh`, `make test`.

## Next

- Close M3 bookkeeping drift: each `Diverged` item gets an explicit owner and one concrete recovery task in agent docs.
- Confirm gate sequence and documented setup paths stay aligned (`README.md`, `AGENTS.md`, `docs/BUILD.md`).
- Keep CI event-driven on push/PR and add cron/manual safety-net runs to detect silent drift.

## Later

- Implement minimum M4 recovery slice: conditional-flow NPE path + smoke calibration diagnostic.
- Extend milestone tracking only after gate stability and docs consistency remain green for repeated runs.
- Polish publication-facing content once recovery milestones are complete and reproducible.
