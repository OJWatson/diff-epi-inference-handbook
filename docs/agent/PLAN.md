# Recovery Plan (M0.REALIGN.END)

## Now

- Stabilize docs topology on `main`: keep canonical spec files at `docs/diff_epi.pdf`, `docs/DIFF_EPI_DEVELOPMENT_PLAN.md`, and `README.md`.
- Remove stale/broken doc references and restore required build instructions (`docs/BUILD.md`) so README links resolve.
- Make local acceptance gates runnable in-order with repo-local wrappers: `./scripts/ci.sh`, `./scripts/test.sh`, `make test`.

## Drift Items

| Item | Status | Owner | Recovery task |
| --- | --- | --- | --- |
| Roadmap/status references existed in plan but files were missing | Recovered | `@maintainers` | Maintain `ROADMAP.md` and `docs/STATUS.md` as source-of-truth trackers on `main`. |
| Local Quarto execution may fail under restricted runtime/cache/log permissions | Recovered | `@maintainers` | Keep runtime-safe env command documented in `AGENTS.md` and `docs/BUILD.md`. |
| Spec-to-repo mapping was implicit and scattered | Recovered | `@maintainers` | Maintain `docs/SPEC_TO_REPO_MAP.md` alongside milestone updates. |

## Next

- Keep recovery item ownership current and ensure each has one concrete maintenance task.
- Confirm gate sequence and documented setup paths stay aligned (`README.md`, `AGENTS.md`, `docs/BUILD.md`).
- Keep CI event-driven on push/PR and add cron/manual safety-net runs to detect silent drift.

## Later

- Implement minimum M4 recovery slice: conditional-flow NPE path + smoke calibration diagnostic.
- Extend milestone tracking only after gate stability and docs consistency remain green for repeated runs.
- Polish publication-facing content once recovery milestones are complete and reproducible.
