# STATUS

Last updated: 2026-02-17

## Current Milestone

- `milestone`: `M8` (Polish + publication)
- `state`: `Complete`
- `nextStep`: Maintain green CI and update content incrementally without widening dependency/runtime scope.

## Gate Health

- `./scripts/ci.sh`: passing
- `./scripts/test.sh`: passing
- `make test`: passing
- `quarto render book --to html --profile ci`: passing with runtime-safe env (`XDG_CACHE_HOME`, `XDG_DATA_HOME`, `JUPYTER_RUNTIME_DIR`)

## Drift Register

| Item | Status | Owner | Recovery task |
| --- | --- | --- | --- |
| Missing roadmap file referenced by dev plan | Recovered | `@maintainers` | Added `ROADMAP.md` and aligned milestone states with code/tests. |
| Missing `docs/STATUS.md` referenced by dev plan | Recovered | `@maintainers` | Added this file with explicit `milestone/state/nextStep` and gate status. |
| Local Quarto execution in restricted runtime (cache/log paths) | Recovered | `@maintainers` | Document and use runtime-safe env overrides in `AGENTS.md` and `docs/BUILD.md`. |
