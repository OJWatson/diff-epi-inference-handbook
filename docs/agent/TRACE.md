# TRACE

This file records milestone ticks with links to evidence (commands/logs).

TRACE entry semantics (Portfolio OS):
- Every entry must include a `task_id` token.
- Optionally record either:
  - `branch` + `commit_parent` + `timestamp` (all known before creating the task commit), and/or
  - `head_after` (the task commit hash, if recorded in a follow-up commit).

- task_id=M4.0 branch=main commit_parent=1ae4b9fe62260444e43a43873a5f4ba5b1473892 timestamp=2026-02-14T06:38:11Z
  - Summary: Decide modern-SBI dependency strategy: add optional JAX/modern-sbi extras; move CI to uv and add optional BlackJAX job
  - Gates: `uv run pytest -q`; `quarto render book --to html`

- task_id=M3.3 branch=main commit_parent=a052dccdab657aa3feadf32b82cce24423389aee timestamp=2026-02-13T20:04:15Z
  - Summary: Update STATUS.md to record M3 completion and advance milestone to M4
  - Acceptance log: /home/kana/.openclaw/workspace/portfolio/reports/logs/diff-epi-inference-handbook/20260213T200415Z_M3.3.log
  - Gates: `uv run pytest -q`; `quarto render book --to html`

- task_id=M3.2 branch=main commit_parent=6a2c2af7396ea63e2f9ae89200304fee0d1c67af timestamp=2026-02-13T19:30:49Z head_after=c3ec60a
  - Summary: Add minimal diagnostics per ABC/SMC-ABC/synthetic likelihood
  - Acceptance log: /home/kana/.openclaw/workspace/portfolio/reports/logs/diff-epi-inference-handbook/20260213T193049Z_M3.2.log
  - Gates: `uv run pytest -q`; `quarto render book --to html`
