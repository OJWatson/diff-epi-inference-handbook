# TRACE

This file records milestone ticks with links to evidence (commands/logs).

TRACE entry semantics (Portfolio OS):
- Every entry must include a `task_id` token.
- Optionally record either:
  - `branch` + `commit_parent` + `timestamp` (all known before creating the task commit), and/or
  - `head_after` (the task commit hash, if recorded in a follow-up commit).

- task_id=M4.0 branch=main commit_parent=9695de20bba1f7cc8cae1f199fb989dc8bf7c770 timestamp=2026-02-15T00:04:14Z
  - Summary: Force CI HTML/PDF builds to execute (including JAX cells) via a Quarto `ci` profile; ignore Quarto *_files artifacts
  - Gates: `uv run pytest -q`; `uv run quarto render book --to html --profile ci`

- task_id=M4.3 branch=main commit_parent=84e4b7b5f37cdb2bbd5e1ce4c5f9eb4cb054a34d timestamp=2026-02-14T16:31:30Z head_after=4a19f36
  - Summary: Replace the Modern SBI chapter’s conditional-Gaussian demo with the package flow helper
  - Gates: `uv run pytest -q`; `quarto render book --to html`

- task_id=M4.2 branch=main commit_parent=29de81b8b1d64f687bcb041c16b0d4f2a6be7a7f timestamp=2026-02-14T16:31:10Z head_after=84e4b7b
  - Summary: Add a tiny JAX-based training loop example (behind `modern-sbi` extra)
  - Gates: `uv run pytest -q`; `quarto render book --to html`

- task_id=M4.1 branch=main commit_parent=89b492cb4bf86a035f392c0a19e9e27682891a61 timestamp=2026-02-14T06:54:37Z head_after=29de81b
  - Summary: Add a minimal conditional affine flow (conditional diagonal Gaussian) with closed-form fit + tests
  - Gates: `uv run pytest -q`; `quarto render book --to html`

- task_id=M4.0 branch=main commit_parent=1ae4b9fe62260444e43a43873a5f4ba5b1473892 timestamp=2026-02-14T06:38:11Z head_after=5b84b4a
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
