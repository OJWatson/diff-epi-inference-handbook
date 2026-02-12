# GitHub Pages publishing

This repo includes a GitHub Actions workflow to publish the rendered Quarto HTML
book to the `gh-pages` branch.

## Workflow

Workflow file: `.github/workflows/pages.yml`

Triggers:

- Manual: `workflow_dispatch`
- Automatic: `push` to tags matching `v*` (e.g. `v0.1.0`)

The workflow renders the book (`quarto render book --to html`) and deploys the
resulting `book/_book` directory to the `gh-pages` branch.

## Repository settings (one-time)

In GitHub repo settings:

1. Settings → Pages
2. Source: **Deploy from a branch**
3. Branch: `gh-pages` (root)

## Notes

- Publishing on version tags avoids updating the public site on every commit.
- If you want continuous publishing from `main`, change the workflow trigger to
  `push: { branches: [main] }` once you are happy with the Pages setup.
