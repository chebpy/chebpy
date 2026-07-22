# GitHub Actions Configuration

This document describes the secrets used by the Rhiza-provided GitHub Actions workflows
(`.github/workflows/rhiza_*.yml`) and how to configure them.

## PAT_TOKEN (template sync)

The sync workflow (`.github/workflows/rhiza_sync.yml`) keeps your repository up to date with the
upstream Rhiza template. It commits the synced files directly to a branch or opens a pull request.

By default the workflow authenticates with the automatic `github.token`. That token **cannot push
changes to files under `.github/workflows/`** — GitHub rejects such pushes unless the token has the
`workflow` scope. Since template syncs regularly update workflow files, you should configure a
Personal Access Token (PAT) with that scope and store it as a repository secret named `PAT_TOKEN`.

If `PAT_TOKEN` is not configured, the workflow falls back to `github.token` and prints a warning.
Syncs that touch only non-workflow files will still succeed.

### Creating the token

**Fine-grained PAT** (recommended):

1. Go to **Settings → Developer settings → Fine-grained tokens → Generate new token**
   (<https://github.com/settings/personal-access-tokens/new>).
2. Restrict **Repository access** to the repository (or repositories) using Rhiza.
3. Under **Repository permissions**, grant:
   - **Contents**: Read and write
   - **Workflows**: Read and write
   - **Pull requests**: Read and write (needed for the scheduled sync-PR mode)
4. Generate the token and copy it.

**Classic PAT** (alternative):

1. Go to **Settings → Developer settings → Tokens (classic) → Generate new token**.
2. Select the `repo` and `workflow` scopes.
3. Generate the token and copy it.

### Storing the secret

In the repository that consumes Rhiza:

1. Go to **Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `PAT_TOKEN`
3. Value: the token created above.

Or with the GitHub CLI:

```bash
gh secret set PAT_TOKEN
```

A PAT expires; when sync pushes start failing with a `refusing to allow ... workflow` error,
regenerate the token and update the secret.

## Release workflow secrets (optional)

The release workflow (`.github/workflows/rhiza_release.yml`) supports additional secrets, all
optional depending on which release features you use:

| Secret | Purpose |
| --- | --- |
| `PYPI_TOKEN` | Publish the built package to PyPI. Not needed when using trusted publishing (OIDC). |
| `GH_PAT` | Git authentication for installing private dependencies during the release build. |
| `UV_EXTRA_INDEX_URL` | Extra package index URL (with credentials) for private dependencies. |

`GITHUB_TOKEN` is provided automatically by GitHub Actions and needs no configuration.
