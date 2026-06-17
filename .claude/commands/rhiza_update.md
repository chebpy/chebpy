---
description: Update the pinned Rhiza version in .rhiza/template.yml, sync, resolve conflicts, and verify
---

Update this repo's Rhiza template to a newer release: bump the pin in
`.rhiza/template.yml`, run the sync, resolve every conflict, verify the quality
gates, and open a PR. Follow the command-execution policy: always prefer
`make <target>`; never invoke `.venv/bin/...` directly.

`$ARGUMENTS` may name a target version (e.g. `v0.19.1`). If empty, target the
**latest** release of the upstream template repo.

## 1. Determine current and target versions

- Read `.rhiza/template.yml`: the `repository:` field is the upstream template
  repo (usually `jebel-quant/rhiza`) and `ref:` is the currently pinned version.
- Resolve the target version:
  - If `$ARGUMENTS` names a version, use it (verify the tag exists:
    `gh api repos/<repository>/git/ref/tags/<ref>`).
  - Otherwise get the latest release:
    `gh release view --repo <repository> --json tagName,publishedAt`.
- If `ref:` already equals the target, report "already up to date" and stop.
- Briefly summarize what's between the two versions when it's cheap to do so
  (`gh release view`/release notes), so the reviewer knows what's landing.

### The Rhiza CLI pin (`.rhiza/.rhiza-version`) — ask before changing it

`.rhiza/.rhiza-version` separately pins the **Rhiza CLI** — the `rhiza` package on
PyPI that `make sync` runs as `uvx "rhiza==<version>"`. It is independent of the
template `ref:` above, and there is no `$ARGUMENTS` override for it: always target
the **latest** published version.

- Read the current pin from `.rhiza/.rhiza-version`.
- Resolve the latest published version on PyPI:
  `curl -s https://pypi.org/pypi/rhiza/json` and read `.info.version`.
- If the pin already equals the latest, there is nothing to do — leave it.
- If a newer version is available, **ask the user whether to update the CLI** to
  that latest version (state current → latest). Only bump `.rhiza/.rhiza-version`
  if they agree; if they decline, leave the file untouched and proceed with the
  template sync alone.

## 2. Bump the pin(s) and commit (the tree must be clean to sync)

- `make sync` refuses to run on a dirty tree, so the bump lands first.
- Branch off the default branch (don't work on `main`/`master` directly):
  `git checkout -b sync/rhiza-<target>`.
- Edit `ref:` in `.rhiza/template.yml` to the target version. If the user agreed
  to a CLI bump above, also set `.rhiza/.rhiza-version` to the latest PyPI version
  in the same step, so `make sync` runs with the chosen CLI.
- Commit the pin change(s) (e.g. `Chore: bump rhiza template ref <old> → <target>`,
  noting the CLI bump in the message when one was made).

## 3. Sync

- Run `make sync` (it invokes `rhiza sync`). Expect it to either complete
  cleanly or report conflicts. It writes the refreshed `.rhiza/template.lock`.
- If it completes with no conflicts, skip to step 5.

## 4. Resolve every conflict

The sync is a 3-way merge. Two kinds of leftovers can appear — handle both, and
finish with **zero** `*.rej` files and **zero** conflict markers
(`<<<<<<<` / `=======` / `>>>>>>>`) anywhere tracked
(`git grep -lE '^(<<<<<<<|=======|>>>>>>>)'`).

**`*.rej` files (rejected hunks).** The 3-way merge often *already applied* a
hunk and still drops a duplicate `.rej`. For each, verify whether the change is
already present in the file (the added `+` lines exist; no conflict markers
remain). If it is, the `.rej` is spurious — delete it. If a hunk genuinely did
not apply, apply it by hand, then delete the `.rej`.

**Conflict-marked files.** Resolve by the ownership rule (see `CLAUDE.md` and the
`files:` block of `.rhiza/template.lock` for the authoritative managed-file list):

- **Rhiza-managed files** (the `.github/workflows/*`, `Makefile`,
  `.pre-commit-config.yaml`, `pytest.ini`, the `.rhiza/` engine, etc.): take the
  **incoming/upstream** side — these are owned by the template and should match
  it (`git checkout --theirs -- <file>` then `git add`).
- **Locally-owned or locally-hardened files** (notably `ruff.toml`, plus
  `pyproject.toml`, `README.md`, `src/`, your `tests/`): **merge by hand** —
  keep the local intent (e.g. stricter lint rules) while folding in genuine
  upstream additions, and make the result internally coherent (dedupe, drop
  comments that now contradict the config).

Validate every touched workflow/YAML still parses before moving on.

## 5. Verify the gates and fix fallout

A version bump can tighten the gates (new lint rules, `mypy --strict`, expanded
docs-coverage scope, etc.) and surface pre-existing issues. Run them and get
them green:

1. `make fmt` — pre-commit + lint
2. `make typecheck`
3. `make docs-coverage`
4. `make deptry`
5. `make security`
6. `make test`

**Scope your fixes.** Fix issues only in **locally-owned** files (`src/`,
`tests/`, `pyproject.toml`, locally-hardened config). If a gate fails because of
a **Rhiza-managed** file, that is an upstream problem: fix it in
`jebel-quant/rhiza` and bump again — do **not** edit the synced artifact in
place. Call out any such upstream-owned failure explicitly rather than papering
over it locally.

### Configure the CI variables/secrets the synced workflows need (fuzzing & mutation)

If this sync added or updated the fuzzing/mutation workflows
(`.github/workflows/rhiza_fuzzing.yml`, `.github/workflows/rhiza_mutation.yml`),
make sure the configuration they read is present. These are **GitHub Actions
repository variables and secrets** — *not* local env vars or files — set under
**Settings → Secrets and variables → Actions** (or via `gh`), and documented in
`.github/CONFIG.md`. Skip this step entirely if neither workflow is present.

Inspect what is already set (presence only — secret values are never readable):

- `gh variable list`
- `gh secret list`

**Mutation** (`rhiza_mutation.yml`) reads:

- `MUTATION_ENABLED` (variable) — must be `true` for the mutation gate/badge to run.
- `GH_PAT` (secret) — git auth for installing private dependencies.
- `UV_EXTRA_INDEX_URL` (secret) — extra package index URL (with credentials) for private deps.

**Fuzzing** (`rhiza_fuzzing.yml`) needs no user configuration — it uses the
automatic `GITHUB_TOKEN`, so do not prompt for any fuzzing secret.

For each of the above that is **missing**, **ask the user** whether to set it and
for its value; set only the ones they provide:

- variables: `gh variable set MUTATION_ENABLED --body true`
- secrets:   `gh secret set GH_PAT` (let `gh` read the value from stdin — never
  echo a secret value into the transcript or a commit).

Leave anything the user declines unset, and note it in the PR body so the reviewer
knows the corresponding workflow step will be skipped or fail until it is configured.

## 6. Commit, push, open a PR

- Commit the resolution and any in-scope fixes with clear messages (one logical
  change per commit: the conflict resolution, then each gate fix).
- Push the branch and open a PR (`gh pr create`) titled for the bump, e.g.
  `Chore: sync Rhiza template <old> → <target>`. In the body, summarize how each
  conflict was resolved, list any gate fallout you fixed, and flag anything that
  needs an **upstream** fix in Rhiza.
- Report a concise per-gate PASS/FAIL summary. If the workflow files changed,
  note that pushing them needs a token with the `workflow` scope.

Do not merge the PR. Stop after it is open and summarize what landed and what
(if anything) is blocked on an upstream Rhiza change.
