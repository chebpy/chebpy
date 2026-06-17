---
description: Run the Rhiza code-quality gate and score the repo (lint, types, docs, deps, security, tests)
---

Assess the quality of this repo against Rhiza standards. Follow the
command-execution policy: always prefer `make <target>`; never invoke
`.venv/bin/...` directly. Run the gates in order — cheapest checks first so fast
failures surface before the slow test suite — and collect results:

1. `make fmt` — pre-commit hooks + linting (ruff format/check, markdownlint, bandit, actionlint, …)
2. `make typecheck` — static type checking (`ty`, and `mypy --strict` if configured) over `src/`
3. `make docs-coverage` — docstring coverage (interrogate) over `src/`
4. `make deptry` — unused/missing/misplaced dependency analysis
5. `make security` — pip-audit + bandit scans
6. `make validate` — validate project structure against the Rhiza template (`.rhiza/template.yml`)
7. `make test` — full test suite **with** its coverage gate (slowest, run last)

Guidelines:

- Run all gates even after an early failure, so the full picture is visible
  rather than stopping at the first red.
- If something fails, show the relevant output, diagnose the root cause, and
  propose (or apply, if clearly correct, low-risk, **and** the fix is in a
  locally-owned file per the scoping rule below) a fix.
- If `$ARGUMENTS` is non-empty, scope the assessment to that path or topic
  instead of the whole repo.
- End with a concise PASS/FAIL summary per gate.

**Coverage expectation.** `make test` enforces a coverage gate
(`COVERAGE_FAIL_UNDER`, default 90%; many projects raise it to 100%). Treat
anything below the configured threshold on locally-owned `src/` as a gap to
flag, not an acceptable baseline. When scoring the test-coverage subcategory,
the configured threshold is the bar for a 10; report uncovered lines
(`file:line`) and the test that would close each.

**`make validate`.** A failure means this repo has drifted from the Rhiza
template (a synced file edited locally, or a missing/extra file). That is
in-scope: fix it by re-syncing from Rhiza or by adjusting `.rhiza/template.yml`,
not by editing the synced artifact in place.

Then report:

- A pass/fail summary per step.
- Failures grouped by file, with the specific rule/error and line.
- A prioritized list of what to fix first (blocking errors before style nits).

Then analyse the repo and give marks on a scale of 1 to 10 for all relevant
subcategories. Pick the subcategories that fit what you actually observe — e.g.
linting/style, type safety, test pass rate, test coverage & depth, code
structure & readability, documentation, dependency & security hygiene, CI/tooling
health. For each: the score, a one-line justification grounded in evidence from
the checks above (and a quick look at the code where needed), and what would
raise it. Close with an overall score and the single highest-leverage
improvement.

**Scope the scorecard to locally-owned items — not what the mother repo (Rhiza)
owns.** This project syncs its dev infrastructure from `jebel-quant/rhiza`; see
`CLAUDE.md` for the authoritative split and the `files:` block of
`.rhiza/template.lock` for the machine-generated list of synced files. Score
only what this repo actually controls — `src/`, `tests/`, `pyproject.toml`,
`README.md`, project-specific docs, `.rhiza/template.yml`, and any
locally-hardened config. Do **not** let Rhiza-managed files (the
`.github/workflows/*`, `Makefile`, `.pre-commit-config.yaml`, `pytest.ini`,
`ruff.toml`, the typecheck/mutation/fuzzing targets, etc.) drive the marks — a
gap there is fixed upstream in Rhiza, not here. If a relevant signal is
Rhiza-owned, note it as "upstream/out-of-scope" rather than scoring it against
this repo.

Then, from the scorecard above, identify **actionable issues to improve the
score** — one per subcategory scoring below 10 (skip any that are maxed). For
each, give: a concrete title, the subcategory and current→target score it moves,
the specific file(s)/lines or config to change, and a crisp acceptance criterion
("done when…"). Keep them in-scope (locally-owned, per the scoping rule above) —
flag anything Rhiza-owned as upstream rather than listing it as a local action.
Order them by leverage (biggest score gain for least effort first). This is a
list of recommendations only — do not create GitHub issues or change code unless
I explicitly ask.

If everything passes, say so plainly — but still produce the 1–10 subcategory
marks. Do not fix anything unless I ask — this command only assesses.
