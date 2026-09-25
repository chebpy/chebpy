# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.11.0] - 2026-08-11

### Bug Fixes
- Drop License:: classifier test that contradicts PEP 639
- Move bumpversion config into pyproject.toml
- Describe the failure when chebfun() cannot build a constant
- Drop the stale self_empty TODO and unblock radon
- Describe unconvertible chebfun() inputs and simplify equifun (#462)

### Documentation
- Document the test-layout opt-out and correct stale CLAUDE.md claims
- Add LaTeX paper describing ChebPy, its algorithms and Chebyshev approximation
- Feature the LaTeX paper in the book

### Maintenance
- Bump rhiza template version to v0.10.9
- Sync rhiza templates to v0.10.9
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump idna from 3.11 to 3.15
- Chore(deps)(deps): bump pymdown-extensions from 10.21.2 to 10.21.3
- Update rhiza template names to github-prefixed variants
- Sync rhiza templates and restore missing workflow files
- Remove redundant noqa comments cleaned up by ruff
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Chore(deps)(deps): bump the github-actions group with 4 updates
- Chore(deps)(deps): bump starlette from 0.52.1 to 1.0.1
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump python-multipart from 0.0.30 to 0.0.31
- Chore(deps)(deps): bump starlette from 1.0.1 to 1.3.1
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group across 1 directory with 2 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Update rhiza to v1.1.2 (#425)
- Chore(deps-dev)(deps-dev): bump ruff in the python-dependencies group
- Close coverage gap in compactfun/singfun/trigtech (#418) (#421)
- Reduce complexity & raise chebfun MI to A (#419) (#422)
- Chore(deps)(deps): bump pillow from 12.2.0 to 12.3.0
- Chore(deps)(deps): bump matplotlib in the python-dependencies group
- *(pyproject)* Drop License classifiers in favour of SPDX (PEP 639)
- Chore(deps)(deps): bump the github-actions group with 17 updates
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates
- Bump rhiza to v1.2.5
- Apply rhiza sync v1.2.5
- Drop files removed from rhiza v1.2.5
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 3 updates
- Bump rhiza to v1.3.2
- Apply rhiza sync v1.3.2
- Cover the Floater-Hormann path and enforce 100% coverage
- Select the github-project profile in template.yml
- Chore(deps)(deps): bump pymdown-extensions from 10.21.3 to 11.0.1
- Add github-paper template to rhiza config
- Apply rhiza sync for github-paper template

### Other Changes
- Docs correction (#384)
- Merge pull request #386 from chebpy/dependabot/uv/python-dependencies-189a206c4b
- Merge pull request #385 from chebpy/dependabot/github_actions/github-actions-937d73b4db
- Merge pull request #389 from chebpy/dependabot/uv/python-dependencies-f78d620da7
- Merge pull request #388 from chebpy/dependabot/github_actions/github-actions-8abaa2cbc6
- Merge pull request #390 from chebpy/dependabot/github_actions/github-actions-bcb0c4251a
- Merge pull request #391 from chebpy/dependabot/uv/python-dependencies-dc63e5bab1
- Merge pull request #392 from chebpy/dependabot/uv/idna-3.15
- Merge pull request #393 from chebpy/dependabot/uv/pymdown-extensions-10.21.3
- Merge remote-tracking branch 'origin/master' into rhiza
- Merge pull request #394 from chebpy/chore/rhiza-v0.10.9
- Fix README pip install command
- Merge pull request #398 from chebpy/codex/create-branch-fix-pip-install-readme
- Merge pull request #403 from chebpy/dependabot/github_actions/github-actions-22b43d3523
- Merge branch 'master' into dependabot/uv/python-dependencies-3035e4b055
- Merge pull request #402 from chebpy/dependabot/uv/python-dependencies-3035e4b055
- Merge pull request #404 from chebpy/dependabot/uv/starlette-1.0.1
- Merge pull request #406 from chebpy/dependabot/uv/python-dependencies-3985e15cef
- Merge branch 'master' into dependabot/github_actions/github-actions-0321e4ed66
- Merge pull request #405 from chebpy/dependabot/github_actions/github-actions-0321e4ed66
- Merge pull request #407 from chebpy/dependabot/uv/python-dependencies-f53abc96ed
- Merge pull request #408 from chebpy/dependabot/uv/python-multipart-0.0.31
- Merge pull request #409 from chebpy/dependabot/uv/starlette-1.3.1
- Add Claude commands from shrinkage repo
- Merge pull request #410 from chebpy/add-claude-commands
- Merge pull request #411 from chebpy/dependabot/uv/python-dependencies-4696eb17b1
- Sync Rhiza template v0.10.9 → v0.19.4 (#412)
- Merge pull request #413 from chebpy/dependabot/uv/python-dependencies-9cdc7840e2
- Fix zensical docs homepage rendering (README as raw HTML block) (#427)
- Merge pull request #429 from chebpy/dependabot/uv/python-dependencies-ae72112a2f
- Merge pull request #430 from chebpy/dependabot/uv/python-dependencies-d6405df956
- Merge pull request #433 from chebpy/dependabot/uv/python-dependencies-1bfa08d1c2
- Merge branch 'master' into dependabot/uv/pillow-12.3.0
- Merge pull request #435 from chebpy/dependabot/uv/pillow-12.3.0
- Add equifun for equispaced sample data (#399)
- Initial plan
- Remove legacy devcontainer button
- Merge pull request #437 from chebpy/copilot/remove-legacy-devcontainer-button
- Merge pull request #440 from chebpy/remove-license-classifier-test
- Merge branch 'master' into chore/python-version-classifiers
- Merge pull request #439 from chebpy/chore/python-version-classifiers
- Merge pull request #442 from chebpy/dependabot/uv/python-dependencies-b3e9c802ec
- Merge branch 'master' into dependabot/github_actions/github-actions-35219769aa
- Merge pull request #441 from chebpy/dependabot/github_actions/github-actions-35219769aa
- Merge pull request #443 from chebpy/rhiza_v1.2.5_20260730
- Merge pull request #445 from chebpy/dependabot/uv/python-dependencies-e2a29b3896
- Merge pull request #444 from chebpy/dependabot/github_actions/github-actions-908328dd72
- Merge pull request #446 from chebpy/rhiza_v1.3.2_20260804
- Merge pull request #452 from chebpy/fix/quality-findings
- Merge pull request #453 from chebpy/chore/rhiza-profile-github-project
- Merge pull request #458 from chebpy/fix/quality-456-457
- Merge branch 'master' into fix/quality-455-error-message
- Merge pull request #460 from chebpy/dependabot/uv/pymdown-extensions-11.0.1
- Merge branch 'master' into fix/quality-455-error-message
- Merge pull request #459 from chebpy/fix/quality-455-error-message
- Remove DEVCONTAINER badge from README
- Merge pull request #463 from chebpy/tschm-patch-1
- Merge pull request #464 from chebpy/rhiza_github-paper
- Merge pull request #465 from chebpy/paper/chebpy-overview
- Merge pull request #466 from chebpy/docs/feature-paper-in-book

