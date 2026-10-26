<!--
Thanks for contributing!

Every section below is optional guidance, not a form to fill in completely.
Delete what does not apply; keep the checklist.
-->

#### Reference issues/PRs

<!--
Example: Fixes #1234. See also #3456.

Use a closing keyword (Fixes, Closes, Resolves) so the linked issue closes
automatically when this PR is merged:
https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
If no issue exists, open one first or explain the motivation below.
-->

#### What does this implement/fix? Explain your changes.

<!--
A clear and concise description of what you changed and why. If the change is
a bug fix, say what the observed behaviour was and what it is now.
-->

#### Does your contribution introduce a new dependency? If yes, which one?

<!--
Only relevant if you changed the project manifest (pyproject.toml, Cargo.toml,
go.mod), a GitHub Actions pin or a pre-commit hook. We try to keep the
dependency set small; say why the new one is needed and whether it is
optional.
-->

#### What should a reviewer concentrate their feedback on?

<!--
Particularly useful for a PR that is still in development: point reviewers
at the parts that are ready for comments. Bullets (* or -) and filled
checkboxes [x] work well here.
-->

#### Did you add any tests for the change?

<!--
New behaviour should come with a test that fails without the change, so
later edits cannot reintroduce the same bug. If the change is not testable,
say why.
-->

#### Any other comments?

<!--
Trade-offs you considered, follow-up work you deliberately left out,
anything a reviewer would otherwise have to guess.
-->

#### PR checklist

<!--
Go through the list below. Remove points that do not apply.
-->

##### For all contributions

- [ ] The PR title follows [Conventional Commits](https://www.conventionalcommits.org/)
      (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, ...). It becomes
      the squash-merge commit and the changelog entry.
- [ ] `make fmt` and `make test` pass locally.
- [ ] No hand edits to template-owned files. Anything listed in
      `.rhiza/template.lock` is overwritten by the next sync; propose the change
      upstream in [Jebel-Quant/rhiza](https://github.com/Jebel-Quant/rhiza) or add the
      path to `exclude:` in `.rhiza/template.yml`.

##### For user-visible changes

- [ ] Documentation (README, docstrings, docs pages) is updated to match.
- [ ] Breaking changes are marked with `!` in the title or a `BREAKING CHANGE:` footer.

<!--
Thanks again for contributing!
-->
