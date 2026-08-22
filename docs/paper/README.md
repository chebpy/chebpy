# LaTeX Paper

This folder is where the `paper` bundle expects your LaTeX sources. `make paper`
compiles them to a PDF with `latexmk`, and `make paper-clean` removes the build
artifacts.

## Layout

Put the root document at the top level of this folder. Chapters, figures and
bibliography files can live in subdirectories — only the top level is searched for a
root document, because a chapter is an `\input`, not something to compile on its own.

```
docs/paper/
├── main.tex          <- the root document
├── references.bib
├── chapters/
│   └── introduction.tex
└── figures/
```

## Which file gets compiled

The root document is chosen by name, in this order:

1. `main.tex`
2. `paper.tex`
3. the first `.tex` file alphabetically

A folder holding one `.tex` file is unambiguous whatever it is called, which is the
common case. Name it `main.tex` if you have several and want to be explicit.

## Targets

| target | does |
| --- | --- |
| `make paper` | `latexmk -pdf -bibtex -interaction=nonstopmode` on the root document |
| `make paper-clean` | `latexmk -C` — removes the PDF and every auxiliary file |

`latexmk` reruns pdflatex and bibtex until the cross-references and citations
converge, so one invocation is enough however many passes the document needs. Both
targets run with this folder as the working directory, so `\input` paths are relative
to it and the auxiliary files land beside the source rather than at the repository
root.

## Requirements

A LaTeX distribution providing `latexmk` — [MacTeX](https://www.tug.org/mactex/) on
macOS, [TeX Live](https://www.tug.org/texlive/) elsewhere. Without it both targets
skip with that as the reason rather than failing, so a contributor who does not build
the paper is not blocked by it. Pass `--strict` to turn the skip into a failure where
the paper *must* build, such as in CI.

## Configuration

The folder is `docs/paper` by default. Point the tasks somewhere else with a
`paper-folder` setting:

```toml
[tool.rhiza-task]
paper-folder = "manuscript"
```

## Continuous integration

The `github-paper` bundle adds a workflow that compiles the paper and publishes the
PDF as a build artifact. It triggers only on changes under `docs/paper/**`, so it
costs nothing until there is a paper to build.

The **durable** copy comes from the book rather than that artifact, which expires after
30 days. `paper` is a prerequisite of `book`, and this folder sits inside the docs tree,
so mkdocs sweeps the compiled PDF up as a site asset at a stable URL. Link it from the
nav to make it reachable:

```yaml
nav:
  - Paper: paper/main.pdf
```

### If your repository has a `paper` branch

The workflow used to push the PDF to an orphan `paper` branch as well. **It no longer
does**, and a repository that still has the branch will see a warning naming it.

The push was removed because git refs are paths: `refs/heads/paper` cannot exist while
`refs/heads/paper/anything` does. So opening a `paper/overview` topic branch — the most
natural convention for the very feature this bundle serves — broke the push outright, and
it stayed broken after that topic branch was merged, until somebody also deleted it.

Nothing needs to change on your side unless something *reads* that branch — a badge, a
Pages source, a direct link. Point those at the PDF in the built site instead, then delete
the branch, which is now nobody's job to update:

```bash
git push origin --delete paper
```

Leaving it in place is harmless except that it holds a PDF frozen at the last run before
this change, with nothing indicating so.
