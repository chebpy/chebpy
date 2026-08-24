# LaTeX Paper

This folder is where the `paper` bundle expects your LaTeX sources. `make paper`
compiles them to a PDF, and `make paper-clean` removes the build artifacts.

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
| `make paper` | compiles the root document to a PDF beside its source |
| `make paper-clean` | removes each document's PDF and auxiliary files |

The engine reruns the TeX pass and bibtex until the cross-references and citations
converge, so one invocation is enough however many passes the document needs. `paper`
runs with this folder as the working directory, so `\input` paths are relative to it
and the output lands beside the source rather than at the repository root — which is
what lets the book publish the PDF with no copy step.

`paper-clean` is scoped by document stem: `paper.tex` authorises deleting `paper.pdf`
and `paper.log`, while a `figures/diagram.pdf` you committed has no `.tex` beside it
and survives.

## Requirements

[tectonic](https://tectonic-typesetting.github.io/), a single binary that resolves
the packages a document cites out of its own web bundle and caches them — so there is
no TeX distribution to install and no package list to keep in step with your
`\usepackage` lines. A cold cache needs the network; after that it does not.

Without tectonic on `PATH` both targets skip, with that as the reason, rather than
failing — so a contributor who does not build the paper is not blocked by it. Pass
`--strict` to turn the skip into a failure where the paper *must* build, such as in
CI, which is what the shipped pipelines do.

## Configuration

The folder is `docs/paper` by default. Point the tasks somewhere else with a
`paper-folder` setting:

```toml
[tool.rhiza-task]
paper-folder = "manuscript"
```

## Continuous integration

The `github-paper` bundle adds a workflow that compiles the paper and publishes it. It
triggers only on changes under `docs/paper/**`, so it costs nothing until there is a
paper to build. It installs tectonic itself; the compile is the same `paper` task you
run locally, under `--strict`, so a runner that never got the engine fails instead of
reporting a skipped build as success.

The PDF is published three ways, which is deliberate — they fail differently:

1. **The run artifact**, named `paper`. Immediate, and it expires after 30 days.
2. **The book.** `paper` is a prerequisite of `book` and this folder sits inside the
   docs tree, so mkdocs sweeps the PDF up as a site asset at a stable URL. Link it
   from the nav to make it reachable:

   ```yaml
   nav:
     - Paper: paper/main.pdf
   ```

3. **The `paper` branch**, which holds the compiled PDF and a generated `README.md`
   explaining what the branch is — and nothing else. Pushed on every default-branch run,
   never from a pull request. This is the copy you can link without building the site and
   without an unexpired run.

   The README is written by the workflow, so anything you commit there by hand is
   overwritten on the next run. It deliberately carries no run number or timestamp: that
   would make the file differ every time, and the branch would collect a commit per push
   whether or not the paper changed. The source commit each PDF was built from is named in
   the commit message instead.

   For the same reason the compile pins `SOURCE_DATE_EPOCH` to the source commit's time.
   tectonic otherwise stamps the PDF `/ID` from the build time, so an unchanged document
   compiles to different bytes on every run — which would commit every time regardless of
   the README. With it, a rebuild of a revision you have already published is a no-op.

   Nothing needs to be tracked for this to work — the template gitignores
   `docs/paper/*.pdf`. If you commit your PDF anyway, the publish still works; it discards
   the freshly compiled copy from the working tree after staging it, which is the only way
   to switch branches with a modified tracked file in the way.

The GitLab pipeline publishes the first two. It does not push the branch: that needs a
token `CI_JOB_TOKEN` cannot stand in for, and this template sets up no project secret.

### If your repository has a `paper/<topic>` branch

Then the branch publish cannot work, and the workflow says so instead of failing
obscurely. Git refs are paths: `refs/heads/paper` cannot exist while
`refs/heads/paper/overview` does — the most natural branch convention for the very
feature this bundle serves. A preflight step lists the colliding ref and fails with it
named; git's own error on that push names neither branch.

Your options, in the order most repositories want them:

1. Rename the topic branch — `paper/overview` → `paper-overview`.
2. Leave it, and take the other two copies. The compile and the artifact upload run
   before the preflight, so the PDF is still attached to the failed run and still
   published by the book. Only the branch step is red.
