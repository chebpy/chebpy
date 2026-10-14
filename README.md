# The compiled paper

This branch is written by a machine. It carries the PDF built from the LaTeX
sources of [chebpy/chebpy](https://github.com/chebpy/chebpy)
and nothing else: no code, no project history, no relation to any other branch.

**Do not edit it and do not merge it anywhere.** The next run of the workflow
that publishes it overwrites whatever is here.

## What is here

- [chebpy.pdf](chebpy.pdf)

## Where it comes from

The (RHIZA) PAPER workflow compiles the paper on every push to the
default branch and pushes the result here. The message of each commit names the
source commit it was built from, so the log of this branch is the history of the
document. Edit the LaTeX sources on the default branch, not here.

## Why a branch at all

The same PDF is published in two other places, and the three fail differently:

- as a **run artifact** on the workflow run, which is immediate but expires;
- as a **site asset** in the documentation, if the project builds a book, which is
  durable but needs the site to build.

This branch is the copy that needs neither: a stable path anyone can link, fetch
or clone without a build and without an unexpired run.
