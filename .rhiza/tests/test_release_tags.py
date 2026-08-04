"""Tests for the release tags every language layer's version config derives from.

This file and its associated tests flow down via a SYNC action from the
jebel-quant/rhiza repository (https://github.com/jebel-quant/rhiza).

Owned by ``core`` because the invariant is about git, not about a language. All three
layers depend on it for the same reason: ``python-core``'s ``[tool.bumpversion]`` table
and the root ``.bumpversion.toml`` that ``rust-core`` and ``go-core`` ship both fall back
to reading the newest tag when they cannot read a version from a file, and git-cliff
places changelog boundaries at tags. An unreachable tag breaks both.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest

_GIT = shutil.which("git") or "/usr/bin/git"


def test_latest_tag_is_reachable_from_a_branch(latest_tag: str, root: Path) -> None:
    """The newest tag must sit on a commit some branch contains (#1454).

    ``git tag --list`` happily reports an orphaned tag, which is how a repo can stay
    green while ``git describe`` disagrees with it. A release cut on a branch that is
    then squash-merged leaves its tag on the pre-squash commit while the content lands
    on the default branch under a new SHA; no branch contains the tagged commit any
    more.

    The consequence is not cosmetic. git-cliff cannot place a boundary at an
    unreachable tag, so regenerating CHANGELOG.md deletes that version's section and
    folds its commits into the next release. Bump tooling reading the version from
    ``git describe`` skips the release for the same reason.
    """
    if (
        subprocess.run(  # nosec B603
            [_GIT, "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, cwd=root
        ).stdout.strip()
        == "true"
    ):
        pytest.skip("shallow clone — the commit graph is incomplete")

    commit = subprocess.run(  # nosec B603
        [_GIT, "rev-parse", f"{latest_tag}^{{commit}}"], capture_output=True, text=True, cwd=root
    )
    if commit.returncode != 0:
        pytest.skip(f"tagged commit for {latest_tag} is not present locally")

    contains = subprocess.run(  # nosec B603
        [_GIT, "branch", "-a", "--contains", commit.stdout.strip(), "--format=%(refname:short)"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert contains.stdout.strip(), (
        f"Tag {latest_tag} points at {commit.stdout.strip()[:12]}, which no branch contains. "
        f"It is most likely the pre-squash commit of a squash-merged release branch: "
        f"`git describe` skips this release and regenerating CHANGELOG.md will delete its "
        f"section. Re-tag the merged commit and delete the orphaned tag."
    )
