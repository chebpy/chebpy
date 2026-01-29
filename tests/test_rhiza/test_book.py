"""Tests for book-related Makefile targets and their resilience."""

import shutil
import subprocess

import pytest

MAKE = shutil.which("make") or "/usr/bin/make"


def test_no_book_folder(git_repo):
    """Test that make targets fail gracefully when book folder is missing."""
    if (git_repo / "book").exists():
        shutil.rmtree(git_repo / "book")
    assert not (git_repo / "book").exists()

    for target in ["book", "docs", "marimushka"]:
        # test resilience
        result = subprocess.run([MAKE, target], cwd=git_repo, capture_output=True, text=True)

        assert result.returncode != 0
        assert "no rule to make target" in result.stderr.lower()


def test_book_folder_but_no_mk(git_repo):
    """Test behavior when book folder exists but book.mk is missing."""
    # ensure book folder exists but has no Makefile
    if (git_repo / "book").exists():
        shutil.rmtree(git_repo / "book")
    # create an empty book folder. Make treats an existing directory as an “up-to-date” target.
    (git_repo / "book").mkdir()

    # assert the book folder exists
    assert (git_repo / "book").exists()
    # assert the book.mk file does not exist
    assert not (git_repo / "book" / "book.mk").exists()
    # assert the git_repo / "book" folder is empty
    assert not list((git_repo / "book").iterdir())

    # test resilience
    result = subprocess.run([MAKE, "book"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "nothing to be done" in result.stdout.lower()

    for target in ["docs", "marimushka"]:
        # test resilience
        result = subprocess.run([MAKE, target], cwd=git_repo, capture_output=True, text=True)

        assert result.returncode != 0
        assert "no rule to make target" in result.stderr.lower()


def test_book_folder(git_repo):
    """Test that book.mk defines the expected phony targets."""
    # if file book/book.mk exists, make should run successfully
    if not (git_repo / "book" / "book.mk").exists():
        pytest.skip("book.mk not found, skipping test")

    makefile = git_repo / "book" / "book.mk"
    content = makefile.read_text()

    # get the list of phony targets from the Makefile
    phony_targets = [line.strip() for line in content.splitlines() if line.startswith(".PHONY:")]
    targets = set(phony_targets[0].split(":")[1].strip().split())
    assert {"book", "docs", "marimushka"} == targets, (
        f"Expected phony targets to include book, docs, and marimushka, got {targets}"
    )
