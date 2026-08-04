"""Tests for the README that hold whatever the project is written in.

This file and its associated tests flow down via a SYNC action from the
jebel-quant/rhiza repository (https://github.com/jebel-quant/rhiza).

Owned by ``core`` because none of it is language-specific: every synced README documents
its gates in ``bash`` fences — ``make install``, ``make test``, ``make all`` — and a
fence with a syntax error is broken the same way in a Rust, Go or Python project. Before
this split (#1472) all of it lived in the ``tests`` bundle, which requires
``python-core``, so a Rust or Go repo had no README coverage at all.

The Python-block half stays behind in ``tests`` as ``test_readme_validation.py``: it
executes ``python`` fences and diffs them against a ``result`` block, which only means
something where the project *is* Python.

Note the split of labour with the fence flags: ``SKIP_FLAG`` and ``_should_skip`` are
duplicated across the two modules rather than shared. Bundles are copied
independently — a Rust project receives this file and not the other — so a shared helper
would need a third home that both bundles ship, which is a worse trade for four lines.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404
from pathlib import Path

import pytest

# Bash code blocks — captures optional flags (e.g. "+RHIZA_SKIP") and the code body.
BASH_BLOCK = re.compile(r"```bash([^\n]*)\n(.*?)```", re.DOTALL)

# Bash executable used for syntax checking; `bash -n` parses without executing.
BASH = "bash"

# Flag marking a fence as intentionally excluded. Usage: add it after the language
# identifier on the opening fence line, e.g. ```bash +RHIZA_SKIP
SKIP_FLAG = "+RHIZA_SKIP"

# Box-drawing characters mean the fence is a directory tree, not runnable shell.
_TREE_MARKERS = ("├──", "└──", "│")


def _should_skip(flags: str) -> bool:
    """Return True if the fence flags string contains the +RHIZA_SKIP marker.

    Args:
        flags: Text following the language identifier on the opening fence line.

    Returns:
        True when the block is intentionally excluded.
    """
    return SKIP_FLAG in flags


class TestReadmeExists:
    """The README has to be there and be readable before anything else applies."""

    def test_readme_file_exists_at_root(self, root: Path) -> None:
        """README.md should exist at repository root."""
        readme = root / "README.md"
        assert readme.exists(), "README.md not found at project root"
        assert readme.is_file(), "README.md is not a regular file"

    def test_readme_is_readable(self, root: Path) -> None:
        """README.md should be readable with UTF-8 encoding and non-empty."""
        content = (root / "README.md").read_text(encoding="utf-8")
        assert content.strip(), "README.md is empty"


class TestReadmeBashFragments:
    """Bash fences must parse, in any language's project.

    Only ``bash -n`` — the blocks are parsed, never executed. A README's shell examples
    are usually destructive-adjacent (`make clean`, `git push`) and running them is not
    what this is for; a fence that cannot even parse is a documentation bug regardless.
    """

    def test_bash_blocks_basic_syntax(self, root: Path, logger) -> None:
        """Every non-skipped bash block should parse under `bash -n`."""
        content = (root / "README.md").read_text(encoding="utf-8")
        bash_blocks = BASH_BLOCK.findall(content)

        logger.info("Found %d bash code block(s) in README", len(bash_blocks))

        for i, (flags, code) in enumerate(bash_blocks):
            if _should_skip(flags):
                logger.info("Skipping bash block %d (%s flag)", i, SKIP_FLAG)
                continue

            if any(marker in code for marker in _TREE_MARKERS):
                logger.info("Skipping bash block %d (directory tree representation)", i)
                continue

            # A block that is only comments has nothing to parse and no way to be wrong.
            lines = [line.strip() for line in code.split("\n") if line.strip()]
            if not [line for line in lines if not line.startswith("#")]:
                logger.info("Skipping bash block %d (only comments)", i)
                continue

            logger.debug("Checking bash block %d:\n%s", i, code)

            result = subprocess.run(  # nosec B603 B607 - `bash -n` parses without executing
                [BASH, "-n"],
                input=code,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                pytest.fail(f"Bash block {i} has syntax errors:\nCode:\n{code}\nError:\n{result.stderr}")


class TestSkipFlag:
    """Tests for the +RHIZA_SKIP flag that excludes an individual fence."""

    def test_should_skip_returns_true_for_skip_flag(self) -> None:
        """+RHIZA_SKIP in flags string should cause _should_skip to return True."""
        assert _should_skip(" +RHIZA_SKIP") is True
        assert _should_skip("+RHIZA_SKIP") is True
        assert _should_skip(" +RHIZA_SKIP other-flag") is True

    def test_should_skip_returns_false_without_flag(self) -> None:
        """Absence of +RHIZA_SKIP should cause _should_skip to return False."""
        assert _should_skip("") is False
        assert _should_skip(" ") is False
        assert _should_skip("other-flag") is False

    def test_bash_block_with_skip_flag_is_excluded(self, tmp_path: Path) -> None:
        """A ```bash +RHIZA_SKIP block should not be syntax-checked."""
        readme = tmp_path / "README.md"
        readme.write_text(
            "```bash +RHIZA_SKIP\nnot-valid-bash @@@@\n```\n```bash\necho hello\n```\n",
            encoding="utf-8",
        )
        all_blocks = BASH_BLOCK.findall(readme.read_text(encoding="utf-8"))
        assert len(all_blocks) == 2
        checked = [code for flags, code in all_blocks if not _should_skip(flags)]
        assert len(checked) == 1
        assert "not-valid-bash" not in checked[0]
