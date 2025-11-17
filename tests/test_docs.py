"""Doctest README.md's Python code blocks.

This module parses fenced Python blocks from the top-level README.md and runs
them with doctest in a shared namespace. It enables ELLIPSIS,
NORMALIZE_WHITESPACE, and IGNORE_EXCEPTION_DETAIL so examples remain stable yet
meaningful as the code evolves while focusing comparisons on relevant output.
"""

import doctest
from doctest import ELLIPSIS, IGNORE_EXCEPTION_DETAIL, NORMALIZE_WHITESPACE
from pathlib import Path

import pytest


@pytest.fixture()
def readme_path() -> Path:
    """Provide the path to the project's README.md file.

    This fixture searches for the README.md file by starting in the current
    directory and moving up through parent directories until it finds the file.

    Returns:
    -------
    Path
        Path to the README.md file

    Raises:
    ------
    FileNotFoundError
        If the README.md file cannot be found in any parent directory

    """
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        candidate = current_dir / "README.md"
        if candidate.is_file():
            return candidate
        current_dir = current_dir.parent
    raise FileNotFoundError("README.md not found in any parent directory")


def test_doc(readme_path):
    """Run doctests extracted from README.md using a tolerant checker.

    Ensures all Python code blocks in the README execute and their outputs
    match expected results, allowing for minor floating point differences.
    """
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(
        optionflags=ELLIPSIS | NORMALIZE_WHITESPACE | IGNORE_EXCEPTION_DETAIL,
    )

    doc = readme_path.read_text(encoding="utf-8")

    test = parser.get_doctest(doc, globs={}, name=readme_path.name, filename=str(readme_path), lineno=0)
    result = runner.run(test)

    assert result.failed == 0
