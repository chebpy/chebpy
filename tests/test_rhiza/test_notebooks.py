"""Tests for Marimo notebooks."""

import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import dotenv_values

# Read .rhiza/.env at collection time (no environment side-effects).
# dotenv_values returns a dict of key -> value (or None for missing).
RHIZA_ENV_PATH = Path(".rhiza/.env")


def collect_marimo_notebooks(env_path: Path = RHIZA_ENV_PATH):
    """Return a sorted list of notebook script Paths discovered from .rhiza/.env.

    - Reads MARIMO_FOLDER from .rhiza/.env (if present), otherwise falls back to "marimo".
    - Returns [] if the folder does not exist.
    """
    values = {}
    if env_path.exists():
        values = dotenv_values(env_path)

    marimo_folder = values.get("MARIMO_FOLDER", "book/marimo/notebooks")
    marimo_path = Path(marimo_folder)

    if not marimo_path.exists():
        return []

    # Return sorted list for stable ordering
    return sorted(marimo_path.glob("*.py"))


# Collect notebook paths at import/collection time so pytest.parametrize can use them.
NOTEBOOK_PATHS = collect_marimo_notebooks()


@pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_execution(notebook_path: Path):
    """Test if a Marimo notebook can be executed without errors.

    We use 'marimo export html' which executes the notebook cells and
    reports if any cells failed.
    """
    # Determine uvx command: prefer local ./bin/uvx, then fall back to uvx on PATH.
    local_uvx = Path("bin/uvx")

    if local_uvx.exists() and local_uvx.is_file():
        uvx_cmd = str(local_uvx.resolve())  # Use absolute path
    else:
        uvx_cmd = shutil.which("uvx")
        if uvx_cmd is None:
            pytest.skip("uvx not found (neither ./bin/uvx nor uvx on PATH); skipping marimo notebook tests")

    cmd = [
        uvx_cmd,
        "--with",
        "typing_extensions",  # Workaround: marimo's starlette dep needs this
        "marimo",
        "export",
        "html",
        "--sandbox",
        str(notebook_path),
        "-o",
        "/dev/null",  # We don't need the actual HTML output
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Ensure process exit code indicates success
    assert result.returncode == 0, (
        f"Marimo export returned non-zero for {notebook_path.name}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )

    # Check stdout/stderr for known failure messages (case-insensitive)
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    lower_output = combined_output.lower()

    failure_keywords = [
        "some cells failed to execute",
        "cells failed to execute",
        "marimoexceptionraisederror",
    ]
    for kw in failure_keywords:
        assert kw.lower() not in lower_output, (
            f"Notebook {notebook_path.name} reported cell failures (found keyword '{kw}'):\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
