"""Tests for the marimushka Makefile target using a sandboxed environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides test fixtures for testing git-based workflows and version management.
"""

import os
import shutil
import subprocess  # nosec

import pytest

# Get shell path and make command once at module level
SHELL = shutil.which("sh") or "/bin/sh"
MAKE = shutil.which("make") or "/usr/bin/make"


def test_marimushka_target_success(git_repo):
    """Test successful execution of the marimushka Makefile target."""
    # only run this test if the marimo folder is present
    if not (git_repo / "book" / "marimo").exists():
        pytest.skip("marimo folder not found, skipping test")

    # Setup directories in the git repo
    marimo_folder = git_repo / "book" / "marimo" / "notebooks"
    marimo_folder.mkdir(parents=True, exist_ok=True)
    (marimo_folder / "notebook.py").touch()

    output_folder = git_repo / "_marimushka"

    # Run the make target
    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo/notebooks"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"

    # Create dummy bin/uv and bin/uvx if they don't exist
    (git_repo / "bin").mkdir(exist_ok=True)
    (git_repo / "bin" / "uv").touch()
    (git_repo / "bin" / "uv").chmod(0o755)
    (git_repo / "bin" / "uvx").touch()
    (git_repo / "bin" / "uvx").chmod(0o755)

    # Put our bin on the PATH so 'command -v uvx' finds it in the test
    env["PATH"] = f"{git_repo}/bin:{env.get('PATH', '')}"

    # In tests, we don't want to actually run marimushka as it's not installed in the mock env
    # But we want to test that the Makefile logic works.
    # We can mock the marimushka CLI call by creating a script that generates the expected files.
    with open(git_repo / "bin" / "marimushka", "w") as f:
        f.write(
            f"#!/bin/sh\nmkdir -p {output_folder}/notebooks\n"
            f"touch {output_folder}/index.html\n"
            f"touch {output_folder}/notebooks/notebook.html\n"
        )
    (git_repo / "bin" / "marimushka").chmod(0o755)

    # Override UVX_BIN to use our mock marimushka CLI
    env["UVX_BIN"] = str(git_repo / "bin" / "marimushka")

    result = subprocess.run([MAKE, "marimushka"], env=env, cwd=git_repo, capture_output=True, text=True)  # nosec

    assert result.returncode == 0
    assert "Exporting notebooks" in result.stdout
    assert (output_folder / "index.html").exists()
    assert (output_folder / "notebooks" / "notebook.html").exists()


def test_marimushka_no_python_files(git_repo):
    """Test marimushka target behavior when MARIMO_FOLDER has no python files."""
    if not (git_repo / "book" / "marimo").exists():
        pytest.skip("marimo folder not found, skipping test")

    marimo_folder = git_repo / "book" / "marimo" / "notebooks"
    marimo_folder.mkdir(parents=True, exist_ok=True)

    # Delete all .py files in the marimo folder
    for file in marimo_folder.glob("*.py"):
        file.unlink()

    # No .py files created

    output_folder = git_repo / "_marimushka"

    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo/notebooks"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"

    result = subprocess.run([MAKE, "marimushka"], env=env, cwd=git_repo, capture_output=True, text=True)  # nosec

    assert result.returncode == 0
    assert (output_folder / "index.html").exists()
