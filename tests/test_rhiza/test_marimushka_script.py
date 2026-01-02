"""Tests for the marimushka.sh script using a sandboxed environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides test fixtures for testing git-based workflows and version management.
"""

import os
import shutil
import subprocess

# Get shell path once at module level
SHELL = shutil.which("sh") or "/bin/sh"


def test_marimushka_script_success(git_repo):
    """Test successful execution of the marimushka script."""
    script = git_repo / ".rhiza" / "scripts" / "marimushka.sh"

    # Setup directories in the git repo
    marimo_folder = git_repo / "book" / "marimo"
    marimo_folder.mkdir(parents=True)
    (marimo_folder / "notebook.py").touch()

    output_folder = git_repo / "_marimushka"

    # Run the script
    # We need to set env vars. git_repo fixture sets cwd to local_dir.
    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"
    # UVX_BIN is defaulted to ./bin/uvx in the script, which matches our mock setup in git_repo

    result = subprocess.run([SHELL, str(script)], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "Exporting notebooks" in result.stdout
    assert (output_folder / "index.html").exists()
    assert (output_folder / ".nojekyll").exists()
    assert (output_folder / "index.html").read_text() == "<html>Mock Export</html>"


def test_marimushka_missing_folder(git_repo):
    """Test script behavior when MARIMO_FOLDER is missing."""
    script = git_repo / ".rhiza" / "scripts" / "marimushka.sh"

    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "missing"

    result = subprocess.run([SHELL, str(script)], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "does not exist" in result.stdout


def test_marimushka_no_python_files(git_repo):
    """Test script behavior when MARIMO_FOLDER has no python files."""
    script = git_repo / ".rhiza" / "scripts" / "marimushka.sh"

    marimo_folder = git_repo / "book" / "marimo"
    marimo_folder.mkdir(parents=True)
    # No .py files created

    output_folder = git_repo / "_marimushka"

    env = os.environ.copy()
    env["MARIMO_FOLDER"] = "book/marimo"
    env["MARIMUSHKA_OUTPUT"] = "_marimushka"

    result = subprocess.run([SHELL, str(script)], env=env, cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "No Python files found" in result.stdout
    assert (output_folder / "index.html").exists()
    assert "No notebooks found" in (output_folder / "index.html").read_text()
