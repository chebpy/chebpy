"""Tests for the .env file and the paths it points to.

This file is part of the tschm/.config-templates repository
(https://github.com/tschm/.config-templates).

This module verifies that:
1. The .env file exists in the project root
2. The folder paths specified in the .env file (MARIMO_FOLDER, SOURCE_FOLDER, TESTS_FOLDER)
   exist in the project structure and are valid directories
"""

from pathlib import Path

import pytest
from dotenv import dotenv_values


def find_project_root(start_path: Path = None) -> Path:
    """Find the project root directory by looking for the .git folder.

    This function iterates up the directory tree from the given starting path
    until it finds a directory containing a .git folder, which is assumed to be
    the project root.

    Args:
        start_path (Path, optional): The path to start searching from.
            If None, uses the directory of the file calling this function.

    Returns:
        Path: The path to the project root directory.

    Raises:
        FileNotFoundError: If no .git directory is found in any parent directory.
    """
    if start_path is None:
        # If no start_path is provided, use the current file's directory
        start_path = Path(__file__).parent

    # Convert to absolute path to handle relative paths
    current_path = start_path.absolute()

    # Iterate up the directory tree
    while current_path != current_path.parent:  # Stop at the root directory
        # Check if .git directory exists
        git_dir = current_path / ".git"
        if git_dir.exists() and git_dir.is_dir():
            return current_path

        # Move up to the parent directory
        current_path = current_path.parent

    # If we've reached the root directory without finding .git
    raise FileNotFoundError("Could not find project root: no .git directory found in any parent directory")


@pytest.fixture
def project_root() -> Path:
    """Fixture that provides the project root directory.

    Returns:
        Path: The path to the project root directory.
    """
    return find_project_root(Path(__file__).parent)


@pytest.fixture
def env_content(project_root: Path) -> dict[str, str]:
    """Fixture that provides the content of the .env file as a dictionary.

    Returns:
        dict: A dictionary containing the key-value pairs from the .env file.

    """
    # Get the project root directory
    env_file_path = project_root / ".env"

    return dotenv_values(env_file_path)


def test_env_file_exists(project_root) -> None:
    """Tests that the .env file exists in the project root.

    Args:
        project_root: Path to the project root directory.

    Verifies:
        The .env file exists in the project root directory.
    """
    # Use the project root from the fixture
    env_file_path = project_root / ".env"

    assert env_file_path.exists(), ".env file does not exist in project root"


@pytest.mark.parametrize("folder_key", ["MARIMO_FOLDER", "SOURCE_FOLDER", "TESTS_FOLDER"])
def test_folder_exists(env_content: dict[str, str], project_root: Path, folder_key: str) -> None:
    """Tests that the folder path specified in the .env file exists.

    Args:
        env_content: Dictionary containing the environment variables from .env file.
        project_root: Path to the project root directory.
        folder_key: The key in the .env file for the folder to check.

    Verifies:
        The folder path exists in the project structure.

    """
    # Get the folder path from the env_content fixture
    folder_path = env_content.get(folder_key)
    assert folder_path is not None, f"{folder_key} not found in .env file"

    # Check if the path exists
    full_path = project_root / folder_path
    assert full_path.exists(), f"{folder_key} path '{folder_path}' does not exist"
    assert full_path.is_dir(), f"{folder_key} path '{folder_path}' is not a directory"
