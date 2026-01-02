"""Tests for the update-readme-help.sh script using a sandboxed git environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).
"""

import shutil
import subprocess

# Get shell path once at module level
SHELL = shutil.which("sh") or "/bin/sh"


def test_update_readme_success(git_repo):
    """Test successful update of README.md."""
    script = git_repo / ".rhiza" / "scripts" / "update-readme-help.sh"
    readme_path = git_repo / "README.md"

    # Create a README with the target section
    initial_content = """# Project

Some description.

Run `make help` to see all available targets:

```makefile
old help content
```

Footer content.
"""
    readme_path.write_text(initial_content)

    # Run the script
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert "README.md updated" in result.stdout

    # Verify content
    new_content = readme_path.read_text()
    assert "Mock Makefile Help" in new_content
    assert "old help content" not in new_content
    assert "Footer content" in new_content


def test_update_readme_no_marker(git_repo):
    """Test script behavior when README.md lacks the marker."""
    script = git_repo / ".rhiza" / "scripts" / "update-readme-help.sh"
    readme_path = git_repo / "README.md"

    # Create a README without the target section
    initial_content = """# Project

No help section here.
"""
    readme_path.write_text(initial_content)

    # Run the script
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, capture_output=True, text=True)

    # The script exits with 0 if pattern not found (based on my reading of the script)
    # Wait, let's check the script again.
    # if (pattern_found == 0) { print ... > "/dev/stderr"; exit 2 }
    # ...
    # if [ $awk_status -eq 2 ]; then ... exit 0

    assert result.returncode == 0
    # It prints to stderr if not found, but then exits 0.
    # Note: The script redirects the awk error to stderr.
    # But the script itself swallows the exit code 2 and exits 0.

    # Verify content is unchanged
    assert readme_path.read_text() == initial_content


def test_update_readme_preserves_surrounding_content(git_repo):
    """Test that content before and after the help block is preserved."""
    script = git_repo / ".rhiza" / "scripts" / "update-readme-help.sh"
    readme_path = git_repo / "README.md"

    initial_content = """Header

Run `make help` to see all available targets:

```makefile
replace me
```

Footer
"""
    readme_path.write_text(initial_content)

    subprocess.run([SHELL, str(script)], cwd=git_repo, check=True)

    new_content = readme_path.read_text()
    assert new_content.startswith("Header\n\nRun `make help`")
    assert new_content.endswith("```\n\nFooter\n")
