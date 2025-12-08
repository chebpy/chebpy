"""Tests for README code examples.

This module extracts Python code and expected result blocks from README.md,
executes the code, and verifies the output matches the documented result.
"""

import re
import subprocess
import sys

import pytest

# Regex for Python code blocks
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

RESULT = re.compile(r"```result\n(.*?)```", re.DOTALL)


def test_readme_runs(logger, root):
    """Execute README code blocks and compare output to documented results."""
    readme = root / "README.md"
    logger.info("Reading README from %s", readme)
    readme_text = readme.read_text(encoding="utf-8")
    code_blocks = CODE_BLOCK.findall(readme_text)
    result_blocks = RESULT.findall(readme_text)
    logger.info("Found %d code block(s) and %d result block(s) in README", len(code_blocks), len(result_blocks))

    code = "".join(code_blocks)  # merged code
    expected = "".join(result_blocks)  # merged results

    # Trust boundary: we execute Python snippets sourced from README.md in this repo.
    # The README is part of the trusted repository content and reviewed in PRs.
    logger.debug("Executing README code via %s -c ...", sys.executable)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603

    stdout = result.stdout
    logger.debug("Execution finished with return code %d", result.returncode)
    if result.stderr:
        logger.debug("Stderr from README code:\n%s", result.stderr)
    logger.debug("Stdout from README code:\n%s", stdout)

    assert result.returncode == 0, f"README code exited with {result.returncode}. Stderr:\n{result.stderr}"
    logger.info("README code executed successfully; comparing output to expected result")
    assert stdout.strip() == expected.strip()
    logger.info("README code output matches expected result")


class TestReadmeTestEdgeCases:
    """Edge cases for README code block testing."""

    def test_readme_file_exists_at_root(self, root):
        """README.md should exist at repository root."""
        readme = root / "README.md"
        assert readme.exists()
        assert readme.is_file()

    def test_readme_is_readable(self, root):
        """README.md should be readable with UTF-8 encoding."""
        readme = root / "README.md"
        content = readme.read_text(encoding="utf-8")
        assert len(content) > 0
        assert isinstance(content, str)

    def test_readme_code_is_syntactically_valid(self, root):
        """Python code blocks in README should be syntactically valid."""
        readme = root / "README.md"
        content = readme.read_text(encoding="utf-8")
        code_blocks = re.findall(r"\`\`\`python\n(.*?)\`\`\`", content, re.DOTALL)

        for i, code in enumerate(code_blocks):
            try:
                compile(code, f"<readme_block_{i}>", "exec")
            except SyntaxError as e:
                pytest.fail(f"Code block {i} has syntax error: {e}")
