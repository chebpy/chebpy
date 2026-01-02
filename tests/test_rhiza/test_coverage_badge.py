"""Tests for coverage badge generation script."""

import json
import subprocess
import tempfile
from pathlib import Path


def test_coverage_badge_generation(tmp_path, root):
    """Test that the coverage badge generation script works correctly."""
    # Setup
    tests_dir = tmp_path / "_tests"
    book_dir = tmp_path / "_book" / "tests"
    tests_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)

    # Create a mock coverage.json with realistic data
    coverage_data = {
        "totals": {
            "percent_covered": 85.7,
            "percent_covered_display": "86",
        }
    }
    coverage_json = tests_dir / "coverage.json"
    coverage_json.write_text(json.dumps(coverage_data))

    # Run the script
    script_path = root / ".rhiza" / "scripts" / "generate-coverage-badge.sh"

    # Change to tmp directory for script execution
    result = subprocess.run(
        ["/bin/sh", str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Verify the script ran successfully
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify the badge JSON was created
    badge_json = book_dir / "coverage-badge.json"
    assert badge_json.exists(), "Badge JSON file was not created"

    # Verify the content
    badge_data = json.loads(badge_json.read_text())
    assert badge_data["schemaVersion"] == 1
    assert badge_data["label"] == "coverage"
    assert badge_data["message"] == "86%"
    assert badge_data["color"] == "green"


def test_coverage_badge_colors(root):
    """Test that coverage badge uses correct colors for different percentages."""
    test_cases = [
        (95, "brightgreen"),
        (85, "green"),
        (75, "yellowgreen"),
        (65, "yellow"),
        (55, "orange"),
        (45, "red"),
    ]

    script_path = root / ".rhiza" / "scripts" / "generate-coverage-badge.sh"

    for percent, expected_color in test_cases:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tests_dir = tmp_path / "_tests"
            book_dir = tmp_path / "_book" / "tests"
            tests_dir.mkdir(parents=True)
            book_dir.mkdir(parents=True)

            # Create coverage.json with specific percentage
            coverage_data = {"totals": {"percent_covered": float(percent)}}
            coverage_json = tests_dir / "coverage.json"
            coverage_json.write_text(json.dumps(coverage_data))

            # Run the script
            result = subprocess.run(
                ["/bin/sh", str(script_path)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Script failed for {percent}%: {result.stderr}"

            # Verify the color
            badge_json = book_dir / "coverage-badge.json"
            badge_data = json.loads(badge_json.read_text())
            assert badge_data["color"] == expected_color, (
                f"Expected {expected_color} for {percent}%, got {badge_data['color']}"
            )
