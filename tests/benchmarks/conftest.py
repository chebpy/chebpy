"""Pytest configuration for benchmark tests.

This file can be used to add custom fixtures or configuration
for your benchmark tests.

Security Notes:
- S101 (assert usage): Asserts are the standard way to validate test conditions in pytest.
  They provide clear test failure messages and are expected in test code.
"""


def pytest_html_report_title(report):
    """Set the HTML report title."""
    report.title = "Benchmark Tests"
