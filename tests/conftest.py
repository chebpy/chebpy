"""Configuration and fixtures for pytest.

This module contains global pytest configuration and fixtures that are
available to all test modules. It handles special configurations needed
for different environments.

Specifically, it:
- Sets the matplotlib backend to 'Agg' (a non-interactive backend) when
  running in a CI environment, which is necessary for running tests that
  generate plots without a display.

Note:
    The 'Agg' backend is used because it doesn't require a graphical display,
    making it suitable for headless CI environments.
"""
import os
import matplotlib

if os.environ.get("CI") == "true":
    matplotlib.use("Agg")
