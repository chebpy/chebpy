"""Utility test fixtures and setup.

This conftest sets up the Python path to allow imports from .rhiza/utils
for testing utility scripts and helpers.
"""

import sys
from pathlib import Path

# Add the utils directory to the path for imports
# From .rhiza/tests/utils/conftest.py, .rhiza/utils is 3 levels up then down into utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utils"))
