"""Unit-tests for the smoothfun module (src/chebpy/smoothfun.py).

The Smoothfun class is an abstract base class tested indirectly through its
concrete subclasses (Chebtech). This file provides minimal direct tests.
"""

import pytest

from chebpy.smoothfun import Smoothfun


class TestSmoothfun:
    """Tests for the Smoothfun abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Smoothfun()
