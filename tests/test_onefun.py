"""Unit-tests for the onefun module (src/chebpy/onefun.py).

The Onefun class is an abstract base class tested indirectly through its
concrete subclasses (Chebtech). This file provides minimal direct tests.
"""

import pytest

from chebpy.onefun import Onefun


class TestOnefun:
    """Tests for the Onefun abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Onefun()
