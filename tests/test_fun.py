"""Unit-tests for the fun module (src/chebpy/fun.py).

The Fun class is an abstract base class tested indirectly through its
concrete subclasses (Bndfun, Chebfun). This file provides minimal direct
tests.
"""

import pytest

from chebpy.fun import Fun


class TestFun:
    """Tests for the Fun abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Fun()
