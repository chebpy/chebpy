"""Unit-tests for the classicfun module (src/chebpy/classicfun.py).

The Classicfun class is an abstract base class tested indirectly through its
concrete subclasses (Bndfun). This file provides minimal direct tests.
"""

import pytest

from chebpy.classicfun import Classicfun


class TestClassicfun:
    """Tests for the Classicfun abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Classicfun()
