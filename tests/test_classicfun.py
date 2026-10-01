"""Unit-tests for the classicfun module (src/chebpy/classicfun.py).

The Classicfun class is an abstract base class tested indirectly through its
concrete subclasses (Bndfun). This file provides minimal direct tests.
"""

import numpy as np
import pytest

from chebpy.bndfun import Bndfun
from chebpy.classicfun import Classicfun
from chebpy.exceptions import IntervalMismatch, NotSubinterval
from chebpy.utilities import Interval


class TestClassicfun:
    """Tests for the Classicfun abstract base class."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Classicfun()

    def test_repr(self):
        f = Bndfun.initfun_fixedlen(np.sin, Interval(0, 1), 5)
        assert repr(f) == "Bndfun([0.0, 1.0], 5)"

    def test_restrict_outside_interval_raises(self):
        f = Bndfun.initfun_adaptive(np.sin, Interval(0, 1))
        with pytest.raises(NotSubinterval):
            f.restrict(Interval(0, 2))

    def test_binary_op_interval_mismatch_raises(self):
        f = Bndfun.initfun_adaptive(np.sin, Interval(0, 1))
        g = Bndfun.initfun_adaptive(np.cos, Interval(0, 2))
        with pytest.raises(IntervalMismatch):
            f + g
