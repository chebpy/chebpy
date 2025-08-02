"""Unit-tests for Chebfun ufunc operations.

This module contains tests for the ufunc operations of Chebfun,
including absolute, trigonometric, exponential, and logarithmic functions.
"""

import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.utilities import Interval

from ..generic.ufuncs import test_emptycase, ufunc_parameter  # noqa: F401
from ..utilities import eps


def test_abs_absolute_alias():
    """Test that abs and absolute are aliases.

    This test verifies that the abs and absolute methods of Chebfun
    are aliases for the same function.
    """
    assert Chebfun.abs == Chebfun.absolute


# Generate test functions for ufuncs
def test_ufuncs():
    """Test ufunc operations on Chebfun objects.

    This test verifies that applying a ufunc to a Chebfun object
    produces the expected result within a specified tolerance.
    """
    yy = np.linspace(-1, 1, 2000)

    for ufunc, f, interval in ufunc_parameter():
        interval = Interval(*interval)
        a, b = interval
        ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

        def gg(x):
            return ufunc(f(x))

        gg_result = getattr(ff, ufunc.__name__)()

        xx = interval(yy)
        vscl = gg_result.vscale
        lscl = sum([fun.size for fun in gg_result])
        assert np.max(np.abs(gg(xx) - gg_result(xx))) <= vscl * lscl * eps
