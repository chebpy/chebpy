"""Unit-tests for Bndfun numpy ufunc overloads."""

import numpy as np

from chebpy.bndfun import Bndfun
from chebpy.utilities import Interval

from ..generic.ufuncs import test_emptycase, ufunc_parameter  # noqa: F401
from ..utilities import eps


def test_ufunc():
    """Test applying ufuncs to Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)

    for ufunc, f, interval in ufunc_parameter():
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(f, subinterval)

        def gg(x):
            return ufunc(f(x))

        gg_result = getattr(ff, ufunc.__name__)()

        xx = subinterval(yy)
        vscl = gg_result.vscale
        lscl = gg_result.size
        assert np.max(np.abs(gg(xx) - gg_result(xx))) <= vscl * lscl * eps
