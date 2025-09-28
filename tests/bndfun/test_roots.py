"""Unit-tests for Bndfun roots functionality."""

import numpy as np
import pytest

from chebpy.bndfun import Bndfun
from chebpy.utilities import Interval

from ..utilities import cos, eps, pi, sin

# Test parameters for roots tests
roots_test_params = [
    (lambda x: 3 * x + 2.0, [-2, 3], np.array([-2 / 3]), eps),
    (lambda x: x**2 + 0.2 * x - 0.08, [-2, 5], np.array([-0.4, 0.2]), 3e1 * eps),
    (lambda x: sin(x), [-7, 7], pi * np.linspace(-2, 2, 5), 1e1 * eps),
    (lambda x: cos(2 * pi * x), [-20, 10], np.linspace(-19.75, 9.75, 60), 3e1 * eps),
    (lambda x: sin(100 * pi * x), [-0.5, 0.5], np.linspace(-0.5, 0.5, 101), eps),
    (lambda x: sin(5 * pi / 2 * x), [-1, 1], np.array([-0.8, -0.4, 0, 0.4, 0.8]), eps),
]


@pytest.mark.parametrize("f, interval, roots_expected, tol", roots_test_params)
def test_roots(f, interval, roots_expected, tol):
    """Test the roots method on various Bndfun objects."""
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(f, subinterval)
    rts = ff.roots()
    assert np.max(np.abs(rts - roots_expected)) <= tol
