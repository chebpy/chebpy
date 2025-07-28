"""Unit-tests for Bndfun roots functionality"""

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval
from .conftest import pi, sin, cos, eps
from ..utilities import infnorm


def test_empty():
    """Test the roots method on an empty Bndfun."""
    ff = Bndfun.initempty()
    assert ff.roots().size == 0


def test_const():
    """Test the roots method on constant Bndfun objects."""
    ff = Bndfun.initconst(0.0, Interval(-2, 3))
    gg = Bndfun.initconst(2.0, Interval(-2, 3))
    assert ff.roots().size == 0
    assert gg.roots().size == 0


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
    assert infnorm(rts - roots_expected) <= tol