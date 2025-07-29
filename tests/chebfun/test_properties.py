"""Unit-tests for Chebfun properties.

This module contains tests for the various properties of Chebfun objects,
including breakpoints, domain, hscale, isempty, isconst, support, and vscale.
"""

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun
from chebpy.core.utilities import Domain


@pytest.fixture
def properties_fixtures():
    """Create Chebfun objects for testing properties.

    This fixture creates several Chebfun objects with different characteristics
    for testing various properties.

    Returns:
        dict: Dictionary containing:
            f0: Empty Chebfun
            f1: Constant Chebfun
            f2: Identity Chebfun on [-1, 1]
            f3: Chebfun on a piecewise domain
    """
    f0 = Chebfun.initempty()
    f1 = Chebfun.initconst(1.0)
    f2 = Chebfun.initidentity([-1, 1])
    f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-2, 0, 1, 3])
    return {"f0": f0, "f1": f1, "f2": f2, "f3": f3}


def test_breakpoints(properties_fixtures):
    """Test the breakpoints property of Chebfun objects.

    This test verifies that the breakpoints property correctly returns
    the domain breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = properties_fixtures["f3"]
    assert np.all(f3.breakpoints == np.array([-2, 0, 1, 3]))


def test_domain(properties_fixtures):
    """Test the domain property of Chebfun objects.

    This test verifies that the domain property correctly returns
    a Domain object with the expected breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert f0.domain == Domain([])
    assert f1.domain == Domain([-1, 1])
    assert f2.domain == Domain([-1, 1])
    assert f3.domain == Domain([-2, 0, 1, 3])


def test_hscale(properties_fixtures):
    """Test the hscale property of Chebfun objects.

    This test verifies that the hscale property correctly returns
    the horizontal scale (maximum domain width) for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f3 = properties_fixtures["f3"]
    assert f0.hscale == 0.0
    assert f1.hscale == 1.0
    assert f3.hscale == 3.0


def test_isempty(properties_fixtures):
    """Test the isempty property of Chebfun objects.

    This test verifies that the isempty property correctly identifies
    empty Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    assert f0.isempty
    assert not f1.isempty


def test_isconst(properties_fixtures):
    """Test the isconst property of Chebfun objects.

    This test verifies that the isconst property correctly identifies
    constant Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert not f0.isconst
    assert f1.isconst
    assert not f2.isconst
    assert not f3.isconst


def test_support(properties_fixtures):
    """Test the support property of Chebfun objects.

    This test verifies that the support property correctly returns
    the first and last breakpoints for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = properties_fixtures["f0"]
    f1 = properties_fixtures["f1"]
    f2 = properties_fixtures["f2"]
    f3 = properties_fixtures["f3"]
    assert f0.support.size == 0
    assert np.all(f1.support == np.array([-1, 1]))
    assert np.all(f2.support == np.array([-1, 1]))
    assert np.all(f3.support == np.array([-2, 3]))


def test_vscale(properties_fixtures):
    """Test the vscale property of Chebfun objects.

    This test verifies that the vscale property correctly returns
    the vertical scale (maximum absolute value) for various Chebfun objects.

    Args:
        properties_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = properties_fixtures["f1"]
    assert f1.vscale == 1.0
