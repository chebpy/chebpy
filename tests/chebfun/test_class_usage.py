"""Unit-tests for Chebfun class usage.

This module contains tests for various aspects of using the Chebfun class,
including string representation, copying, iteration, the x property,
restriction, simplification, and translation.
"""

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun
from chebpy.core.utilities import Domain

from ..generic.class_usage import test_support, test_translate_empty  # noqa: F401
from .conftest import eps


@pytest.fixture
def class_usage_fixtures() -> dict:
    """Create Chebfun objects for testing class usage.

    This fixture creates several Chebfun objects with different characteristics
    for testing various aspects of class usage.

    Returns:
        dict: Dictionary containing:
            f1: Chebfun on [-1, 1]
            f2: Chebfun on a piecewise domain
    """
    f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])
    return {"f1": f1, "f2": f2}


def test__str__(class_usage_fixtures: dict) -> None:
    """Test string representation of Chebfun objects.

    This test verifies that the string representation of Chebfun objects
    contains the expected information.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    assert "Chebfun" in str(f1)
    assert "domain" in str(f1)


def test__repr__(class_usage_fixtures: dict) -> None:
    """Test repr representation of Chebfun objects.

    This test verifies that the repr representation of Chebfun objects
    contains the expected information.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    assert "Chebfun" in repr(f1)
    assert "domain" in repr(f1)


def test__iter__(class_usage_fixtures: dict) -> None:
    """Test iteration over Chebfun objects.

    This test verifies that iterating over a Chebfun object yields
    its constituent Bndfun objects.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    assert len(list(f1)) == 1

    # check piecewise case
    assert len(list(f2)) == 3


def test_x_property(class_usage_fixtures: dict) -> None:
    """Test the x property of Chebfun objects.

    This test verifies that the x property returns a Chebfun object
    representing the identity function on the same domain.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(f1.x(xx) - xx)) <= eps

    # check piecewise case
    xx = np.linspace(-1, 2, 100)
    assert np.max(np.abs(f2.x(xx) - xx)) <= eps


def test_restrict_(class_usage_fixtures: dict) -> None:
    """Test the restrict_ method of Chebfun objects.

    This test verifies that the restrict_ method correctly restricts
    a Chebfun to a subdomain, modifying the object in place.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    g1 = f1.copy()
    g1.restrict_([-0.5, 0.5])
    assert g1.domain == Domain([-0.5, 0.5])
    xx = np.linspace(-0.5, 0.5, 100)
    assert np.max(np.abs(g1(xx) - f1(xx))) <= 5 * eps

    # check piecewise case
    g2 = f2.copy()
    g2.restrict_([-0.5, 1.5])
    assert g2.domain == Domain([-0.5, 0, 1, 1.5])
    xx = np.linspace(-0.5, 1.5, 100)
    assert np.max(np.abs(g2(xx) - f2(xx))) <= 5 * eps

    # check with a domain that has more breakpoints
    g2 = f2.copy()
    g2.restrict_([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    assert g2.domain == Domain([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    xx = np.linspace(-0.8, 1.8, 100)
    assert np.max(np.abs(g2(xx) - f2(xx))) <= 5 * eps


def test_simplify(class_usage_fixtures: dict) -> None:
    """Test the simplify method of Chebfun objects.

    This test verifies that the simplify method correctly simplifies
    a Chebfun by removing unnecessary coefficients.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    g1 = f1.simplify()
    assert g1 == f1

    # check piecewise case
    g2 = f2.simplify()
    assert g2 == f2


def test_simplify_empty(emptyfun) -> None:
    """Test the simplify method on an empty Chebfun.

    This test verifies that the simplify method on an empty Chebfun
    returns an empty Chebfun.

    Args:
        emptyfun: Fixture providing an empty Chebfun object.
    """
    g0 = emptyfun.simplify()
    assert g0.isempty


def test_restrict(class_usage_fixtures: dict) -> None:
    """Test the restrict method of Chebfun objects.

    This test verifies that the restrict method correctly restricts
    a Chebfun to a subdomain, returning a new Chebfun object.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    g1 = f1.restrict([-0.5, 0.5])
    assert g1.domain == Domain([-0.5, 0.5])
    xx = np.linspace(-0.5, 0.5, 100)
    assert np.max(np.abs(g1(xx) - f1(xx))) <= 5 * eps

    # check piecewise case
    g2 = f2.restrict([-0.5, 1.5])
    assert g2.domain == Domain([-0.5, 0, 1, 1.5])
    xx = np.linspace(-0.5, 1.5, 100)
    assert np.max(np.abs(g2(xx) - f2(xx))) <= 5 * eps

    # check with a domain that has more breakpoints
    g2 = f2.restrict([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    assert g2.domain == Domain([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
    xx = np.linspace(-0.8, 1.8, 100)
    assert np.max(np.abs(g2(xx) - f2(xx))) <= 5 * eps


def test_translate(class_usage_fixtures: dict) -> None:
    """Test the translate method of Chebfun objects.

    This test verifies that the translate method correctly translates
    a Chebfun by a specified amount.

    Args:
        class_usage_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = class_usage_fixtures["f1"]
    f2 = class_usage_fixtures["f2"]

    # check continuous case
    g1 = f1.translate(1)
    assert g1.domain == Domain([0, 2])
    xx = np.linspace(-1, 1, 100)
    yy = xx + 1
    assert np.max(np.abs(g1(yy) - f1(xx))) <= 2 * eps

    # check piecewise case
    g2 = f2.translate(1)
    assert g2.domain == Domain([0, 1, 2, 3])
    xx = np.linspace(-1, 2, 100)
    yy = xx + 1
    assert np.max(np.abs(g2(yy) - f2(xx))) <= 2 * eps

    # check with a negative translation
    g1 = f1.translate(-1)
    assert g1.domain == Domain([-2, 0])
    xx = np.linspace(-1, 1, 100)
    yy = xx - 1
    assert np.max(np.abs(g1(yy) - f1(xx))) <= 2 * eps


def test_copy(constfun):
    """Test the copy method of fun.

    Args:
        constfun: Fixture providing a constant function object.
    """
    ff = constfun
    gg = ff.copy()
    assert ff == ff
    assert gg == gg

    # we check for identity via function values, etc.
    assert ff == gg

    # check that modifying the copy does not affect the original
    gg.domain = Domain([-1, 0, 0.5, 1])
    assert gg != ff
