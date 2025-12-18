"""Configuration and fixtures for pytest.

This module contains global pytest configuration and fixtures that are
available to all test modules. It handles special configurations needed
for different environments.

Specifically, it:
- Sets the matplotlib backend to 'Agg' (a non-interactive backend) when
  running in a CI environment, which is necessary for running tests that
  generate plots without a display.
- Provides a generic emptyfun fixture that can return different types of
  empty function objects based on the class name.

Note:
    The 'Agg' backend is used because it doesn't require a graphical display,
    making it suitable for headless CI environments.
"""

import dataclasses
import operator
import os

import matplotlib
import numpy as np
import pytest

from chebpy.bndfun import Bndfun
from chebpy.chebfun import Chebfun
from chebpy.chebtech import Chebtech
from chebpy.utilities import Interval

if os.environ.get("CI") == "true":  # pragma: no cover
    matplotlib.use("Agg")


@pytest.fixture(scope="session", autouse=True)
def testfunctions() -> list:
    """Create a collection of test functions used throughout the test suite.

    Each function is represented as a tuple containing:
    1. The function itself
    2. A name for the function (used in test printouts)
    3. The expected adaptive degree on [-1,1]
    4. A boolean indicating whether the function has roots on the real line

    These functions are used to test various aspects of the chebpy library,
    particularly the approximation and evaluation capabilities.

    Returns:
        list: List of tuples, each containing a test function and its metadata.
    """
    test_functions = []
    fun_details = [
        # Use the convention:
        #  function,
        #  name for the test printouts,
        #  expected adaptive degree on [-1,1],
        #  Any roots on the real line?
        (lambda x: x**3 + x**2 + x + 1.1, "poly3(x)", 4, True),
        (lambda x: np.exp(x), "exp(x)", 15, False),
        (lambda x: np.sin(x), "sin(x)", 14, True),
        (lambda x: 0.2 + 0.1 * np.sin(x), "(.2+.1*sin(x))", 14, False),
        (lambda x: np.cos(20 * x), "cos(20x)", 51, True),
        (lambda x: 0.0 * x + 1.0, "constfun", 1, False),
        (lambda x: 0.0 * x, "zerofun", 1, True),
    ]
    for k, items in enumerate(fun_details):
        fun = items[0]
        fun.__name__ = items[1]
        test_functions.append((fun, items[2], items[3]))

    return test_functions


@pytest.fixture
def random_points() -> np.ndarray:
    """Create an array of random points in [-1, 1] for testing.

    This fixture creates an array of 1000 random points in the interval [-1, 1]
    that can be used for evaluating and testing Chebtech objects.

    Returns:
        numpy.ndarray: Array of 1000 random points in [-1, 1]
    """
    # Ensure reproducibility
    rng = np.random.default_rng(0)

    return -1 + 2 * rng.random(1000)


@pytest.fixture()
def binops():
    """Binary operators for testing algebraic operations."""
    return [operator.add, operator.mul, operator.sub, operator.truediv]


@pytest.fixture()
def div_binops():
    """Binary operators for testing division."""
    return (operator.truediv,)


@pytest.fixture
def emptyfun(request):
    """Create an empty function object for testing.

    This generic fixture creates an empty function object of the appropriate type
    based on the test module that requests it. It automatically determines the
    correct class (Bndfun, Chebfun, or Chebtech) based on the module name.

    Args:
        request: The pytest request object, used to determine the calling module

    Returns:
        Union[Bndfun, Chebfun, Chebtech]: An empty function object of the appropriate type
    """
    module_name = request.module.__name__

    if "bndfun" in module_name:
        fun = Bndfun.initempty()
    elif "chebfun" in module_name:
        fun = Chebfun.initempty()
    elif "chebtech" in module_name:
        fun = Chebtech.initempty()
    else:
        # Default to Chebfun if the module name doesn't match any specific type
        fun = Chebfun.initempty()

    assert not fun.isconst
    assert fun.isempty
    assert fun.vscale == 0.0
    assert fun.roots().size == 0

    return fun


@pytest.fixture
def constfun(request):
    """Create a constant function object for testing.

    This generic fixture creates a constant function object of the appropriate type
    based on the test module that requests it. It automatically determines the
    correct class (Bndfun, Chebfun, or Chebtech) based on the module name.
    The constant value is set to 1.0.

    Args:
        request: The pytest request object, used to determine the calling module

    Returns:
        Union[Bndfun, Chebfun, Chebtech]: A constant function object of the appropriate type
    """
    module_name = request.module.__name__

    if "bndfun" in module_name:
        # Bndfun requires an interval

        fun = Bndfun.initconst(1.0, Interval())
    elif "chebfun" in module_name:
        fun = Chebfun.initconst(1.0)
    elif "chebtech" in module_name:
        fun = Chebtech.initconst(1.0)
    else:
        # Default to Chebfun if the module name doesn't match any specific type
        fun = Chebfun.initconst(1.0)

    assert fun.isconst
    assert not fun.isempty
    assert fun.roots().size == 0

    return fun


@pytest.fixture
def complexfun(request):
    """Create a complex function object for testing.

    This generic fixture creates a complex function object of the appropriate type
    based on the test module that requests it. It automatically determines the
    correct class (Bndfun, Chebfun, or Chebtech) based on the module name.
    The complex function is set to exp(π·i·x).

    Args:
        request: The pytest request object, used to determine the calling module

    Returns:
        Union[Bndfun, Chebfun, Chebtech]: A complex function object of the appropriate type
    """
    module_name = request.module.__name__

    if "bndfun" in module_name:
        # Bndfun requires an interval

        fun = Bndfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), Interval(-1, 1))
    elif "chebfun" in module_name:
        fun = Chebfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), [-1, 1])
    elif "chebtech" in module_name:
        fun = Chebtech.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x))
    else:
        # Default to Chebfun if the module name doesn't match any specific type
        fun = Chebfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), [-1, 1])

    assert fun.iscomplex
    assert not fun.isempty
    assert not fun.isconst
    assert fun.roots().size == 0

    return fun


@dataclasses.dataclass(frozen=True)
class TestFunction:
    """Container for test functions."""

    cheb: Chebfun | Chebtech | Bndfun
    raw: callable
    degree: int
    has_roots: bool


@pytest.fixture
def ttt(request, testfunctions):
    """Create a collection of test functions for testing."""
    module_name = request.module.__name__

    t_functions = []

    if "bndfun" in module_name:
        interval = Interval()
        for fun, degree, roots in testfunctions:
            bndfun = Bndfun.initfun_adaptive(fun, interval)
            bndfun.name = fun.__name__
            t_functions.append(TestFunction(cheb=bndfun, raw=fun, degree=degree, has_roots=roots))

    if "chebfun" in module_name:
        for fun, degree, roots in testfunctions:
            chebfun = Chebfun.initfun_adaptive(fun, [-1, 1])
            chebfun.name = fun.__name__
            t_functions.append(TestFunction(cheb=chebfun, raw=fun, degree=degree, has_roots=roots))

    if "chebtech" in module_name:
        for fun, degree, roots in testfunctions:
            chebtech = Chebtech.initfun_adaptive(fun)
            chebtech.name = fun.__name__
            t_functions.append(TestFunction(cheb=chebtech, raw=fun, degree=degree, has_roots=roots))
    return t_functions
