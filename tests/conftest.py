"""Configuration and fixtures for pytest.

This module contains global pytest configuration and fixtures that are
available to all test modules. It handles special configurations needed
for different environments.

Specifically, it:
- Sets the matplotlib backend to 'Agg' (a non-interactive backend) when
  running in a CI environment, which is necessary for running tests that
  generate plots without a display.

Note:
    The 'Agg' backend is used because it doesn't require a graphical display,
    making it suitable for headless CI environments.
"""
import os
import matplotlib
import numpy as np
import pytest

from chebpy.core.settings import DefaultPreferences

if os.environ.get("CI") == "true":   # pragma: no cover
    matplotlib.use("Agg")

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps




@pytest.fixture(scope="session", autouse=True)
def testfunctions():
    # Collection of test functions used throughout the test suite.
    #
    # Each function is represented as a tuple containing:
    # 1. The function itself
    # 2. A name for the function (used in test printouts)
    # 3. The Matlab chebfun adaptive degree on [-1,1]
    # 4. A boolean indicating whether the function has roots on the real line
    #
    # These functions are used to test various aspects of the chebpy library,
    # particularly the approximation and evaluation capabilities.

    test_functions = []
    fun_details = [
        # (
        #  function,
        #  name for the test printouts,
        #  Matlab chebfun adaptive degree on [-1,1],
        #  Any roots on the real line?
        # )
        (lambda x: x ** 3 + x ** 2 + x + 1.1, "poly3(x)", 4, True),
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


def scaled_tol(n):
    """Calculate a scaled tolerance based on the size of the input.

    This function returns a tolerance that increases with the size of the input,
    which is useful for tests where the expected error grows with problem size.

    Args:
        n (int): Size parameter, typically the length of an array.

    Returns:
        float: Scaled tolerance value.
    """
    tol = 5e1 * eps if n < 20 else np.log(n) ** 2.5 * eps
    return tol
