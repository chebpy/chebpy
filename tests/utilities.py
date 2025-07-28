import numpy as np

eps = np.finfo(float).eps


def infnorm(x):
    """Calculate the infinity norm of an array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        float: The infinity norm (maximum absolute value) of the input.
    """
    return np.linalg.norm(x, np.inf)


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


# bespoke test generators
def inf_norm_less_than_tol(a, b, tol):
    """Create a test function that asserts the infinity norm of (a-b) is less than tol.

    This is a test generator that returns a function which, when called,
    asserts that the infinity norm of the difference between arrays a and b
    is less than or equal to the specified tolerance.

    Args:
        a (numpy.ndarray): First array to compare.
        b (numpy.ndarray): Second array to compare.
        tol (float): Tolerance for the comparison.

    Returns:
        function: A function that when called performs the assertion.
    """
    def asserter():
        assert infnorm(a - b) <= tol

    return asserter


def joukowsky(z):
    """Apply the Joukowsky transformation to z.

    The Joukowsky transformation maps the unit circle to an ellipse and is used
    in complex analysis and fluid dynamics. It is defined as f(z) = 0.5 * (z + 1/z).

    Args:
        z (complex or numpy.ndarray): Complex number or array of complex numbers.

    Returns:
        complex or numpy.ndarray: Result of the Joukowsky transformation.
    """
    return 0.5 * (z + 1 / z)


# test functions
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
testfunctions = []
fun_details = [
    # (
    #  function,
    #  name for the test printouts,
    #  Matlab chebfun adaptive degree on [-1,1],
    #  Any roots on the real line?
    # )
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
    testfunctions.append((fun, items[2], items[3]))

# TODO: check these lengths against Chebfun
# TODO: more examples
