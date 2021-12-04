import numpy as np

eps = np.finfo(float).eps


def infnorm(x):
    return np.linalg.norm(x, np.inf)


def scaled_tol(n):
    tol = 5e1 * eps if n < 20 else np.log(n) ** 2.5 * eps
    return tol


# bespoke test generators
def infNormLessThanTol(a, b, tol):
    def asserter(self):
        self.assertLessEqual(infnorm(a - b), tol)

    return asserter


def joukowsky(z):
    return 0.5 * (z + 1 / z)


# test functions
testfunctions = []
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
    testfunctions.append((fun, items[2], items[3]))

# TODO: check these lengths against Chebfun
# TODO: more examples
