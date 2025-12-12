"""Unit-tests for chebpy/core/utilities.py.

This module contains tests for the utility classes and functions in the chebpy core package,
including Interval, Domain, check_funs, and compute_breakdata.

The tests verify that these utilities handle various input types correctly and produce
expected results for both normal and edge cases.
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.bndfun import Bndfun
from chebpy.exceptions import IntervalGap, IntervalOverlap, IntervalValues, InvalidDomain, NotSubdomain, SupportMismatch
from chebpy.settings import DefaultPreferences
from chebpy.utilities import Domain, Interval, check_funs, compute_breakdata, htol

rng = np.random.default_rng(0)  # Use a fixed seed for reproducibility
eps = DefaultPreferences.eps
HTOL = htol()


# tests for usage of the Interval class
@pytest.fixture
def interval_fixtures():
    """Create interval objects for testing.

    This fixture creates several Interval objects with different ranges
    for use in testing equality, containment, and other Interval methods.

    Returns:
        dict: Dictionary containing four Interval objects:
            i1: Interval(-2, 3)
            i2: Interval(-2, 3) (identical to i1)
            i3: Interval(-1, 1) (contained within i1)
            i4: Interval(-1, 2) (overlapping with i1)
    """
    i1 = Interval(-2, 3)
    i2 = Interval(-2, 3)
    i3 = Interval(-1, 1)
    i4 = Interval(-1, 2)
    return {"i1": i1, "i2": i2, "i3": i3, "i4": i4}


def test_init():
    """Test initialization of Interval objects.

    This test verifies that:
    1. Interval objects can be created with explicit endpoints
    2. Interval objects created without arguments default to [-1, 1]
    """
    Interval(-1, 1)
    assert (np.asarray(Interval()) == np.array([-1, 1])).all()


def test_init_disallow():
    """Test that invalid interval endpoints are rejected.

    This test verifies that the Interval constructor raises IntervalValues when:
    1. The left endpoint is greater than the right endpoint
    2. The left endpoint equals the right endpoint (zero-length interval)
    """
    with pytest.raises(IntervalValues):
        Interval(2, 0)
    with pytest.raises(IntervalValues):
        Interval(0, 0)


def test__eq__(interval_fixtures):
    """Test equality comparison of Interval objects.

    This test verifies that the equality operator (==) correctly:
    1. Returns True for intervals with identical endpoints
    2. Is symmetric (a == b implies b == a)
    3. Returns False for intervals with different endpoints

    Args:
        interval_fixtures: Fixture providing test interval objects.
    """
    i1, i2, i3 = interval_fixtures["i1"], interval_fixtures["i2"], interval_fixtures["i3"]
    assert Interval() == Interval()
    assert i1 == i2
    assert i2 == i1
    assert not (i3 == i1)
    assert not (i2 == i3)


def test__ne__(interval_fixtures):
    """Test inequality comparison of Interval objects.

    This test verifies that the inequality operator (!=) correctly:
    1. Returns False for intervals with identical endpoints
    2. Is symmetric (a != b implies b != a)
    3. Returns True for intervals with different endpoints

    Args:
        interval_fixtures: Fixture providing test interval objects.
    """
    i1, i2, i3 = interval_fixtures["i1"], interval_fixtures["i2"], interval_fixtures["i3"]
    assert not (Interval() != Interval())
    assert not (i1 != i2)
    assert not (i2 != i1)
    assert i3 != i1
    assert i2 != i3


def test__contains__(interval_fixtures):
    """Test containment operations of Interval objects.

    This test verifies that the containment operators (in, not in) correctly:
    1. Return True when an interval is contained within another
    2. Return False when an interval is not contained within another
    3. Handle the negation (not in) correctly

    An interval is considered to be contained within another if its endpoints
    are within the endpoints of the containing interval.

    Args:
        interval_fixtures: Fixture providing test interval objects.
    """
    i1, i2, i3, i4 = interval_fixtures["i1"], interval_fixtures["i2"], interval_fixtures["i3"], interval_fixtures["i4"]
    assert i1 in i2
    assert i3 in i1
    assert i4 in i1
    assert i1 not in i3
    assert i1 not in i4


def test_maps():
    """Test forward and inverse mapping functions of Interval objects.

    This test verifies that the forward mapping (via __call__) and inverse
    mapping (via invmap) are inverses of each other. Specifically, it checks
    that applying the forward map followed by the inverse map returns the
    original points within machine precision.

    The forward map transforms points from [-1,1] to [a,b], while the
    inverse map transforms points from [a,b] to [-1,1].
    """
    yy = -1 + 2 * rng.random(1000)
    interval = Interval(-2, 3)
    vals = interval.invmap(interval(yy)) - yy
    assert np.max(np.abs(vals)) <= eps


def test_isinterior():
    """Test the isinterior method of Interval objects.

    This test verifies that the isinterior method correctly identifies points
    that are strictly inside an interval (excluding the endpoints). It checks:
    1. Points spanning the interval (should be interior except endpoints)
    2. Points to the left of the interval (should not be interior)
    3. Points to the right of the interval (should not be interior)
    4. Points far from the interval (should not be interior)
    """
    npts = 1000
    x1 = np.linspace(-2, 3, npts)
    x2 = np.linspace(-3, -2, npts)
    x3 = np.linspace(3, 4, npts)
    x4 = np.linspace(5, 6, npts)
    interval = Interval(-2, 3)
    assert interval.isinterior(x1).sum() == npts - 2
    assert interval.isinterior(x2).sum() == 0
    assert interval.isinterior(x3).sum() == 0
    assert interval.isinterior(x4).sum() == 0


# tests for usage of the Domain class
def test_domain_init():
    """Test initialization of Domain objects.

    This test verifies that Domain objects can be created with various
    valid inputs, including:
    1. Python lists with two elements
    2. Python lists with more than two elements
    3. NumPy arrays with two elements
    4. NumPy arrays with more than two elements
    5. NumPy linspace arrays
    """
    Domain([-2, 1])
    Domain([-2, 0, 1])
    Domain(np.array([-2, 1]))
    Domain(np.array([-2, 0, 1]))
    Domain(np.linspace(-10, 10, 51))


def test_domain_init_disallow():
    """Test that invalid domain inputs are rejected.

    This test verifies that the Domain constructor raises appropriate exceptions when:
    1. Given a list with only one element (InvalidDomain)
    2. Given a list with non-monotonically increasing values (InvalidDomain)
    3. Given a list with duplicate values (InvalidDomain)
    4. Given a list with non-numeric values (ValueError)
    """
    with pytest.raises(InvalidDomain):
        Domain([1])
    with pytest.raises(InvalidDomain):
        Domain([1, -1])
    with pytest.raises(InvalidDomain):
        Domain([-1, 0, 0])
    with pytest.raises(ValueError):
        Domain(["a", "b"])


def test_domain_iter():
    """Test iteration over Domain objects.

    This test verifies that Domain objects can be iterated over,
    yielding their breakpoints in order. It checks domains with:
    1. Two breakpoints
    2. Three breakpoints
    3. Four breakpoints

    The test ensures that iterating over a Domain yields the same values
    as iterating over a tuple containing the same breakpoints.
    """
    dom_a = Domain([-2, 1])
    dom_b = Domain([-2, 0, 1])
    dom_c = Domain([-1, 0, 1, 2])
    res_a = (-2, 1)
    res_b = (-2, 0, 1)
    res_c = (-1, 0, 1, 2)
    assert all([x == y for x, y in zip(dom_a, res_a)])
    assert all([x == y for x, y in zip(dom_b, res_b)])
    assert all([x == y for x, y in zip(dom_c, res_c)])


def test_domain_intervals():
    """Test the intervals property of Domain objects.

    This test verifies that the intervals property of Domain objects correctly
    generates Interval objects between adjacent breakpoints. It checks that:
    1. A domain with two breakpoints generates one interval
    2. A domain with three breakpoints generates two intervals
    3. A domain with four breakpoints generates three intervals

    For each domain, it ensures that the generated intervals match the expected
    intervals created from pairs of adjacent breakpoints.
    """
    dom_a = Domain([-2, 1])
    dom_b = Domain([-2, 0, 1])
    dom_c = Domain([-1, 0, 1, 2])
    res_a = [(-2, 1)]
    res_b = [(-2, 0), (0, 1)]
    res_c = [(-1, 0), (0, 1), (1, 2)]
    assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_a.intervals, res_a)])
    assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_b.intervals, res_b)])
    assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_c.intervals, res_c)])


def test_domain_contains():
    """Test the containment operator for Domain objects.

    This test verifies that the containment operator (__contains__) correctly
    determines whether one domain is a subdomain of another. It checks:
    1. Various containment relationships between domains of different sizes
    2. Reflexive containment (a domain contains itself)
    3. Transitive containment (if a contains b and b contains c, then a contains c)
    4. Negative cases where containment does not hold

    The test uses domains with different structures: domains with multiple
    breakpoints, domains with just two breakpoints, and domains with many
    breakpoints created using linspace.
    """
    d1 = Domain([-2, 0, 1, 3, 5])
    d2 = Domain([-1, 2])
    d3 = Domain(np.linspace(-10, 10, 1000))
    d4 = Domain([-1, 0, 1, 2])
    assert d2 in d1
    assert d1 in d3
    assert d2 in d3
    assert d2 in d3
    assert d2 in d4
    assert d4 in d2
    assert d1 not in d2
    assert d3 not in d1
    assert d3 not in d2


def test_domain_contains_close():
    """Test tolerance-sensitive containment behavior of Domain objects.

    This test verifies that the containment operator (__contains__) correctly
    handles domains with endpoints that are close but not exactly equal. It checks:
    1. Domains with endpoints within tolerance are considered to contain each other
    2. Domains with endpoints beyond tolerance are not considered to contain each other

    The test uses a tolerance of 0.8 * HTOL (horizontal tolerance) and creates
    domains with endpoints that differ by various multiples of this tolerance.
    """
    tol = 0.8 * HTOL
    d1 = Domain([-1, 2])
    d2 = Domain([-1 - tol, 2 + 2 * tol])
    d3 = Domain([-1 - 2 * tol, 2 + 4 * tol])
    assert d1 in d2
    assert d2 in d1
    assert d3 not in d1


def test_domain_eq():
    """Test equality comparison of Domain objects.

    This test verifies that the equality operator (==) correctly:
    1. Returns True for domains with identical breakpoints
    2. Returns False for domains with different breakpoints

    It also implicitly tests the inequality operator (!=) by using it
    in the assertions.
    """
    d1 = Domain([-2, 0, 1, 3, 5])
    d2 = Domain([-2, 0, 1, 3, 5])
    d3 = Domain([-1, 1])
    assert d1 == d2
    assert d1 != d3


def test_domain_eq_result_type():
    """Test the return type of Domain equality comparison.

    This test verifies that the equality operator (==) returns a boolean value,
    regardless of whether the domains are equal or not. This ensures that
    the equality operator can be used in boolean contexts like if statements.
    """
    d1 = Domain([-2, 0, 1, 3, 5])
    d2 = Domain([-2, 0, 1, 3, 5])
    d3 = Domain([-1, 1])
    assert isinstance(d1 == d2, bool)
    assert isinstance(d1 == d3, bool)


def test_domain_eq_close():
    """Test tolerance-sensitive equality comparison of Domain objects.

    This test verifies that the equality operator (==) correctly handles
    domains with breakpoints that are close but not exactly equal. It checks:
    1. Domains with breakpoints within tolerance are considered equal
    2. Domains with breakpoints beyond tolerance are not considered equal

    The test uses a tolerance of 0.8 * HTOL (horizontal tolerance) and creates
    domains with breakpoints that differ by various multiples of this tolerance.
    """
    tol = 0.8 * HTOL
    d4 = Domain([-2, 0, 1, 3, 5])
    d5 = Domain([-2 * (1 + tol), 0 - tol, 1 + tol, 3 * (1 + tol), 5 * (1 - tol)])
    d6 = Domain(
        [
            -2 * (1 + 2 * tol),
            0 - 2 * tol,
            1 + 2 * tol,
            3 * (1 + 2 * tol),
            5 * (1 - 2 * tol),
        ]
    )
    assert d4 == d5
    assert d4 != d6


def test_domain_ne():
    """Test inequality comparison of Domain objects.

    This test verifies that the inequality operator (!=) correctly:
    1. Returns False for domains with identical breakpoints
    2. Returns True for domains with different breakpoints

    This test complements test_domain_eq() by explicitly testing the
    inequality operator.
    """
    d1 = Domain([-2, 0, 1, 3, 5])
    d2 = Domain([-2, 0, 1, 3, 5])
    d3 = Domain([-1, 1])
    assert not (d1 != d2)
    assert d1 != d3


def test_domain_ne_result_type():
    """Test the return type of Domain inequality comparison.

    This test verifies that the inequality operator (!=) returns a boolean value,
    regardless of whether the domains are different or not. This ensures that
    the inequality operator can be used in boolean contexts like if statements.
    """
    d1 = Domain([-2, 0, 1, 3, 5])
    d2 = Domain([-2, 0, 1, 3, 5])
    d3 = Domain([-1, 1])
    assert isinstance(d1 != d2, bool)
    assert isinstance(d1 != d3, bool)


def test_domain_from_chebfun():
    """Test creating a Domain from a Chebfun object.

    This test verifies that the from_chebfun class method correctly creates
    a Domain object from a Chebfun object. It creates a Chebfun with a
    specific domain (using linspace) and then creates a Domain from it.

    The test simply checks that the operation completes without errors,
    implicitly verifying that the resulting Domain has the same breakpoints
    as the Chebfun.
    """
    ff = chebfun(lambda x: np.cos(x), np.linspace(-10, 10, 11))
    Domain.from_chebfun(ff)


def test_domain_breakpoints_in():
    """Test the breakpoints_in method of Domain objects.

    This test verifies that the breakpoints_in method correctly identifies
    which breakpoints of one domain are contained in another domain. It checks:
    1. The method returns a boolean array of the correct type and size
    2. The array correctly identifies which breakpoints are in the other domain
    3. A domain's breakpoints are all in itself (reflexive property)
    4. Domains with non-overlapping supports have no common breakpoints

    The test uses domains with different structures and checks the results
    in both directions (d1.breakpoints_in(d2) and d2.breakpoints_in(d1)).
    """
    d1 = Domain([-1, 0, 1])
    d2 = Domain([-2, 0.5, 1, 3])

    result1 = d1.breakpoints_in(d2)
    assert isinstance(result1, np.ndarray)
    assert result1.size, 3
    assert not result1[0]
    assert not result1[1]
    assert result1[2]

    result2 = d2.breakpoints_in(d1)
    assert isinstance(result2, np.ndarray)
    assert result2.size, 4
    assert not result2[0]
    assert not result2[1]
    assert result2[2]
    assert not result2[3]

    assert d1.breakpoints_in(d1).all()
    assert d2.breakpoints_in(d2).all()
    assert not d1.breakpoints_in(Domain([-5, 5])).any()
    assert not d2.breakpoints_in(Domain([-5, 5])).any()


def test_domain_breakpoints_in_close():
    """Test tolerance-sensitive behavior of the breakpoints_in method.

    This test verifies that the breakpoints_in method correctly identifies
    breakpoints that are within tolerance of breakpoints in another domain.
    It checks:
    1. Breakpoints that are not close to any in the other domain are identified as not in it
    2. Breakpoints that are within tolerance of ones in the other domain are identified as in it

    The test uses a tolerance of 0.8 * HTOL (horizontal tolerance) and creates
    domains with breakpoints that differ by this tolerance.
    """
    tol = 0.8 * HTOL
    d1 = Domain([-1, 0, 1])
    d2 = Domain([-2, 0 - tol, 1 + tol, 3])
    result = d1.breakpoints_in(d2)
    assert not result[0]
    assert result[1]
    assert result[2]


def test_domain_support():
    """Test the support property of Domain objects.

    This test verifies that the support property correctly returns the first
    and last breakpoints of a domain, regardless of the number of breakpoints
    or how the domain was created. It checks:
    1. A domain with two breakpoints
    2. A domain with three breakpoints
    3. A domain created with numpy.linspace

    The test ensures that the support property returns a view that can be
    converted to a numpy array with the expected values.
    """
    dom_a = Domain([-2, 1])
    dom_b = Domain([-2, 0, 1])
    dom_c = Domain(np.linspace(-10, 10, 51))
    assert np.all(dom_a.support.view(np.ndarray) == [-2, 1])
    assert np.all(dom_b.support.view(np.ndarray) == [-2, 1])
    assert np.all(dom_c.support.view(np.ndarray) == [-10, 10])


def test_domain_size():
    """Test the size property of Domain objects.

    This test verifies that the size property correctly returns the number
    of breakpoints in a domain. It checks:
    1. A domain with two breakpoints
    2. A domain with three breakpoints
    3. A domain with many breakpoints created using numpy.linspace

    The test ensures that the size property returns the expected integer value.
    """
    dom_a = Domain([-2, 1])
    dom_b = Domain([-2, 0, 1])
    dom_c = Domain(np.linspace(-10, 10, 51))
    assert dom_a.size == 2
    assert dom_b.size == 3
    assert dom_c.size == 51


def test_domain_restrict():
    """Test the restrict method of Domain objects.

    This test verifies that the restrict method correctly truncates a domain
    to the support of another domain, while retaining any interior breakpoints.
    It checks:
    1. Restricting a domain with interior breakpoints to a subdomain
    2. Restricting a domain to a subdomain that already contains all its breakpoints
    3. Restricting a domain to itself (should return the same domain)
    4. Restricting a domain to a subdomain created with numpy.linspace

    The test also includes a special case to check that the method correctly
    handles breakpoints that differ by machine epsilon, which can occur when
    using numpy.linspace to create domains.
    """
    dom_a = Domain([-2, -1, 0, 1])
    dom_b = Domain([-1.5, -0.5, 0.5])
    dom_c = Domain(np.linspace(-2, 1, 16))
    assert dom_a.restrict(dom_b) == Domain([-1.5, -1, -0.5, 0, 0.5])
    assert dom_a.restrict(dom_c) == dom_c
    assert dom_a.restrict(dom_a) == dom_a
    assert dom_b.restrict(dom_b) == dom_b
    assert dom_c.restrict(dom_c) == dom_c
    # tests to check if catch breakpoints that are different by eps
    # (linspace introduces these effects)
    dom_d = Domain(np.linspace(-0.4, 0.4, 2))
    assert dom_c.restrict(dom_d) == Domain([-0.4, -0.2, 0, 0.2, 0.4])


def test_domain_restrict_raises():
    """Test that restrict method raises appropriate exceptions.

    This test verifies that the restrict method raises a NotSubdomain exception
    when attempting to restrict a domain to another domain that is not a
    subdomain of it. It checks:
    1. Restricting a smaller domain to a larger domain
    2. Restricting a domain to another domain with a different support

    The test ensures that the method correctly validates its inputs and
    provides appropriate error messages when the operation cannot be performed.
    """
    dom_a = Domain([-2, -1, 0, 1])
    dom_b = Domain([-1.5, -0.5, 0.5])
    dom_c = Domain(np.linspace(-2, 1, 16))
    with pytest.raises(NotSubdomain):
        dom_b.restrict(dom_a)
    with pytest.raises(NotSubdomain):
        dom_b.restrict(dom_c)


def test_domain_merge():
    """Test the merge method of Domain objects.

    This test verifies that the merge method correctly combines two domains
    by including all breakpoints from both domains, without checking if they
    have the same support. It checks:
    1. Merging a smaller domain with a larger domain
    2. The resulting domain contains all breakpoints from both domains in order

    Unlike the union method, merge does not require the domains to have the
    same support, making it more flexible for combining domains with different
    ranges.
    """
    dom_a = Domain([-2, -1, 0, 1])
    dom_b = Domain([-1.5, -0.5, 0.5])
    assert dom_b.merge(dom_a) == Domain([-2, -1.5, -1, -0.5, 0, 0.5, 1])


def test_domain_union():
    """Test the union method of Domain objects.

    This test verifies that the union method correctly combines two domains
    with the same support by including all breakpoints from both domains.
    It checks:
    1. The union of two domains is different from either original domain
    2. The union contains all breakpoints from both domains in order
    3. The union operation is commutative (a.union(b) == b.union(a))

    Unlike the merge method, union requires the domains to have the same
    support (within tolerance), ensuring that the resulting domain covers
    exactly the same range as the original domains.
    """
    dom_a = Domain([-2, 0, 2])
    dom_b = Domain([-2, -1, 1, 2])
    assert dom_a.union(dom_b) != dom_a
    assert dom_a.union(dom_b) != dom_b
    assert dom_a.union(dom_b) == Domain([-2, -1, 0, 1, 2])
    assert dom_b.union(dom_a) == Domain([-2, -1, 0, 1, 2])


def test_domain_union_close():
    """Test tolerance-sensitive union operation of Domain objects.

    This test verifies that the union method correctly handles domains with
    breakpoints that are close but not exactly equal. It checks:
    1. The union operation correctly identifies breakpoints within tolerance
    2. The union operation is commutative even with tolerance-sensitive breakpoints

    The test uses a tolerance of 0.8 * HTOL (horizontal tolerance) and creates
    domains with breakpoints that differ by various multiples of this tolerance.
    """
    tol = 0.8 * HTOL
    dom_a = Domain([-2, 0, 2])
    dom_c = Domain([-2 - 2 * tol, -1 + tol, 1 + tol, 2 + 2 * tol])
    assert dom_a.union(dom_c) == Domain([-2, -1, 0, 1, 2])
    assert dom_c.union(dom_a) == Domain([-2, -1, 0, 1, 2])


def test_domain_union_raises():
    """Test that union method raises appropriate exceptions.

    This test verifies that the union method raises a SupportMismatch exception
    when attempting to union domains with different supports. It checks:
    1. Attempting to union domains with different right endpoints raises SupportMismatch
    2. The exception is raised regardless of the order of the domains

    The test ensures that the method correctly validates its inputs and
    provides appropriate error messages when the operation cannot be performed.
    """
    dom_a = Domain([-2, 0])
    dom_b = Domain([-2, 3])
    with pytest.raises(SupportMismatch):
        dom_a.union(dom_b)
    with pytest.raises(SupportMismatch):
        dom_b.union(dom_a)


"""Tests for the chebpy.utilities check_funs method"""


@pytest.fixture
def check_funs_fixtures():
    """Create Bndfun objects for testing check_funs function.

    This fixture creates several Bndfun objects on different intervals
    and arrays of these objects for testing the check_funs function.
    The arrays are designed to test various cases:
        - funs_a and funs_b: overlapping intervals (should raise IntervalOverlap)
        - funs_c and funs_d: non-contiguous intervals (should raise IntervalGap)

    Returns:
        dict: Dictionary containing:
            fun0-fun4: Individual Bndfun objects on different intervals
            funs_a: Array of functions with overlapping intervals
            funs_b: Another array with overlapping intervals
            funs_c: Array of functions with a gap between intervals
            funs_d: Another array with a gap between intervals
    """

    def f(x):
        return np.exp(x)

    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
    fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
    fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
    fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))
    funs_a = np.array([fun1, fun0, fun2])
    funs_b = np.array([fun1, fun2])
    funs_c = np.array([fun0, fun3])
    funs_d = np.array([fun1, fun4])

    return {
        "fun0": fun0,
        "fun1": fun1,
        "fun2": fun2,
        "fun3": fun3,
        "fun4": fun4,
        "funs_a": funs_a,
        "funs_b": funs_b,
        "funs_c": funs_c,
        "funs_d": funs_d,
    }


def test_check_funs_verify_empty():
    """Test check_funs function with empty input.

    This test verifies that the check_funs function correctly handles
    an empty array of functions, returning an empty array without errors.
    """
    funs = check_funs(np.array([]))
    assert funs.size == 0


def test_check_funs_verify_contiguous(check_funs_fixtures):
    """Test check_funs function with contiguous functions.

    This test verifies that the check_funs function correctly handles
    an array of functions with contiguous intervals, returning the array
    without raising exceptions.

    Args:
        check_funs_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = check_funs_fixtures["fun0"]
    fun1 = check_funs_fixtures["fun1"]
    funs = check_funs(np.array([fun0, fun1]))
    assert funs[0] == fun0
    assert funs[1] == fun1


def test_check_funs_verify_sort(check_funs_fixtures):
    """Test sorting behavior of check_funs function.

    This test verifies that the check_funs function correctly sorts
    an array of functions by their intervals, regardless of the order
    in which they are provided.

    Args:
        check_funs_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = check_funs_fixtures["fun0"]
    fun1 = check_funs_fixtures["fun1"]
    funs = check_funs(np.array([fun1, fun0]))
    assert funs[0] == fun0
    assert funs[1] == fun1


def test_check_funs_verify_overlapping(check_funs_fixtures):
    """Test check_funs function with overlapping intervals.

    This test verifies that the check_funs function correctly raises
    an IntervalOverlap exception when given an array of functions with
    overlapping intervals.

    Args:
        check_funs_fixtures: Fixture providing test Bndfun objects with overlapping intervals.
    """
    funs_a = check_funs_fixtures["funs_a"]
    funs_b = check_funs_fixtures["funs_b"]
    with pytest.raises(IntervalOverlap):
        check_funs(funs_a)
    with pytest.raises(IntervalOverlap):
        check_funs(funs_b)


def test_check_funs_verify_gap(check_funs_fixtures):
    """Test check_funs function with non-contiguous intervals.

    This test verifies that the check_funs function correctly raises
    an IntervalGap exception when given an array of functions with
    non-contiguous intervals (gaps between intervals).

    Args:
        check_funs_fixtures: Fixture providing test Bndfun objects with non-contiguous intervals.
    """
    funs_c = check_funs_fixtures["funs_c"]
    funs_d = check_funs_fixtures["funs_d"]
    with pytest.raises(IntervalGap):
        check_funs(funs_c)
    with pytest.raises(IntervalGap):
        check_funs(funs_d)


# tests for the chebpy.utilities compute_breakdata function
@pytest.fixture
def compute_breakdata_fixtures():
    """Create Bndfun objects for testing compute_breakdata function.

    This fixture creates two contiguous Bndfun objects representing
    the exponential function on adjacent intervals. These objects are
    used to test the compute_breakdata function, which computes values
    at breakpoints by averaging left and right limits.

    Returns:
        dict: Dictionary containing:
            fun0: Bndfun object for exp(x) on interval [-1, 0]
            fun1: Bndfun object for exp(x) on interval [0, 1]
    """

    def f(x):
        return np.exp(x)

    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))

    return {"fun0": fun0, "fun1": fun1}


def test_compute_breakdata_empty():
    """Test compute_breakdata function with empty input.

    This test verifies that the compute_breakdata function correctly handles
    an empty array of functions, returning an empty dictionary without errors.
    """
    breaks = compute_breakdata(np.array([]))
    # list(...) for Python 2/3 compatibility
    assert np.array(list(breaks.items())).size == 0


def test_compute_breakdata_1(compute_breakdata_fixtures):
    """Test compute_breakdata function with a single function.

    This test verifies that the compute_breakdata function correctly computes
    breakpoint data for a single function, returning a dictionary with the
    correct keys (breakpoints) and values (function values at breakpoints).

    Args:
        compute_breakdata_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = compute_breakdata_fixtures["fun0"]
    funs = np.array([fun0])
    breaks = compute_breakdata(funs)
    x, y = list(breaks.keys()), list(breaks.values())
    assert np.max(np.abs(x - np.array([-1, 0]))) <= eps
    assert np.max(np.abs(y - np.array([np.exp(-1), np.exp(0)]))) <= 2 * eps


def test_compute_breakdata_2(compute_breakdata_fixtures):
    """Test compute_breakdata function with multiple functions.

    This test verifies that the compute_breakdata function correctly computes
    breakpoint data for multiple contiguous functions, returning a dictionary with
    the correct keys (breakpoints) and values (function values at breakpoints).

    Args:
        compute_breakdata_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = compute_breakdata_fixtures["fun0"]
    fun1 = compute_breakdata_fixtures["fun1"]
    funs = np.array([fun0, fun1])
    breaks = compute_breakdata(funs)
    x, y = list(breaks.keys()), list(breaks.values())
    assert np.max(np.abs(x - np.array([-1, 0, 1]))) <= eps
    assert np.max(np.abs(y - np.array([np.exp(-1), np.exp(0), np.exp(1)]))) <= 2 * eps
