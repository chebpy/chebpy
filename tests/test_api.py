"""Tests for the api module.

This module tests the user-facing factory functions in chebpy.api:
- chebfun() constructor with various input types
- pwc() piecewise constant constructor
- Serialization (pickle) round-trip
- Version information
"""

import pickle  # nosec B403

import numpy as np
import pytest

import chebpy
from chebpy import chebfun, equifun, pwc
from chebpy.settings import DefaultPreferences


def test_chebfun_null_args() -> None:
    """Test creating an empty chebfun with no arguments."""
    assert chebfun().isempty


def test_chebfun_callable() -> None:
    """Test creating chebfun objects with callable functions."""
    n = 100
    d = np.array([-2, 0, 1])
    f1 = chebfun(np.sin)
    f2 = chebfun(np.sin, d)
    f3 = chebfun(np.sin, n=n)
    f4 = chebfun(np.sin, d, n)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == d
    assert f3.domain == DefaultPreferences.domain
    assert f4.domain == d

    # check lengths of f3 and f4
    assert f3.funs[0].size == n
    assert np.all([fun.size == n for fun in f4])


def test_chebfun_alphanum_char() -> None:
    """Test creating chebfun objects with alphanumeric characters."""
    n = 100
    d = np.array([-2, 0, 1])
    f1 = chebfun("x")
    f2 = chebfun("y", d)
    f3 = chebfun("z", n=n)
    f4 = chebfun("a", d, n)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == d
    assert f3.domain == DefaultPreferences.domain
    assert f4.domain == d

    # check lengths of f3 and f4
    assert np.sum([fun.size for fun in f3]) == n
    assert np.all([fun.size == n for fun in f4])


def test_chebfun_float_arg() -> None:
    """Test creating chebfun objects with float arguments."""
    d = np.array([-2, 0, 1])
    f1 = chebfun(3.14)
    f2 = chebfun("3.14")
    f3 = chebfun(2.72, d)
    f4 = chebfun("2.72", d)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == DefaultPreferences.domain
    assert f3.domain == d
    assert f4.domain == d

    # check all are constant
    assert f1.isconst
    assert f2.isconst
    assert f3.isconst
    assert f4.isconst


def test_chebfun_raises() -> None:
    """Test that invalid inputs raise appropriate exceptions."""
    with pytest.raises(ValueError, match="asdfasdf"):
        chebfun("asdfasdf")


def test_pwc() -> None:
    """Test creating piecewise constant functions."""
    dom = [-1, 0, 1]
    vals = [0, 1]
    f = pwc(dom, vals)
    assert f.funs.size == 2
    for fun, val in zip(f, vals, strict=False):
        assert fun.isconst
        assert fun.coeffs[0] == val


def test_pwc_defaults() -> None:
    """Test creating piecewise constant functions with default arguments."""
    f = pwc()
    assert f.funs.size == 2
    # Default values should be [0, 1] on domain [-1, 0, 1]
    assert f.funs[0].coeffs[0] == 0
    assert f.funs[1].coeffs[0] == 1
    assert np.array_equal(f.domain, np.array([-1, 0, 1]))


# -------------------------------------------------------------------
# equifun
# -------------------------------------------------------------------


def test_equifun_default_domain_approximates_sine() -> None:
    """Test equifun from endpoint-inclusive equispaced samples."""
    nodes = np.linspace(-1, 1, 41)
    f = equifun(np.sin(nodes))
    xx = np.linspace(-1, 1, 101)
    assert f.domain == DefaultPreferences.domain
    np.testing.assert_allclose(f(xx), np.sin(xx), atol=1e-12)


def test_equifun_custom_domain() -> None:
    """Test equifun on a custom finite interval."""
    domain = [0.0, np.pi]
    nodes = np.linspace(domain[0], domain[1], 35)
    f = equifun(np.cos(nodes), domain)
    xx = np.linspace(domain[0], domain[1], 101)
    assert f.domain == np.array(domain)
    np.testing.assert_allclose(f(xx), np.cos(xx), atol=1e-12)


def test_equifun_behaves_like_regular_chebfun() -> None:
    """Test equifun returns a regular Chebfun object."""
    nodes = np.linspace(-1, 1, 21)
    f = equifun(np.exp(nodes))
    g = chebfun(np.exp)
    assert isinstance(f, type(g))
    assert abs((f - g).sum()) < 1e-12


@pytest.mark.parametrize("n", [2, 3, 16, 40])
def test_equifun_interpolates_original_nodes(n: int) -> None:
    """Test values are reproduced at the original equispaced nodes."""
    nodes = np.linspace(-1, 1, n)
    values = np.exp(nodes) * np.cos(3 * nodes)
    f = equifun(values)
    np.testing.assert_allclose(f(nodes), values, atol=5e-12)


def test_equifun_single_sample_is_constant() -> None:
    """Test a single sample constructs a constant function."""
    f = equifun([2.5], [2.0, 5.0])
    assert f.isconst
    np.testing.assert_allclose(f(np.linspace(2.0, 5.0, 5)), 2.5)


def test_equifun_two_samples_are_linear() -> None:
    """Test two samples construct the endpoint line."""
    f = equifun([-3.0, 3.0])
    xx = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(f(xx), 3 * xx, atol=1e-13)


def test_equifun_complex_samples() -> None:
    """Test complex-valued equispaced samples."""
    nodes = np.linspace(-1, 1, 17)
    values = np.exp(1j * np.pi * nodes)
    f = equifun(values)
    assert f.iscomplex
    np.testing.assert_allclose(f(nodes), values, atol=5e-12)


def test_equifun_single_complex64_sample() -> None:
    """Test one complex64 sample preserves the imaginary part."""
    f = equifun(np.array([1.0 + 2.0j], dtype=np.complex64))
    assert f.iscomplex
    np.testing.assert_allclose(f(np.linspace(-1, 1, 5)), 1.0 + 2.0j)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([], "non-empty"),
        ([[1.0, 2.0]], "one-dimensional"),
        (["not", "numeric"], "numeric"),
    ],
)
def test_equifun_rejects_invalid_values(values: list[object], match: str) -> None:
    """Test equifun validates sample data."""
    with pytest.raises(ValueError, match=match):
        equifun(values)


@pytest.mark.parametrize(
    ("domain", "match"),
    [
        ([-1.0, 0.0, 1.0], "exactly two"),
        ([-np.inf, 1.0], "finite"),
        ([0.0, np.inf], "finite"),
    ],
)
def test_equifun_rejects_invalid_domains(domain: list[float], match: str) -> None:
    """Test equifun rejects unsupported domains."""
    with pytest.raises(ValueError, match=match):
        equifun([1.0, 2.0], domain)


def test_evaluate() -> None:
    """Test that pickled/unpickled chebfun objects evaluate correctly."""
    f0 = chebfun(np.sin, [-2, 0, 1])
    f1 = pickle.loads(pickle.dumps(f0))  # noqa: S301

    x = -1
    assert f0(x) == f1(x)


def test_version() -> None:
    """Test that the version of the chebpy library is defined."""
    assert chebpy.__version__ is not None


# -------------------------------------------------------------------
# chebpts
# -------------------------------------------------------------------


def test_chebpts_default_domain() -> None:
    """Test chebpts on the default domain [-1, 1]."""
    from chebpy import chebpts

    pts, wts = chebpts(4)
    np.testing.assert_allclose(pts, [-1, -0.5, 0.5, 1], atol=1e-14)
    np.testing.assert_allclose(wts, [-0.5, 1, -1, 0.5], atol=1e-14)


def test_chebpts_custom_domain() -> None:
    """Test chebpts on a custom domain [0, 3]."""
    from chebpy import chebpts

    pts, wts = chebpts(4, [0, 3])
    np.testing.assert_allclose(pts, [0, 0.75, 2.25, 3], atol=1e-14)
    np.testing.assert_allclose(wts, [-0.5, 1, -1, 0.5], atol=1e-14)


def test_chebpts_single_point() -> None:
    """Test chebpts with a single point."""
    from chebpy import chebpts

    pts, wts = chebpts(1)
    np.testing.assert_allclose(pts, [0.0], atol=1e-14)
    np.testing.assert_allclose(wts, [1.0], atol=1e-14)
