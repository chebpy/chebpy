"""Unit-tests for Bndfun complex operations"""

import pytest
import numpy as np

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval


@pytest.fixture
def complex_fixtures():
    """Create fixtures for testing Bndfun complex operations."""
    z = Bndfun.initfun_adaptive(
        lambda x: np.exp(np.pi * 1j * x), Interval(-1, 1)
    )

    return {
        "z": z
    }


def test_init_empty():
    """Test initialization of an empty Bndfun."""
    Bndfun.initempty()


def test_roots(complex_fixtures):
    """Test the roots method on complex Bndfun objects."""
    z = complex_fixtures["z"]
    r0 = z.roots()
    r1 = (z - 1).roots()
    r2 = (z - 1j).roots()
    r3 = (z + 1).roots()
    r4 = (z + 1j).roots()
    assert r0.size == 0
    assert np.allclose(r1, [0])
    assert np.allclose(r2, [0.5])
    assert np.allclose(r3, [-1, 1])
    assert np.allclose(r4, [-0.5])


def test_rho_ellipse_construction(complex_fixtures):
    """Test construction of rho-ellipses with complex Bndfun objects."""
    z = complex_fixtures["z"]
    zz = 1.2 * z
    e = 0.5 * (zz + 1 / zz)
    assert abs(e(1) - e(-1)) <= 1e-14
    assert abs(e(0) + e(-1)) <= 1e-14
    assert abs(e(0) + e(1)) <= 1e-14


def test_calculus(complex_fixtures):
    """Test calculus operations on complex Bndfun objects."""
    z = complex_fixtures["z"]
    assert np.allclose([z.sum()], [0])
    assert (z.cumsum().diff() - z).size == 1
    assert (z - z.cumsum().diff()).size == 1


def test_real_imag(complex_fixtures):
    """Test real and imaginary parts of complex Bndfun objects."""
    z = complex_fixtures["z"]
    # check definition of real and imaginary
    zreal = z.real()
    zimag = z.imag()
    np.testing.assert_equal(zreal.coeffs, np.real(z.coeffs))
    np.testing.assert_equal(zimag.coeffs, np.imag(z.coeffs))
    # check real part of real chebtech is the same chebtech
    assert zreal.real() == zreal
    # check imaginary part of real chebtech is the zero chebtech
    assert zreal.imag().isconst
    assert zreal.imag().coeffs[0] == 0