"""Unit-tests for chebpy/gpr.py — Gaussian process regression."""

from __future__ import annotations

import numpy as np
import pytest

from chebpy.chebfun import Chebfun
from chebpy.gpr import gpr
from chebpy.quasimatrix import Quasimatrix


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _assert_chebfun(obj: object) -> None:
    """Assert that *obj* is a Chebfun."""
    assert isinstance(obj, Chebfun)


# ---------------------------------------------------------------------------
#  Basic return types
# ---------------------------------------------------------------------------
class TestGPRReturnTypes:
    """Verify the types and shapes of objects returned by gpr."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        rng = np.random.default_rng(42)
        self.x = np.sort(-2 + 4 * rng.random(10))
        self.y = np.sin(np.exp(self.x))

    def test_returns_two_chebfuns(self) -> None:
        result = gpr(self.x, self.y, domain=[-2, 2])
        assert len(result) == 2
        _assert_chebfun(result[0])
        _assert_chebfun(result[1])

    def test_returns_samples_quasimatrix(self) -> None:
        result = gpr(self.x, self.y, domain=[-2, 2], n_samples=3)
        assert len(result) == 3
        _assert_chebfun(result[0])
        _assert_chebfun(result[1])
        assert isinstance(result[2], Quasimatrix)
        assert result[2].shape[1] == 3


# ---------------------------------------------------------------------------
#  Interpolation at the data points
# ---------------------------------------------------------------------------
class TestGPRInterpolation:
    """The noiseless posterior mean must interpolate the data."""

    def test_interpolates_data(self) -> None:
        rng = np.random.default_rng(7)
        x = np.sort(-1 + 2 * rng.random(8))
        y = np.cos(x)
        f_mean, _ = gpr(x, y, domain=[-1, 1])
        np.testing.assert_allclose(f_mean(x), y, atol=1e-6)

    def test_interpolates_data_with_sigma(self) -> None:
        rng = np.random.default_rng(11)
        x = np.sort(-1 + 2 * rng.random(6))
        y = np.exp(-x)
        f_mean, _ = gpr(x, y, domain=[-1, 1], sigma=2.0)
        np.testing.assert_allclose(f_mean(x), y, atol=1e-6)

    def test_interpolates_data_with_length_scale(self) -> None:
        rng = np.random.default_rng(11)
        x = np.sort(-1 + 2 * rng.random(8))
        y = np.sin(x)
        f_mean, _ = gpr(x, y, domain=[-1, 1], length_scale=0.5)
        np.testing.assert_allclose(f_mean(x), y, atol=1e-6)


# ---------------------------------------------------------------------------
#  Variance properties
# ---------------------------------------------------------------------------
class TestGPRVariance:
    """Posterior variance must be non-negative and small near data."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        rng = np.random.default_rng(0)
        self.x = np.sort(-2 + 4 * rng.random(10))
        self.y = np.sin(self.x)
        self.f_mean, self.f_var = gpr(self.x, self.y, domain=[-2, 2])

    def test_variance_nonnegative(self) -> None:
        # Evaluate on a fine grid
        t = np.linspace(-2, 2, 500)
        assert np.all(self.f_var(t) >= -1e-10)

    def test_variance_small_at_data(self) -> None:
        np.testing.assert_allclose(self.f_var(self.x), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
#  Noisy observations
# ---------------------------------------------------------------------------
class TestGPRNoisy:
    """Behaviour with noise > 0."""

    def test_noisy_runs(self) -> None:
        rng = np.random.default_rng(3)
        x = np.sort(-2 + 4 * rng.random(10))
        y = np.sin(x) + 0.1 * rng.standard_normal(10)
        f_mean, f_var = gpr(x, y, domain=[-2, 2], noise=0.1)
        _assert_chebfun(f_mean)
        _assert_chebfun(f_var)

    def test_noisy_does_not_interpolate(self) -> None:
        rng = np.random.default_rng(3)
        x = np.sort(-2 + 4 * rng.random(10))
        y = np.sin(x) + 0.1 * rng.standard_normal(10)
        f_mean, _ = gpr(x, y, domain=[-2, 2], noise=0.1)
        # With noise the posterior mean will NOT pass exactly through the data
        residuals = np.abs(f_mean(x) - y)
        assert np.max(residuals) > 1e-4


# ---------------------------------------------------------------------------
#  Periodic kernel
# ---------------------------------------------------------------------------
class TestGPRTrig:
    """Tests for the periodic (trig) kernel variant."""

    def test_trig_runs(self) -> None:
        x = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        y = np.sin(x)
        f_mean, f_var = gpr(x, y, domain=[0, 2 * np.pi], trig=True)
        _assert_chebfun(f_mean)
        _assert_chebfun(f_var)

    def test_trig_interpolates(self) -> None:
        x = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        y = np.sin(x)
        f_mean, _ = gpr(x, y, domain=[0, 2 * np.pi], trig=True, length_scale=1.0)
        np.testing.assert_allclose(f_mean(x), y, atol=1e-4)


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------
class TestGPREdgeCases:
    """Edge case handling."""

    def test_empty_input(self) -> None:
        f_mean, f_var = gpr([], [], domain=[-1, 1])
        # Prior: mean=0, variance=sigma^2 const.
        _assert_chebfun(f_mean)
        _assert_chebfun(f_var)
        np.testing.assert_allclose(f_mean(0.0), 0.0, atol=1e-12)

    def test_empty_input_default_domain(self) -> None:
        f_mean, _f_var = gpr([], [])
        _assert_chebfun(f_mean)
        support = f_mean.support
        np.testing.assert_allclose(support[0], -1.0, atol=1e-12)
        np.testing.assert_allclose(support[-1], 1.0, atol=1e-12)

    def test_single_point(self) -> None:
        f_mean, _f_var = gpr([0.0], [1.0])
        _assert_chebfun(f_mean)
        np.testing.assert_allclose(f_mean(0.0), 1.0, atol=1e-6)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            gpr([1, 2, 3], [1, 2])

    def test_default_domain_from_data(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        f_mean, _ = gpr(x, y)
        # domain should be [0, 2]
        support = f_mean.support
        np.testing.assert_allclose(support[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(support[-1], 2.0, atol=1e-12)

    def test_trig_default_domain(self) -> None:
        x = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        y = np.sin(x)
        f_mean, _ = gpr(x, y, trig=True)
        _assert_chebfun(f_mean)
        # domain should span data plus 10% padding
        support = f_mean.support
        assert support[0] <= 0.0
        assert support[-1] > x[-1]


# ---------------------------------------------------------------------------
#  Samples from the posterior
# ---------------------------------------------------------------------------
class TestGPRSamples:
    """Posterior samples should be Chebfun objects on the correct domain."""

    def test_samples_correct_count(self) -> None:
        rng = np.random.default_rng(1)
        x = np.sort(-1 + 2 * rng.random(6))
        y = np.cos(x)
        _, _, samples = gpr(x, y, domain=[-1, 1], n_samples=5)
        assert samples.shape[1] == 5

    def test_prior_samples(self) -> None:
        _, _, samples = gpr([], [], domain=[-1, 1], sigma=1.0, n_samples=3)
        assert isinstance(samples, Quasimatrix)
        assert samples.shape[1] == 3

    def test_prior_samples_trig(self) -> None:
        _, _, samples = gpr([], [], domain=[0, 2 * np.pi], sigma=1.0, trig=True, n_samples=3)
        assert isinstance(samples, Quasimatrix)
        assert samples.shape[1] == 3


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------
class TestGPRInternal:
    """Cover internal helper branches."""

    def test_log_marginal_likelihood_scalar(self) -> None:
        from chebpy.gpr import _GPROptions, _log_marginal_likelihood

        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        opts = _GPROptions(sigma=1.0, sigma_given=True, domain=np.array([-1.0, 1.0]))
        result = _log_marginal_likelihood(0.5, x, y, opts)
        assert isinstance(result, float)
