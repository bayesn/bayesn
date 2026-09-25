import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import norm

from bayesn.pdf_utils import ChebyshevICDF, MonoEICDF


class TestChebyshevICDF:
    def test_from_quantiles(self):
        probs = np.linspace(0.01, 0.99, 15)
        vals = norm.ppf(probs)
        icdf_obj = ChebyshevICDF(probs=probs, vals=vals, method="quantiles")

        assert icdf_obj.K == 14
        assert len(icdf_obj.c) == 15

        # Check median ~ 0
        p_eval = jnp.array([0.5])
        median_est = icdf_obj.icdf(p_eval)
        np.testing.assert_allclose(median_est, 0.0, atol=0.05)

        # Direct clenshaw check
        x_eval = jnp.array([0.0])
        val_clenshaw = icdf_obj.clenshaw(x_eval)
        np.testing.assert_allclose(val_clenshaw, median_est, atol=1e-6)

        # Sampling
        key = jr.PRNGKey(42)
        samples = icdf_obj.sample(key, sample_shape=(2000,))
        assert samples.shape == (2000,)
        np.testing.assert_allclose(float(jnp.mean(samples)), 0.0, atol=0.1)

    def test_from_samples_with_k(self):
        rng = np.random.default_rng(123)
        data = rng.normal(loc=5.0, scale=2.0, size=5000)
        icdf_obj = ChebyshevICDF(samples=data, K=12, method="samples")

        assert icdf_obj.K == 12
        p_eval = jnp.array([0.5])
        np.testing.assert_allclose(icdf_obj.icdf(p_eval), 5.0, atol=0.15)

    def test_from_samples_with_probs(self):
        rng = np.random.default_rng(123)
        data = rng.normal(loc=0.0, scale=1.0, size=5000)
        probs = np.linspace(0.05, 0.95, 11)
        icdf_obj = ChebyshevICDF(samples=data, probs=probs, method="samples")

        assert icdf_obj.K == 10
        np.testing.assert_allclose(icdf_obj.icdf(jnp.array([0.5])), 0.0, atol=0.1)

    def test_from_samples_with_matching_probs_and_k(self):
        rng = np.random.default_rng(123)
        data = rng.normal(loc=0.0, scale=1.0, size=2000)
        probs = np.linspace(0.05, 0.95, 9)
        icdf_obj = ChebyshevICDF(samples=data, probs=probs, K=8, method="samples")
        assert icdf_obj.K == 8

    def test_errors(self):
        with pytest.raises(ValueError, match="All probs must be in"):
            ChebyshevICDF(probs=np.array([-0.1, 0.5]), vals=np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match="All probs must be in"):
            ChebyshevICDF(probs=np.array([0.1, 1.5]), vals=np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match='probs and vals must be provided'):
            ChebyshevICDF(method="quantiles")

        with pytest.raises(ValueError, match='samples must be provided'):
            ChebyshevICDF(method="samples")

        with pytest.raises(ValueError, match='one of probs or K must be provided'):
            ChebyshevICDF(samples=np.array([1.0, 2.0]), method="samples")

        with pytest.raises(ValueError, match='Both probs and K provided in an ambiguous way'):
            ChebyshevICDF(
                samples=np.array([1.0, 2.0]),
                probs=np.linspace(0.1, 0.9, 5),
                K=10,
                method="samples",
            )

        with pytest.raises(ValueError, match='not recognised'):
            ChebyshevICDF(method="invalid_method")


class TestMonoEICDF:
    def test_linear(self):
        samples = np.linspace(0.0, 10.0, 50)
        mono_obj = MonoEICDF(samples=samples, kind="linear")

        p_eval = jnp.array([0.5])
        np.testing.assert_allclose(mono_obj.icdf(p_eval), 5.0, atol=0.3)

        key = jr.PRNGKey(99)
        draws = mono_obj.sample(key, sample_shape=(100,))
        assert draws.shape == (100,)
        assert jnp.all(draws >= 0.0)
        assert jnp.all(draws <= 10.0)

    def test_cubic_error_without_interpax(self):
        samples = np.linspace(0.0, 10.0, 50)
        with pytest.raises(ModuleNotFoundError, match="interpax"):
            MonoEICDF(samples=samples, kind="cubic")

    def test_invalid_kind(self):
        samples = np.linspace(0.0, 10.0, 50)
        with pytest.raises(ValueError, match='kind must be either "linear or "cubic"!'):
            MonoEICDF(samples=samples, kind="quadratic")
