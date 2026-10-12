import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import IdentityTransform
from numpyro.handlers import seed, trace

from bayesn.zltn_utils import (
    _FirstPositive,
    firstpositive,
    _transform_to_real,
    MultiZLTN,
    AutoMultiZLTNGuide,
    My_Exponential,
)


class TestFirstPositiveConstraint:
    def test_constraint_and_bijector(self):
        c = _FirstPositive()
        assert c(jnp.array([1.0, -5.0]))
        assert not c(jnp.array([-0.1, 5.0]))

        prototype = jnp.ones((3,))
        feasible = c.feasible_like(prototype)
        np.testing.assert_allclose(feasible, jnp.zeros((3,)))

        transform = _transform_to_real(firstpositive)
        assert isinstance(transform, IdentityTransform)


class TestMultiZLTN:
    def test_init_with_scale_tril(self):
        loc = jnp.array([1.0, 2.0, 3.0])
        scale_tril = jnp.array([
            [1.5, 0.0, 0.0],
            [0.3, 1.2, 0.0],
            [0.1, 0.2, 1.0],
        ])

        d = MultiZLTN(loc=loc, scale_tril=scale_tril)
        assert d.support == firstpositive

        # Sample with empty shape
        key = jr.PRNGKey(10)
        s0 = d.sample(key, sample_shape=())
        assert s0.shape == (3,)
        assert s0[0] >= 0.0

        # Sample with batch shape
        s_batch = d.sample(key, sample_shape=(100,))
        assert s_batch.shape == (100, 3)
        assert jnp.all(s_batch[:, 0] >= 0.0)

        # Log prob
        val = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        lp = d.log_prob(val)
        assert lp.shape == (2,)
        assert jnp.all(jnp.isfinite(lp))

    def test_init_with_covariance(self):
        loc = jnp.array([0.5, 1.5, -0.5])
        cov = jnp.array([
            [2.0, 0.4, 0.1],
            [0.4, 1.5, 0.2],
            [0.1, 0.2, 1.0],
        ])

        d = MultiZLTN(loc=loc, covariance_matrix=cov)
        expected_scale_tril = jnp.linalg.cholesky(cov)
        np.testing.assert_allclose(d.scale_tril, expected_scale_tril, rtol=1e-5)

    def test_init_missing_covariance(self):
        with pytest.raises(ValueError, match="One of `covariance_matrix`"):
            MultiZLTN(loc=jnp.array([1.0, 2.0]))


class TestAutoMultiZLTNGuide:
    def test_invalid_init_scale(self):
        def model():
            numpyro.sample("x", dist.Normal(0, 1))

        with pytest.raises(ValueError, match="Expected init_scale > 0"):
            AutoMultiZLTNGuide(model, init_scale=-0.1)

    def test_guide_execution_and_posterior(self):
        def model():
            numpyro.sample("x", dist.Normal(jnp.zeros(3), 1).to_event(1))

        guide = AutoMultiZLTNGuide(model, init_scale=0.2)
        # Trace guide execution to exercise _get_posterior
        tr = trace(seed(guide, jr.PRNGKey(42))).get_trace()
        assert "x" in tr

        params = {
            "auto_loc": jnp.array([0.5, 1.0, -0.2]),
            "auto_scale_tril": jnp.array([
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.1, 0.1, 0.9],
            ]),
        }

        transform = guide.get_transform(params)
        np.testing.assert_allclose(transform.loc, params["auto_loc"])
        np.testing.assert_allclose(transform.scale_tril, params["auto_scale_tril"])

        post = guide.get_posterior(params)
        assert isinstance(post, MultiZLTN)
        np.testing.assert_allclose(post.scale_tril, params["auto_scale_tril"])

    def test_guide_with_custom_scale_tril(self):
        def model():
            numpyro.sample("x", dist.Normal(jnp.zeros(3), 1).to_event(1))

        custom_scale = jnp.eye(3) * 0.5
        guide = AutoMultiZLTNGuide(model, init_scale_tril=custom_scale)
        tr = trace(seed(guide, jr.PRNGKey(123))).get_trace()
        assert "x" in tr


class TestMyExponential:
    def test_distribution(self):
        rate = 2.5
        d = My_Exponential(rate=rate)

        np.testing.assert_allclose(float(d.mean), 1.0 / rate)
        np.testing.assert_allclose(float(d.variance), 1.0 / (rate ** 2))

        key = jr.PRNGKey(77)
        samples = d.sample(key, sample_shape=(100,))
        assert samples.shape == (100,)
        assert jnp.all(samples >= 0.0)

        # CDF and ICDF
        x = jnp.array([0.1, 0.5, 1.0])
        cdf_vals = d.cdf(x)
        x_rec = d.icdf(cdf_vals)
        np.testing.assert_allclose(x, x_rec, rtol=1e-5)

        # Log prob check
        lp_pos = d.log_prob(jnp.array([0.5]))
        assert jnp.isfinite(lp_pos)

        # Barrier on negative support
        lp_neg = d.log_prob(jnp.array([-1.0]))
        assert lp_neg < lp_pos
