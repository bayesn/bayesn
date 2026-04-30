import numpyro
from numpyro.infer.autoguide import AutoContinuous
from numpyro.distributions import constraints
from numpyro.infer.initialization import init_to_median
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.continuous import MultivariateNormal
from numpyro.distributions.truncated import TruncatedNormal
from numpyro.distributions.constraints import _SingletonConstraint
from numpyro.distributions.transforms import biject_to, IdentityTransform, LowerCholeskyAffine
import jax.numpy as jnp
from numpyro.distributions.util import is_prng_key, promote_shapes, validate_sample
from jax import lax
from jax.random import normal, exponential


class _FirstPositive(_SingletonConstraint):
    event_dim = 1

    def __call__(self, x):
        return (x[0] >= 0)

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


firstpositive = _FirstPositive()


@biject_to.register(firstpositive)
def _transform_to_real(constraint):
    return IdentityTransform()


class MultiZLTN(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    reparametrized_params = [
        "loc",
        "covariance_matrix",
        "precision_matrix",
        "scale_tril",
    ]

    @constraints.dependent_property
    def support(self):
        return firstpositive

    def __init__(
            self,
            loc=0.0,
            covariance_matrix=None,
            precision_matrix=None,
            scale_tril=None,
            validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        loc = loc[..., jnp.newaxis]
        if covariance_matrix is not None:
            loc, self.covariance_matrix = promote_shapes(loc, covariance_matrix)
            self.scale_tril = jnp.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            loc, self.precision_matrix = promote_shapes(loc, precision_matrix)
            self.scale_tril = cholesky_of_inverse(self.precision_matrix)
        elif scale_tril is not None:
            loc, self.scale_tril = promote_shapes(loc, scale_tril)
        else:
            raise ValueError(
                "One of `covariance_matrix`, `precision_matrix`, `scale_tril`"
                " must be specified."
            )
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(self.scale_tril)[:-2]
        )
        event_shape = jnp.shape(self.scale_tril)[-1:]
        self.mu = loc
        self.mu_1 = self.mu[0]
        self.mu_2 = jnp.squeeze(self.mu[1:])

        L = self.scale_tril
        self.sigma_11 = L[0, 0] * L[0, 0]
        self.sigma_21 = L[1:, 0:1] * L[0, 0]
        self.sigma_12 = self.sigma_21.T
        self.sigma_22 = jnp.matmul(L[1:, :], L[1:, :].T)

        self._sigma_11_inv = 1.0 / self.sigma_11
        self._sigma_hat = (
            self.sigma_22
            - jnp.matmul(self.sigma_21 * self._sigma_11_inv, self.sigma_12)
            + 1.0e-9 * jnp.eye(event_shape[0] - 1)
        )
        self._chol_sigma_hat = jnp.linalg.cholesky(self._sigma_hat)

        self.zltn_dist = TruncatedNormal(self.mu_1, self.scale_tril[0][0], low=0.)
        super(MultiZLTN, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        first_samples = self.zltn_dist.sample(key, sample_shape)
        if sample_shape == ():
            mu_2_b = self.mu_2
            concat_axis = 0
        else:
            mu_2_b = self.mu_2[:, None]
            concat_axis = 1
        mu_hat = mu_2_b + jnp.matmul(
            self.sigma_21,
            (self._sigma_11_inv * (first_samples - self.mu_1)).T,
        )
        eps = normal(
            key, shape=sample_shape + self.batch_shape + (self.event_shape[0] - 1,)
        )
        second_samples = mu_hat.T + jnp.squeeze(
            jnp.matmul(self._chol_sigma_hat, eps[..., jnp.newaxis]), axis=-1
        )
        return jnp.concatenate((first_samples, second_samples), axis=concat_axis)

    def log_prob(self, value):
        log_prob_x = self.zltn_dist.log_prob(value[..., 0])
        mu_hat = self.mu_2[..., None] + jnp.matmul(
            self.sigma_21,
            self._sigma_11_inv * (value[..., 0] - self.mu_1[None, ...]),
        )
        log_prob_y_given_x = MultivariateNormal(
            mu_hat.T, scale_tril=self._chol_sigma_hat
        ).log_prob(value[..., 1:])
        return log_prob_x + log_prob_y_given_x

    # old method without vectorization
    # def log_prob(self, value):
    #     log_prob_x = self.zltn_dist.log_prob(value[0])
    #     sigma_11_inverse = 1. / self.sigma_11
    #     # mu_hat = self.mu_2 + self.sigma_21*sigma_11_inverse*(value[0]- self.mu_1)
    #     mu_hat = self.mu_2 + jnp.matmul(self.sigma_21, sigma_11_inverse * (value[0] - self.mu_1))
    #     sigma_hat = self.sigma_22 - jnp.matmul((self.sigma_21 * sigma_11_inverse), self.sigma_12)
    #     log_prob_y_given_x = MultivariateNormal(mu_hat, sigma_hat).log_prob(value[1:])
    #     print(log_prob_x + log_prob_y_given_x)
    #     return log_prob_x + log_prob_y_given_x


class AutoMultiZLTNGuide(AutoContinuous):
    # scale_tril_constraint = constraints.scaled_unit_lower_cholesky
    scale_tril_constraint = constraints.lower_cholesky

    def __init__(
            self,
            model,
            *,
            prefix="auto",
            init_loc_fn=init_to_median,
            init_scale=0.1,
            init_scale_tril=None,
    ):
        if init_scale <= 0:
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale
        self._init_scale_tril = init_scale_tril
        super().__init__(model, prefix=prefix, init_loc_fn=init_loc_fn)

    def _get_posterior(self):
        loc = numpyro.param("{}_loc".format(self.prefix), self._init_latent)
        if self._init_scale_tril is None:
            init_scale_tril = jnp.identity(self.latent_dim) * self._init_scale
        else:
            init_scale_tril = self._init_scale_tril
        scale_tril = numpyro.param(
            "{}_scale_tril".format(self.prefix),
            init_scale_tril,
            constraint=self.scale_tril_constraint,
        )
        return MultiZLTN(loc, scale_tril=scale_tril)

    def get_base_dist(self):
        return dist.Normal(jnp.zeros(self.latent_dim), 1).to_event(1)

    def get_transform(self, params):
        loc = params["{}_loc".format(self.prefix)]
        scale_tril = params["{}_scale_tril".format(self.prefix)]
        return LowerCholeskyAffine(loc, scale_tril)

    def get_posterior(self, params):
        """
        Returns a Multi ZLTN posterior distribution.
        """
        transform = self.get_transform(params)
        return MultiZLTN(transform.loc, scale_tril=transform.scale_tril)


class My_Exponential(Distribution):
    reparametrized_params = ["rate"]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.real

    def __init__(self, rate=1.0, *, validate_args=None):
        self.rate = rate
        super(My_Exponential, self).__init__(
            batch_shape=jnp.shape(rate), validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        return (
                exponential(key, shape=sample_shape + self.batch_shape) / self.rate
        )

    @validate_sample
    def log_prob(self, value):
        # One-sided quadratic barrier: zero (and zero gradient) on the support.
        base = jnp.log(self.rate) - self.rate * value
        neg = jnp.where(value < 0, value, 0.0)
        return base - 1.0e2 * neg * neg

    @property
    def mean(self):
        return jnp.reciprocal(self.rate)

    @property
    def variance(self):
        return jnp.reciprocal(self.rate ** 2)

    def cdf(self, value):
        return -jnp.expm1(-self.rate * value)

    def icdf(self, q):
        return -jnp.log1p(-q) / self.rate
