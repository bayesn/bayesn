import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from bayesn.lm_optim import (
    _jacfwd_lax_map,
    _make_residuals_fn,
    _gn_hessian,
    _lm_minimise,
    run_lm_laplace_gn,
    compute_gn_scale_tril,
)


class TestLMOptim:
    def test_jacfwd_lax_map_vs_jax_jacfwd(self):
        def f(p):
            return jnp.array([
                p[0] ** 2 + p[1],
                jnp.sin(p[1]) * p[2],
                p[0] * p[2] + jnp.exp(p[1]),
                p[2] ** 3,
            ])

        p0 = jnp.array([1.5, -0.5, 2.0])
        j_lax = _jacfwd_lax_map(f, p0)
        j_ref = jax.jacfwd(f)(p0)

        assert j_lax.shape == (4, 3)
        np.testing.assert_allclose(j_lax, j_ref, rtol=1e-5, atol=1e-6)

    def test_make_residuals_and_gn_hessian(self):
        x = jnp.array([1.0, 2.0, 3.0])
        data = jnp.array([2.1, 3.9, 6.2])
        scale = jnp.array([0.1, 0.1, 0.1])
        mask = jnp.array([1.0, 1.0, 1.0])

        def predict_fn(param_dict):
            pred = param_dict["w"] * x + param_dict["b"]
            return pred, scale, data, mask

        z_template = {"w": jnp.array(1.0), "b": jnp.array(0.0)}
        flat0, unflatten = ravel_pytree(z_template)

        residuals_fn = _make_residuals_fn(predict_fn, unflatten)
        r0 = residuals_fn(flat0)

        params0 = unflatten(flat0)
        expected_r = ((data - (params0["w"] * x + params0["b"])) / scale * mask).ravel()
        np.testing.assert_allclose(r0, expected_r, rtol=1e-6)

        def prior_fn(p):
            return 0.5 * jnp.sum(p ** 2)

        H = _gn_hessian(residuals_fn, prior_fn, flat0)
        assert H.shape == (2, 2)
        # GN Hessian should be symmetric positive definite
        np.testing.assert_allclose(H, H.T, atol=1e-6)
        eigvals = jnp.linalg.eigvalsh(H)
        assert jnp.all(eigvals > 0)

    def test_lm_minimise_convergence_and_bounds(self):
        # Fit y = a * x + b
        x = jnp.linspace(0.0, 5.0, 20)
        true_a, true_b = 2.5, 1.2
        y = true_a * x + true_b
        sigma = 0.05

        def residuals_fn(p):
            pred = p[0] * x + p[1]
            return (y - pred) / sigma

        def prior_fn(p):
            return 0.01 * jnp.sum(p ** 2)

        init_p = jnp.array([0.0, 0.0])
        bounds_lo = jnp.array([-10.0, -10.0])
        bounds_hi = jnp.array([10.0, 10.0])

        # Test without linesearch
        p_opt_no_ls, diag_no_ls = _lm_minimise(
            init_p, bounds_lo, bounds_hi, residuals_fn, prior_fn, maxiter=25,
            use_linesearch=False
        )
        np.testing.assert_allclose(p_opt_no_ls, jnp.array([true_a, true_b]), rtol=1e-2, atol=1e-2)
        assert diag_no_ls["f_val"][-1] < diag_no_ls["f_val"][0]

        # Test with linesearch
        p_opt_ls, diag_ls = _lm_minimise(
            init_p, bounds_lo, bounds_hi, residuals_fn, prior_fn, maxiter=25,
            use_linesearch=True
        )
        np.testing.assert_allclose(p_opt_ls, jnp.array([true_a, true_b]), rtol=1e-2, atol=1e-2)
        assert diag_ls["f_val"][-1] < diag_ls["f_val"][0]

    def test_run_lm_laplace_gn_and_scale_tril(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        data = 1.5 * x + 0.5
        scale = jnp.ones_like(x) * 0.1
        mask = jnp.ones_like(x)

        def predict_fn(param_dict):
            pred = param_dict["slope"] * x + param_dict["intercept"]
            return pred, scale, data, mask

        def prior_potential_fn(param_dict):
            return 0.5 * (param_dict["slope"] ** 2 + param_dict["intercept"] ** 2)

        def postprocess_fn_sn(param_dict):
            # E.g. exponentiate or map back
            return {"slope": param_dict["slope"], "intercept": param_dict["intercept"]}

        z_template = {"slope": jnp.array(0.0), "intercept": jnp.array(0.0)}

        median_dict, losses, z_unc_dict = run_lm_laplace_gn(
            predict_fn, prior_potential_fn, postprocess_fn_sn, z_template,
            maxiter=20, use_linesearch=True
        )

        assert "slope" in median_dict and "intercept" in median_dict
        np.testing.assert_allclose(median_dict["slope"], 1.5, atol=0.1)
        np.testing.assert_allclose(median_dict["intercept"], 0.5, atol=0.1)
        assert losses[-1] < losses[0]

        # Scale tril check
        scale_tril = compute_gn_scale_tril(predict_fn, prior_potential_fn, z_template)
        assert scale_tril.shape == (2, 2)
        # Check lower triangular structure
        np.testing.assert_allclose(scale_tril, jnp.tril(scale_tril))
        # Check diagonals are positive
        assert jnp.all(jnp.diag(scale_tril) > 0)
