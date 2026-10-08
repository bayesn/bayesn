"""
Gauss-Newton Levenberg-Marquardt MAP solver used to warm-start BayeSN's VI fits

run_lm_laplace_gn finds the MAP and compute_gn_scale_tril builds the Laplace
covariance Cholesky factor for initialising a variational guide. Both use the
Gauss-Newton approximation H = J^T J + H_prior, with the Jacobian built one column
at a time to keep memory low.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _jacfwd_lax_map(f, p):
    """
    Forward-mode Jacobian built one column at a time with lax.map

    For f: R^d -> R^n, returns J of shape (n, d) matching jax.jacfwd, but runs
    each tangent's forward pass sequentially rather than as a parallel vmap, so
    peak memory stays at a single forward pass. Total compute matches jax.jacfwd.
    """
    d = p.shape[0]

    def col(e):
        _, Jv = jax.jvp(f, (p,), (e,))
        return Jv

    cols = jax.lax.map(col, jnp.eye(d))  # shape (d, n)
    return cols.T  # shape (n, d)


def _make_residuals_fn(predict_fn, unflatten):
    """
    Build the residual function r(p) = ((data - flux) / scale * mask).ravel() used by
    the solver, closing over predict_fn and the flat-vector unflattener
    """
    def residuals_fn(p):
        flux, scale, data, mask = predict_fn(unflatten(p))
        return ((data - flux) / scale * mask).ravel()
    return residuals_fn


def _gn_hessian(residuals_fn, prior_fn, p):
    """
    Gauss-Newton Hessian H = J^T J + H_prior at p, where J is the Jacobian of the
    residuals and H_prior is the Hessian of the prior potential
    """
    J = _jacfwd_lax_map(residuals_fn, p)
    return J.T @ J + jax.hessian(prior_fn)(p)


def _lm_minimise(init_p, bounds_lo, bounds_hi, residuals_fn, prior_fn, maxiter,
                 lam_init=1e-3, use_linesearch=False):
    """
    Gauss-Newton Levenberg-Marquardt minimisation of 0.5 * sum(residuals^2) + prior

    Damped-Newton loop with gain-ratio acceptance and lambda control. Each step forms
    the Gauss-Newton Hessian H = J^T J + H_prior and solves (H + lam * diag(H)) d = -g,
    which avoids the full model Hessian and stays positive-definite.

    Parameters
    ----------
    init_p: array-like
        1D initial parameter vector in unconstrained space
    bounds_lo, bounds_hi: array-like
        Parameter bounds, same shape as init_p; use -inf/+inf in unconstrained space
    residuals_fn: callable
        Residual vector r(p); the data term of the loss is 0.5 * sum(r^2)
    prior_fn: callable
        Prior potential p -> scalar, -log p(z) including bijector log-det
    maxiter: int
        Number of LM steps, the length of the lax.scan
    lam_init: float
        Initial LM damping parameter
    use_linesearch: Boolean
        If True, try multiple step sizes along the Newton direction

    Returns
    -------

    p_final: array-like
        Unconstrained minimiser
    diagnostics: dict
        Per-iteration f_val, lam, accept, rho and grad_norm
    """
    ls_alphas = jnp.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    def loss_fn(p):
        r = residuals_fn(p)
        return 0.5 * jnp.sum(r * r) + prior_fn(p)

    grad_fn = jax.grad(loss_fn)

    def newton_step(carry, _):
        p, lam, f_val = carry
        g = grad_fn(p)
        H = _gn_hessian(residuals_fn, prior_fn, p)

        diag_H = jnp.maximum(jnp.abs(jnp.diag(H)), 1e-6)
        d = -jnp.linalg.solve(H + lam * jnp.diag(diag_H), g)
        is_bad = (jnp.dot(g, d) >= -1e-12) | jnp.any(jnp.isnan(d))
        scale_guess = 1.0 / (jnp.mean(jnp.abs(jnp.diag(H))) + 1e-8)
        d = jnp.where(is_bad, -g * scale_guess, d)

        if use_linesearch:
            def eval_alpha(alpha):
                tp = jnp.clip(p + alpha * d, bounds_lo, bounds_hi)
                return loss_fn(tp)
            f_trials = jax.vmap(eval_alpha)(ls_alphas)
            best_idx = jnp.argmin(f_trials)
            best_alpha = ls_alphas[best_idx]
            trial_p = jnp.clip(p + best_alpha * d, bounds_lo, bounds_hi)
            f_trial = f_trials[best_idx]
        else:
            trial_p = jnp.clip(p + d, bounds_lo, bounds_hi)
            f_trial = loss_fn(trial_p)

        s = trial_p - p

        actual = f_val - f_trial
        predicted = -(jnp.dot(g, s) + 0.5 * s @ H @ s)
        predicted = jnp.maximum(predicted, 1e-30)
        rho = actual / predicted

        accept = rho > 0.01
        p_out = jnp.where(accept, trial_p, p)
        f_out = jnp.where(accept, f_trial, f_val)

        lam_out = jnp.where(
            rho > 0.75, lam * 0.3,
            jnp.where(rho > 0.25, lam, lam * 3.0),
        )
        lam_out = jnp.clip(lam_out, 1e-8, 1e8)

        diagnostics = {'f_val': f_out, 'lam': lam_out, 'accept': accept,
                       'rho': rho, 'grad_norm': jnp.max(jnp.abs(g))}
        return (p_out, lam_out, f_out), diagnostics

    f_init = loss_fn(init_p)
    (p_final, _, _), diagnostics = jax.lax.scan(
        newton_step, (init_p, lam_init, f_init), None, length=maxiter
    )
    return p_final, diagnostics


def run_lm_laplace_gn(predict_fn, prior_potential_fn, postprocess_fn_sn, z_template,
                     maxiter=30, lam_init=1e-3, use_linesearch=True):
    """
    Find the MAP using Gauss-Newton Levenberg-Marquardt

    Parameters
    ----------
    predict_fn: callable
        Maps a constrained parameter dict to (flux, scale, data, mask) by running
        the model, giving the predicted photometry, per-observation noise scale,
        observed data and validity mask
    prior_potential_fn: callable
        Maps an unconstrained parameter dict to the negative log prior in
        unconstrained space, including bijector log-det
    postprocess_fn_sn: callable
        Maps the unconstrained MAP back to constrained space via the model bijectors
    z_template: dict
        Unconstrained initial values keyed by sample-site name; shapes and dtypes
        define the flat vector layout
    maxiter: int
        LM iteration budget
    lam_init: float
        Initial LM damping
    use_linesearch: Boolean
        If True, perform a line search along the Newton direction at every step

    Returns
    -------

    median_dict: dict
        Constrained MAP, same keys/shapes as AutoLaplaceApproximation.median
    losses: array-like
        Loss at each iteration, shape (maxiter,)
    z_unc_dict: dict
        Unconstrained MAP in the same layout as z_template
    """
    flat0, unflatten = ravel_pytree(z_template)
    bounds_hi = jnp.full_like(flat0, jnp.inf)
    bounds_lo = -bounds_hi

    residuals_fn = _make_residuals_fn(predict_fn, unflatten)

    def prior_fn(p):
        return prior_potential_fn(unflatten(p))

    p_final, diag = _lm_minimise(
        flat0, bounds_lo, bounds_hi, residuals_fn, prior_fn, maxiter,
        lam_init=lam_init, use_linesearch=use_linesearch,
    )
    z_unc_dict = unflatten(p_final)
    median_dict = postprocess_fn_sn(z_unc_dict)
    return median_dict, diag['f_val'], z_unc_dict


def compute_gn_scale_tril(predict_fn, prior_potential_fn, z_template):
    """
    Cholesky factor of inv(J^T J + H_prior) at z_template

    Gauss-Newton estimate of the Laplace posterior covariance in unconstrained
    latent space (returned as its lower-Cholesky factor), avoiding the full model
    Hessian.
    """
    flat0, unflatten = ravel_pytree(z_template)

    residuals_fn = _make_residuals_fn(predict_fn, unflatten)

    def prior_fn(p):
        return prior_potential_fn(unflatten(p))

    H = _gn_hessian(residuals_fn, prior_fn, flat0)
    cov = jnp.linalg.inv(H)
    return jnp.linalg.cholesky(cov)
