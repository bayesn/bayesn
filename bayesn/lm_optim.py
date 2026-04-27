"""Levenberg-Marquardt MAP solver used as an alternative to the Adam-based
Laplace stage of BayeSN's VI fit.

The core ``_newton_lm_round`` routine is copied verbatim from
``jax_SNANA_claude/jax_snlc_fit/fitter.py`` (lines 225-332) to avoid pulling
``jaxopt`` as a dependency.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _newton_lm_round(init_p, bounds_lo, bounds_hi, loss_closed, maxiter,
                     lam_init=1e-3, use_linesearch=False, debug=False):
    """Pure LM with gain-ratio acceptance and lambda control.

    Args:
        init_p: 1-D initial parameter vector (unconstrained).
        bounds_lo, bounds_hi: same shape as init_p; pass ``-jnp.inf``/``+jnp.inf``
            when optimising in unconstrained space.
        loss_closed: callable ``p -> scalar``, the objective to minimise.
        maxiter: number of LM steps (lax.scan length).
        lam_init: initial LM damping parameter.
        use_linesearch: if True, try multiple step sizes along the Newton direction.
        debug: if True, per-iteration diagnostics are returned as a dict.

    Returns:
        (p_final, f_final, grad_norm, diagnostics)
    """

    ls_alphas = jnp.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    def newton_step(carry, _):
        p, lam, f_val = carry
        g = jax.grad(loss_closed)(p)
        H = jax.hessian(loss_closed)(p)

        diag_H = jnp.maximum(jnp.abs(jnp.diag(H)), 1e-6)
        H_reg = H + lam * jnp.diag(diag_H)

        d = -jnp.linalg.solve(H_reg, g)
        slope = jnp.dot(g, d)

        is_bad = (slope >= -1e-12) | jnp.any(jnp.isnan(d))
        abs_diag = jnp.abs(jnp.diag(H))
        scale_guess = 1.0 / (jnp.mean(abs_diag) + 1e-8)
        d = jnp.where(is_bad, -g * scale_guess, d)

        if use_linesearch:
            def eval_alpha(alpha):
                tp = jnp.clip(p + alpha * d, bounds_lo, bounds_hi)
                return loss_closed(tp)
            f_trials = jax.vmap(eval_alpha)(ls_alphas)
            best_idx = jnp.argmin(f_trials)
            best_alpha = ls_alphas[best_idx]
            trial_p = jnp.clip(p + best_alpha * d, bounds_lo, bounds_hi)
            f_trial = f_trials[best_idx]
        else:
            trial_p = jnp.clip(p + d, bounds_lo, bounds_hi)
            f_trial = loss_closed(trial_p)

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

        if debug:
            diagnostics = {
                'f_val': f_out,
                'lam': lam_out,
                'accept': accept,
                'rho': rho,
                'grad_norm': jnp.max(jnp.abs(g)),
            }
        else:
            diagnostics = None
        return (p_out, lam_out, f_out), diagnostics

    f_init = loss_closed(init_p)
    (p_final, _, f_final), diagnostics = jax.lax.scan(
        newton_step, (init_p, lam_init, f_init), None, length=maxiter
    )
    g_final = jax.grad(loss_closed)(p_final)
    grad_norm = jnp.max(jnp.abs(g_final))
    return p_final, f_final, grad_norm, diagnostics


def run_lm_laplace(potential_fn_sn, postprocess_fn_sn, z_template,
                   maxiter=30, lam_init=1e-3, use_linesearch=True):
    """Find the MAP of ``potential_fn_sn`` using LM and return the constrained
    median dict plus per-iteration loss history.

    Args:
        potential_fn_sn: callable ``params_unc_dict -> scalar`` giving
            ``-log p(data, z)`` for one SN (typically built from
            ``numpyro.infer.util.initialize_model(dynamic_args=True)``).
        postprocess_fn_sn: callable ``params_unc_dict -> constrained_dict``
            that applies the bijectors back to constrained parameter space.
        z_template: dict of unconstrained init values (matching the model's
            sample-site names). Shape/dtype define the flat vector layout.
        maxiter: LM iteration budget.
        lam_init: initial LM damping.
        use_linesearch: if True, perform a 7-point backtracking line search
            along the Newton direction at every step.

    Returns:
        (median_dict, losses_1d, z_unc_dict)
            ``median_dict`` has the same keys/shapes as
            ``AutoLaplaceApproximation.median(params)`` (constrained space);
            ``losses_1d`` is shape ``(maxiter,)``; ``z_unc_dict`` is the
            unconstrained MAP in the same dict layout as ``z_template``.
    """
    flat0, unflatten = ravel_pytree(z_template)
    bounds_hi = jnp.full_like(flat0, jnp.inf)
    bounds_lo = -bounds_hi

    def loss_closed(p):
        return potential_fn_sn(unflatten(p))

    p_final, _, _, diag = _newton_lm_round(
        flat0, bounds_lo, bounds_hi, loss_closed,
        maxiter=maxiter, lam_init=lam_init,
        use_linesearch=use_linesearch, debug=True,
    )
    z_unc_dict = unflatten(p_final)
    median_dict = postprocess_fn_sn(z_unc_dict)
    return median_dict, diag['f_val'], z_unc_dict


def compute_laplace_scale_tril(potential_fn_sn, z_template):
    """Cholesky of the inverse Hessian of ``potential_fn_sn`` at ``z_template``.

    Returns the lower-Cholesky factor of the Laplace posterior covariance in
    unconstrained latent space, suitable for initialising a variational
    guide's ``scale_tril`` parameter.
    """
    flat0, unflatten = ravel_pytree(z_template)
    H = jax.hessian(lambda p: potential_fn_sn(unflatten(p)))(flat0)
    cov = jnp.linalg.inv(H)
    return jnp.linalg.cholesky(cov)
