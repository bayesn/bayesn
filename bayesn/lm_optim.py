"""Levenberg-Marquardt MAP solvers used in BayeSN's VI fit's Stage 2.

Three flavours are provided:

- ``run_lm_laplace`` / ``compute_laplace_scale_tril``: full-Hessian LM via
  ``jax.hessian``. Used by Stage 1 (small noeps latent — fits in GPU memory).
- ``run_lm_laplace_gn`` / ``compute_gn_scale_tril``: Gauss-Newton LM with the
  Jacobian built serially via ``jax.lax.map`` (option 2). Avoids the autodiff
  vmap multiplier on the per-SN forward intermediate that OOMs on GPU at
  realistic batch sizes. ~half the compute of full-Hessian LM.
- ``run_lm_laplace_hvp_cg`` / ``compute_hvp_scale_tril``: exact-Hessian LM
  via Hessian-vector products and conjugate-gradient linsolve (option 3).
  Same memory profile as the GN variant; uses the exact Hessian instead of
  GN; about 2x compute of GN per LM step.

The core ``_newton_lm_round`` routine is copied verbatim from
``jax_SNANA_claude/jax_snlc_fit/fitter.py`` (lines 225-332) to avoid pulling
``jaxopt`` as a dependency. The GN and HVP-CG variants extend the same
gain-ratio-with-line-search pattern.
"""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from jax.flatten_util import ravel_pytree


def _jacfwd_lax_map(f, p):
    """Memory-efficient forward-mode Jacobian via serial ``lax.map`` over tangents.

    For ``f: R^d -> R^n``, returns ``J`` of shape ``(n, d)`` matching ``jax.jacfwd``.
    Unlike ``jax.jacfwd`` (which vmaps tangents in parallel and adds a `d`
    dimension to every per-SN forward intermediate), ``lax.map`` runs each
    tangent's forward pass sequentially, so peak memory stays at one
    forward-pass worth instead of d-times-forward-pass.

    Compute is the same as ``jax.jacfwd`` (d forward passes total) plus a
    modest per-iter dispatch overhead from the sequential boundary.
    """
    d = p.shape[0]

    def col(e):
        _, Jv = jax.jvp(f, (p,), (e,))
        return Jv

    cols = jax.lax.map(col, jnp.eye(d))  # shape (d, n)
    return cols.T  # shape (n, d)


def _hessian_lax_map(f, p):
    """Memory-efficient Hessian via column-by-column HVPs with ``lax.map``.

    For scalar ``f: R^d -> R``, returns ``H`` of shape ``(d, d)`` matching
    ``jax.hessian(f)(p)``. Avoids the autodiff vmap multiplier on the
    per-SN forward intermediate.

    Compute is the same as ``jax.hessian`` (d HVP evaluations, each ~2
    forward-pass-equivalents).
    """
    grad_f = jax.grad(f)
    d = p.shape[0]

    def col(e):
        _, Hv = jax.jvp(grad_f, (p,), (e,))
        return Hv

    return jax.lax.map(col, jnp.eye(d))  # shape (d, d), symmetric


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


def _newton_lm_round_gn(init_p, bounds_lo, bounds_hi,
                        residuals_fn, prior_potential_fn, maxiter,
                        lam_init=1e-3, use_linesearch=False, debug=False):
    """LM with the Gauss-Newton approximation to the Hessian.

    The full-Hessian variant ``_newton_lm_round`` builds ``H = jax.hessian(loss)``,
    which materialises an O(per-SN forward × N_lat^2) intermediate during
    autodiff. This GN variant builds ``H = J^T J + H_prior`` where ``J`` is the
    Jacobian of the residuals and ``H_prior`` is the Hessian of the prior
    potential alone (no model intermediates). Memory drops by a factor of ~N_lat
    on the dominant intermediate; compute halves (one autodiff pass instead of
    two). Mathematically valid at a stationary point of a Gaussian likelihood;
    see Approach in plan.

    Args:
        residuals_fn: callable ``p -> 1-D residuals vector r(p)``. Defines the
            data-likelihood part of the loss as ``0.5 * sum(r^2)``.
        prior_potential_fn: callable ``p -> scalar``, the prior contribution
            to ``-log p(z)`` (including bijector log-det).

    Other args match ``_newton_lm_round``.
    """

    ls_alphas = jnp.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])

    def loss_total(p):
        r = residuals_fn(p)
        return 0.5 * jnp.sum(r * r) + prior_potential_fn(p)

    def newton_step(carry, _):
        p, lam, f_val = carry
        g = jax.grad(loss_total)(p)
        J = _jacfwd_lax_map(residuals_fn, p)
        H_lik = J.T @ J
        H_prior = jax.hessian(prior_potential_fn)(p)
        H = H_lik + H_prior

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
                return loss_total(tp)
            f_trials = jax.vmap(eval_alpha)(ls_alphas)
            best_idx = jnp.argmin(f_trials)
            best_alpha = ls_alphas[best_idx]
            trial_p = jnp.clip(p + best_alpha * d, bounds_lo, bounds_hi)
            f_trial = f_trials[best_idx]
        else:
            trial_p = jnp.clip(p + d, bounds_lo, bounds_hi)
            f_trial = loss_total(trial_p)

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

    f_init = loss_total(init_p)
    (p_final, _, f_final), diagnostics = jax.lax.scan(
        newton_step, (init_p, lam_init, f_init), None, length=maxiter
    )
    g_final = jax.grad(loss_total)(p_final)
    grad_norm = jnp.max(jnp.abs(g_final))
    return p_final, f_final, grad_norm, diagnostics


def run_lm_laplace_gn(predict_fn, prior_potential_fn, postprocess_fn_sn, z_template,
                     maxiter=30, lam_init=1e-3, use_linesearch=True):
    """Like ``run_lm_laplace`` but uses Gauss-Newton.

    Args:
        predict_fn: callable ``z_constrained_dict -> (flux, scale, data)`` that
            runs the model and returns the predicted obs, the per-obs noise
            scale, and the observed data. Built via numpyro handlers in
            ``bayesn_model.make_predict_fn``.
        prior_potential_fn: callable ``z_unc_dict -> scalar`` giving the
            negative log prior in unconstrained space (with bijector log-det).
            Built in ``bayesn_model.make_prior_potential_fn``.

    Note: the residuals function is constructed internally as
    ``r(p) = ((data - flux) / scale).ravel()``.
    """
    flat0, unflatten = ravel_pytree(z_template)
    bounds_hi = jnp.full_like(flat0, jnp.inf)
    bounds_lo = -bounds_hi

    def residuals_fn(p):
        z = unflatten(p)
        flux, scale, data, mask = predict_fn(z)
        return ((data - flux) / scale * mask).ravel()

    def prior_pot_flat(p):
        return prior_potential_fn(unflatten(p))

    p_final, _, _, diag = _newton_lm_round_gn(
        flat0, bounds_lo, bounds_hi, residuals_fn, prior_pot_flat,
        maxiter=maxiter, lam_init=lam_init,
        use_linesearch=use_linesearch, debug=True,
    )
    z_unc_dict = unflatten(p_final)
    median_dict = postprocess_fn_sn(z_unc_dict)
    return median_dict, diag['f_val'], z_unc_dict


def _newton_lm_round_hvp_cg(init_p, bounds_lo, bounds_hi,
                             loss_total_fn, maxiter,
                             lam_init=1e-3, use_linesearch=False, debug=False,
                             cg_maxiter=None):
    """LM with Hessian-vector products and conjugate-gradient linsolve.

    Avoids materialising J or H entirely. Each LM step queries Hv via
    ``jax.jvp(jax.grad(loss))`` (exact Hessian, no GN approximation) and
    solves ``(H + lam*I) d = -g`` iteratively with CG. Memory drops to
    per-SN-fwd × O(1) (no autodiff vmap multiplier). Uses Levenberg
    damping ``lam*I`` rather than Marquardt's ``lam*diag(H)`` because
    diag(H) would itself require d Hv evaluations to extract.
    """
    ls_alphas = jnp.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    grad_fn = jax.grad(loss_total_fn)
    if cg_maxiter is None:
        cg_maxiter = 2 * init_p.shape[0]

    def newton_step(carry, _):
        p, lam, f_val = carry
        g = grad_fn(p)

        def Hv_damped(v):
            _, Hv = jax.jvp(grad_fn, (p,), (v,))
            return Hv + lam * v

        d, _ = jsla.cg(Hv_damped, -g, maxiter=cg_maxiter)

        slope = jnp.dot(g, d)
        is_bad = (slope >= -1e-12) | jnp.any(jnp.isnan(d))
        # Fallback: gradient descent direction; line search picks magnitude.
        d = jnp.where(is_bad, -g, d)

        if use_linesearch:
            def eval_alpha(alpha):
                tp = jnp.clip(p + alpha * d, bounds_lo, bounds_hi)
                return loss_total_fn(tp)
            f_trials = jax.vmap(eval_alpha)(ls_alphas)
            best_idx = jnp.argmin(f_trials)
            best_alpha = ls_alphas[best_idx]
            trial_p = jnp.clip(p + best_alpha * d, bounds_lo, bounds_hi)
            f_trial = f_trials[best_idx]
        else:
            trial_p = jnp.clip(p + d, bounds_lo, bounds_hi)
            f_trial = loss_total_fn(trial_p)

        s = trial_p - p

        # Predicted reduction needs s^T H s; compute via one extra HVP.
        _, Hs = jax.jvp(grad_fn, (p,), (s,))
        actual = f_val - f_trial
        predicted = -(jnp.dot(g, s) + 0.5 * jnp.dot(s, Hs))
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

    f_init = loss_total_fn(init_p)
    (p_final, _, f_final), diagnostics = jax.lax.scan(
        newton_step, (init_p, lam_init, f_init), None, length=maxiter
    )
    g_final = grad_fn(p_final)
    grad_norm = jnp.max(jnp.abs(g_final))
    return p_final, f_final, grad_norm, diagnostics


def run_lm_laplace_hvp_cg(predict_fn, prior_potential_fn, postprocess_fn_sn,
                           z_template, maxiter=30, lam_init=1e-3,
                           use_linesearch=True, cg_maxiter=None):
    """HVP-CG variant of run_lm_laplace_gn. Uses exact Hessian (no GN approx)."""
    flat0, unflatten = ravel_pytree(z_template)
    bounds_hi = jnp.full_like(flat0, jnp.inf)
    bounds_lo = -bounds_hi

    def residuals_fn(p):
        z = unflatten(p)
        flux, scale, data, mask = predict_fn(z)
        return ((data - flux) / scale * mask).ravel()

    def loss_total_fn(p):
        r = residuals_fn(p)
        return 0.5 * jnp.sum(r * r) + prior_potential_fn(unflatten(p))

    p_final, _, _, diag = _newton_lm_round_hvp_cg(
        flat0, bounds_lo, bounds_hi, loss_total_fn,
        maxiter=maxiter, lam_init=lam_init,
        use_linesearch=use_linesearch, debug=True,
        cg_maxiter=cg_maxiter,
    )
    z_unc_dict = unflatten(p_final)
    median_dict = postprocess_fn_sn(z_unc_dict)
    return median_dict, diag['f_val'], z_unc_dict


def compute_hvp_scale_tril(predict_fn, prior_potential_fn, z_template):
    """Build the exact Hessian via ``_hessian_lax_map``, then chol(inv(H)).
    Memory matches that of one HVP. Same return shape as
    ``compute_gn_scale_tril``.
    """
    flat0, unflatten = ravel_pytree(z_template)

    def residuals_fn(p):
        z = unflatten(p)
        flux, scale, data, mask = predict_fn(z)
        return ((data - flux) / scale * mask).ravel()

    def loss_total_fn(p):
        r = residuals_fn(p)
        return 0.5 * jnp.sum(r * r) + prior_potential_fn(unflatten(p))

    H = _hessian_lax_map(loss_total_fn, flat0)
    cov = jnp.linalg.inv(H)
    return jnp.linalg.cholesky(cov)


def compute_gn_scale_tril(predict_fn, prior_potential_fn, z_template):
    """Cholesky of inv(J^T J + H_prior) at z_template.

    Memory-cheap GN replacement for ``compute_laplace_scale_tril``. Same
    return shape and semantics (Cholesky factor of the Laplace posterior
    covariance in unconstrained latent space) but avoids the
    ``jax.hessian`` autodiff blowup over the model.
    """
    flat0, unflatten = ravel_pytree(z_template)

    def residuals_fn(p):
        z = unflatten(p)
        flux, scale, data, mask = predict_fn(z)
        return ((data - flux) / scale * mask).ravel()

    def prior_pot_flat(p):
        return prior_potential_fn(unflatten(p))

    J = _jacfwd_lax_map(residuals_fn, flat0)
    H_lik = J.T @ J
    H_prior = jax.hessian(prior_pot_flat)(flat0)
    H = H_lik + H_prior
    cov = jnp.linalg.inv(H)
    return jnp.linalg.cholesky(cov)
