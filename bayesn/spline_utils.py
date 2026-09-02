"""
BayeSN Spline Utilities. Defines a set of functions which carry out the
2D spline operations essential to BayeSN.
"""

from numbers import Number
from typing import Any
from warnings import warn

import numpy as np
import jax.numpy as jnp
from jax.lax import cond
from jax.typing import ArrayLike

no_op = lambda x: x
def _parse_bc_type(bc_type: ArrayLike | str):
    bcs = bc_type
    if isinstance(bc_type, str) or (len(bc_type) == 2 and isinstance(bc_type[1], Number)):
        bcs = (bc_type, bc_type)
    if len(bcs) != 2:
        raise ValueError(
            f"Unrecognised bc_type: {bc_type}. There are three families of acceptable values:\n"
            "    1) A string in 'natural', 'nat', 'not-a-knot', or 'nak'.\n"
            "    2) An ArrayLike with length 2 where the first element is 1, 'lin', 2, "
            "or 'quad' and the second element is a number.\n"
            "    3) An ArrayLike with length 2 where the elements match the previous two "
            "patterns."
        )
    return bcs

def invKD(x, bc_type="natural"):
    """
    Compute K^{-1}D for a set of spline knots.

    For knots y at locations x, the vector, y'' of non-zero second
    derivatives is constructed from y'' = K^{-1}Dy, where K^{-1}D
    is independent of y, meaning it can be precomputed and reused for
    arbitrary y to compute the second derivatives of y.

    Parameters
    ----------
    x : :py:class:`numpy.array`
        Numpy array containing the locations of the cubic spline knots.

    Returns
    -------
    KD : :py:class:`numpy.array`
        y independent matrix whose product can be taken with y to
        obtain a vector of second derivatives of y.
    """
    bc_type = _parse_bc_type(bc_type)
    x = np.asarray(x)
    n = len(x)

    K = np.zeros((n,n))
    D = np.zeros((n,n+1))
    # continuity equation
    for j in np.arange(1,n-1):
        K[j, j-1:j+2] = [(x[j] - x[j-1])/6, (x[j+1] - x[j-1])/3, (x[j+1] - x[j])/6]
        D[j, j:j+3] = [1./(x[j] - x[j-1]), -(1./(x[j+1] - x[j]) + 1./(x[j] - x[j-1])), 1./(x[j+1] - x[j])]

    # handling boundary conditions the same as scipy.interpolate.CubicSpline
    for bc, i, start in zip(bc_type, (0, n-1), (True, False)):
        # start and end have similar logic, but need to do some index tricks
        i3 = i - 2*int(not start) # 0 or n-3
        i2 = i - int(not start) # 0 or n-2
        if bc == "natural" or bc == "nat":
            K[i,i] = 1
        elif bc == "not-a-knot" or bc == "nak":
            K[i,i3:i3+3] = [x[i3+1] - x[i3+2], x[i3+2] - x[i3], x[i3] - x[i3+1]]
        elif bc == "periodic":
            raise NotImplementedError
        elif len(bc) == 2 and (bc[0] == 1 or bc[0] == "lin"):
            K[i,i2:i2+2] = [-(x[i2+1] - x[i2])*(3*int(start)-1)/6, (x[i2+1] - x[i2])*(3*int(not start)-1)/6]
            # assuming \vec{y} will be padded with a 1 at the end such that D[i,-1] can
            # provide arbitrary constants for the ith y''. Need to pad col_idx by 1.
            D[i,0] = bc[1]
            D[i,i2+1:i2+1+2] = [1/(x[i2+1] - x[i2]), -1/(x[i2+1] - x[i2])]
        elif len(bc) == 2 and (bc[0] == 2 or bc[0] == "quad"):
            K[i,i] = 1
            D[i,0] = bc[1]
        else:
            raise ValueError(
                f"bc_type {bc_type} not recognized. It should be either 'natural', "
                "'not-a-knot', or a 2-tuple for the left and right boundaries. The "
                "elements can either be the aforementioned strings or themselves "
                "2-tuples with the first element indicating 1 or 2 for the first or "
                "second derivative and the second element indicating its value."
            )
    return np.linalg.solve(K, D)
def cartesian_prod(x, y):
	"""
	Compute cartesian product of two vectors.

	Parameters
	----------
	x : :py:class:`numpy.array`
		First vector.
	x : :py:class:`numpy.array`
		Second vector.

	Returns
	-------
	z : :py:class:`numpy.array`
		Cartesian product of x and y.
	"""
	n_x = len(x)
	n_y = len(y)
	return np.array([np.repeat(x,n_y),np.tile(y,n_x)]).T

def _clean_extrap_arg(extrap: None | int | str) -> tuple[str, Any]:
    """ Given a string type argument containing substrings "lin", "quad", or "cub" or
    a number type argument, return ("poly", order) where order is 1, 2, or 3.
    Given a string type argument of the form "to_0_x0_xN1" or a container of length 3
    comprising the string "to 0" and values for x0 and xN1, return ("to 0", x0, xN1).
    Given an argument of None, return ("no extrap", None).
    """
    if isinstance(extrap, Number):
        order = int(extrap)
        num_warning = (
            "Polynomial extrapolation of a cubic spline is limited to constant, "
            "linear, quadratic, or cubic order."
        )
        if order > 3:
            warn(UserWarning(
                f"{num_warning} Higher order extrapolation is identical to cubic "
                f"extrapolation. int(extrap)={order} will be interpreted as extrap=3."
            ))
            order = 3
        elif order < 0:
            raise ValueError(
                f"{num_warning} Negative order int(extrap)={order} is not supported."
            )

        return "poly", order
    elif isinstance(extrap, str):
        order = 0
        lin = "lin" in extrap
        quad = "quad" in extrap
        cub = "cub" in extrap
        if lin + quad + cub > 1:
            raise ValueError(
                "extrap argument not recognised. Multiple substrings from 'lin', "
                f"'quad', and 'cub' were identified within {extrap}. This cannot be "
                "unambiguously mapped to a polynomial extrapolation order."
            )
        if lin: return "poly", 1
        if quad: return "poly", 2
        if cub: return "poly", 3
        if extrap.startswith("to_0"):
            x0, xN1 = extrap.split("_")[2:4]
            return "to 0", (float(x0), float(xN1))
        raise ValueError(
            "extrap argument not recognised. If a string, it should contain one of "
            "substrings 'lin', 'quad', or 'cub', or else have the form 'to_0_x0_xN1' "
            "where x0 and xN1 are numbers indicating the locations of the extremal "
            "knots with values and first derivatives equal to 0."
        )
    elif (
        isinstance(extrap, tuple | list | ArrayLike)
        and len(extrap) == 3
        and str(extrap[0]).replace("_", " ") == "to 0"
        and isinstance(extrap[1], Number)
        and isinstance(extrap[2], Number)
        and extrap[1] < extrap[2]
    ):
        return "to 0", (extrap[1], extrap[2])
    raise ValueError("""extrap argument not recognised. Supported values include:
        None;
        numbers with int(extrap) mappings 1, 2, or 3;
        strings with one of the substrings 'lin', 'quad', or 'cub'
        strings of the form 'to_0_x0_xN1' where x0 and xN1 are numbers indicating
            the locations of the extremal knots with values and first derivatives equal
            to 0.
        or tuple/list/ArrayLike with length 3 where the first element is "to 0" or
            "to_0" and the second and third elements are the knot locations.""")

def spline_coeffs(x_int: ArrayLike, x: ArrayLike, invkd: ArrayLike, extrap: None | int | str = 3):
    """
    Compute a matrix of spline coefficients.

    Given a set of knots at x, with values y, compute a matrix, J, which
    can be multiplied into a [1, *y] to evaluate the cubic spline at points x_int.
    Numerical recipes in C (https://numerical.recipes; Press et al. 1992) gives
    y(x[j] < x_int < x[j+1]) = A*y[j] + B*y[j+1] + C*y''[j] + D*y''[j+1]
    where
    A = 1 - B = (x[j+1] - x_int)/(x[j+1] - x[j])
    C = 1/6 * (A**3 - A) * (x[j+1] - x_j)**2
    D = 1/6 * (B**3 - B) * (x[j+1] - x_j)**2
    Given that y''(x) = K^{-1}D y = invkd y, the eqn can be rewritten as
    y(x[j] < x_int < x[j+1]) = A*y[j] + B*y[j+1] + C*(invkd y)[j] + D*(invkd y)[j+1]
    By iterating over knot indices, and we can get matrix J such that y(x_int) = Jy.

    Parameters
    ----------
    x_int: shape (N_x)
    	2D array containing the locations which the output matrix will
    	interpolate the spline to.
    x: shape (N_xk)
    	2D array containing the locations of the spline knots.
    invkd: shape (N_xk, N_xk+1)
    	Precomputed matrix for generating second derivatives. Can be obtained
    	from the output of ``invKD``.
    extrap: None or str in "linear", "quad", or "cubic" (default)
    	Extrapolation type. If None, there will be an error raised when x_int is not
        spanned by the knot locations at x.
        If "linear", "quad", or "cubic", extrapolation will follow the corresponding
    Returns
    -------
    J: shape (N_x, N_xk+1)
    	y independent matrix whose product can be taken with [1, *y] to evaluate
    	the spline at x_int.
    """
    x_int = np.asarray(x_int)
    x = np.asarray(x)
    invkd = np.asarray(invkd)
    n_x_int = len(x_int)
    n_x = len(x)
    J = jnp.zeros((n_x_int,n_x+1))

    up = x_int > x[-1]
    down = x_int < x[0]
    interp = ~(up | down)

    if interp.any():
        i = np.where(interp)[0]
        xi = x_int[i]
        # bracket index q so x[q] <= xi < x[q+1]; clip handles xi == x[-1]
        q = np.minimum(np.searchsorted(x, xi, side='right') - 1, n_x - 2)
        dx = x[q + 1] - x[q]
        A = (x[q + 1] - xi) / dx
        B = 1 - A
        C = ((A ** 3 - A) / 6) * dx ** 2
        D = ((B ** 3 - B) / 6) * dx ** 2
        J = J.at[i, q+1].set(A)
        J = J.at[i, q+2].set(B)
        J = J.at[i, :].add(C[:,None] * invkd[q] + D[:,None] * invkd[q+1])
    if not (up | down).any():
        return J
    extrap_type, extrap_args  = _clean_extrap_arg(extrap)
    if extrap_type == "to 0":
        up_fn = _extrap_to_0_up
        down_fn = _extrap_to_0_down
        down_arg, up_arg = extrap_args
    if extrap_type == "poly":
        up_fn = _poly_extrap_up
        down_fn = _poly_extrap_down
        up_arg, down_arg = [extrap_args for _ in range(2)]
    if up.any():
        J = J.at[up].set(up_fn(x_int[up], x, invkd, up_arg))
    if down.any():
        J = J.at[down].set(down_fn(x_int[down], x, invkd, down_arg))
    return J

def _poly_extrap_up(x_int: ArrayLike, x: ArrayLike, invkd: ArrayLike, order: int = 3):
    r"""
    The spline evaluation from spline_coeffs may require extrapolation, which this
    method provides as an "order"th order Taylor expansion with 1-based knot indices
        T_{order}(x_int < x_1) = \sum_{k=0}^order y^{(k)}_1}(x^* - x_1)^k/k!
        T_{order}(x_int > x_N) = \sum_{k=0}^order y^{(k)}_N}(x^* - x_N)^k/k!
    y^{(k)} indicates the kth derivative of y(x) at x_1 or x_N.
    """
    J_up = jnp.zeros((len(x_int), len(x)+1))
    dx = x[-1] - x[-2]
    A = (x[-1] - x_int)/dx
    B = 1 - A
    f = (x_int - x[-1])*dx/6.0
    extrap_const = (x_int - x[-1])/dx

    J_up = J_up.at[:,-2].set(A)
    J_up = J_up.at[:,-1].set(B)
    E = jnp.ones(len(x_int))
    F = jnp.ones(len(x_int))*2
    E = cond(order == 3, lambda E: E - extrap_const**2, no_op, E)
    F = cond(order == 2, lambda F: F + 3*extrap_const, no_op, F)
    F = cond(order == 3, lambda F: F + 3*extrap_const + extrap_const**2, no_op, F)
    J_up = J_up.at[:].add(f[:,None]*(E[:,None]*invkd[-2][None,:] + F[:,None]*invkd[-1][None,:]))
    return J_up

def _poly_extrap_down(x_int: ArrayLike, x: ArrayLike, invkd: ArrayLike, order: int = 3):
    J_down = jnp.zeros((len(x_int), len(x)+1))
    dx = x[1] - x[0]
    B = (x_int - x[0])/dx
    A = 1 - B
    f = -(x_int - x[0])*dx/6.0

    J_down = J_down.at[:,1].set(A)
    J_down = J_down.at[:,2].set(B)
    extrap_const = B
    E = jnp.ones(len(x_int))*2
    F = jnp.ones(len(x_int))
    E = cond(order == 2, lambda E: E - 3*extrap_const, no_op, E)
    E = cond(order == 3, lambda E: E - 3*extrap_const + extrap_const**2, no_op, E)
    F = cond(order == 3, lambda F: F - extrap_const**2, no_op, F)
    J_down = J_down.at[:].add(f[:,None]*(E[:,None]*invkd[0][None, :] + F[:,None]*invkd[1][None, :]))
    return J_down

def _extrap_to_0_up(x_int: ArrayLike, x: ArrayLike, invkd: ArrayLike, xN1: float = 25000):
    """ Reversion to 0 where y an y' are smooth and 0 by xN1."""
    J_up = jnp.zeros((len(x_int), len(x)+1))
    idx = x_int < xN1
    t = (x_int[idx] - x[-1])/(xN1 - x[-1])
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    f = h10*(x[-1] - x[-2])*(xN1 - x[-1])
    J_up = J_up.at[idx,-2].set(-h10*(xN1 - x[-1])/(x[-1] - x[-2]))
    J_up = J_up.at[idx,-1].set(h00 + h10*(xN1 - x[-1])/(x[-1] - x[-2]))
    J_up = J_up.at[idx].add(f[:,None]*(invkd[-2][None,:] + 2*invkd[-1][None,:])/6)
    return J_up

def _extrap_to_0_down(x_int: ArrayLike, x: ArrayLike, invkd: ArrayLike, x0: float = 1000):
    """ Reversion to 0 where y an y' are smooth and 0 by x0."""
    J_down = jnp.zeros((len(x_int), len(x)+1))
    idx = x_int > x0
    t = (x_int[idx] - x0)/(x[0] - x0)
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    f = -h11*(x[1] - x[0])*(x[0] - x0)
    J_down = J_down.at[idx,1].set(h01 - h11*((x[0] - x0)/(x[1] - x[0])))
    J_down = J_down.at[idx,2].set(h11*((x[0] - x0)/(x[1] - x[0])))
    J_down = J_down.at[idx].add(f[:, None]*(2*invkd[0][None, :] + invkd[1][None, :])/6)
    return J_down

def spline_coeffs_step(x_now, x, invkd, extrap=1):
    """ JAX friendly version of spline_coeffs.
    To be used with vmapping e.g.
    J_t_map = jax.jit(jax.vmap(spline_coeffs_stest_step, in_axes=(0, None, None)))
    """
    shape = list(x.shape)
    shape[0] += 1
    J = jnp.zeros(shape)
    up_extrap = x_now > x[-1]
    down_extrap = x_now < x[0]
    interp = 1 - up_extrap - down_extrap

    # extrapolation for x_now > max(x)
    dx = x[-1] - x[-2]
    A = (x[-1] - x_now) / dx
    B = 1 - A
    f = (x_now - x[-1]) * dx / 6.0

    J = J.at[-2].set(J[-2] + A * up_extrap)
    J = J.at[-1].set(J[-1] + B * up_extrap)
    E, F = 1., 2.
    extrap_const = -A
    E = cond(extrap == 3, lambda E: E - extrap_const**2, no_op, E)
    F = cond(extrap == 2, lambda F: F + 3*extrap_const, no_op, F)
    F = cond(extrap == 3, lambda F: F + 3*extrap_const + extrap_const**2, no_op, F)
    J = J.at[:].set(J[:] + f * (E*invkd[-2, :] + F*invkd[-1, :]) * up_extrap)

    # extrapolation for x_now < min(x)
    dx = x[1] - x[0]
    B = (x_now - x[0]) / dx
    A = 1 - B
    f = (x_now - x[0]) * dx / 6.0

    J = J.at[1].set(J[1] + A * down_extrap)
    J = J.at[2].set(J[2] + B * down_extrap)
    E, F = 2., 1.
    extrap_const = B
    E = cond(extrap == 2, lambda E: E - 3*extrap_const, no_op, E)
    E = cond(extrap == 3, lambda E: E - 3*extrap_const + extrap_const**2, no_op, E)
    F = cond(extrap == 3, lambda F: F - extrap_const**2, no_op, F)
    J = J.at[:].set(J[:] - f * (E*invkd[0, :] + F*invkd[1, :]) * down_extrap)

    # spline interpolation
    q = jnp.argmax(x_now <= x) - 1
    dx = x[q + 1] - x[q]
    A = (x[q + 1] - x_now) / dx
    B = 1 - A
    C = ((A ** 3 - A) / 6) * dx ** 2
    D = ((B ** 3 - B) / 6) * dx ** 2

    J = J.at[q+1].set(J[q+1] + A * interp)
    J = J.at[q+2].set(J[q+2] + B * interp)
    J = J.at[:].set(J[:] + C * invkd[q, :] * interp + D * invkd[q + 1, :] * interp)
    return J
