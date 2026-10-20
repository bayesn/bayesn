from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.interpolate import CubicSpline

from bayesn.spline_utils import *
from bayesn.spline_utils import _parse_bc_type, _clean_extrap_arg

def init_spline(rng, range_mod):
    """ positive range_mod gives x_int within the span of x.
    If x_range is less than x_int will be reverse sorted.
    """
    x = np.sort(rng.uniform(*sorted([rng.choice(100), rng.choice(100)]), 30))
    x -= np.median(x)
    x_min, x_max = min(x), max(x)
    x_range = x_max - x_min
    x_int = np.linspace(x_min+range_mod*x_range, x_max-range_mod*x_range, 50)
    # Testign with a polynomial with degree 4 <= deg <= 6.
    yk = np.sin(x) + rng.normal(0, 1, len(x))
    padded_yk = np.concatenate([[1], yk])
    return x, x_int, yk, padded_yk

def test_cartesian_prod():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, rng.choice(10))
    y = rng.normal(10, 1, rng.choice(10))
    cart_ver = cartesian_prod(x, y)
    iter_ver = np.array([i for i in product(x, y)])
    assert (cart_ver == iter_ver).flatten().all()

@pytest.mark.parametrize("bc_type", ("natural", "not-a-knot", "1", "2"))
@pytest.mark.parametrize("rng_seed", (0, 1))
def test_spline_coeffs(rng_seed, bc_type):
    rng = np.random.default_rng(rng_seed)
    if bc_type == "1":
        bc_type = ((1, rng.normal()), (1, rng.normal()))
    elif bc_type == "2":
        bc_type = ((2, rng.normal()), (2, rng.normal()))
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=0.1)
    J = spline_coeffs(x_int, x, invKD(x, bc_type=bc_type))
    bayesn_ver = np.matmul(J, padded_yk)
    scipy_ver = CubicSpline(x, yk, bc_type=bc_type)(x_int)
    assert np.isclose(bayesn_ver, scipy_ver).all()

def test_parse_bc_type_bad():
    with pytest.raises(ValueError, match="Unrecognised bc_type"):
        _parse_bc_type((1, 2, 3))

def test_invKD_bad_bc_type():
    with pytest.raises(NotImplementedError):
        invKD(np.arange(5), bc_type="periodic")
    with pytest.raises(ValueError, match="not recognised"):
        invKD(np.arange(5), bc_type="bad_type")

def test_clean_extrap_arg():
    with pytest.warns(UserWarning, match="Higher order extrap"):
        _, extrap = _clean_extrap_arg(extrap=4)
        assert extrap == 3
    with pytest.raises(ValueError, match="Negative order"):
        _clean_extrap_arg(extrap=-1)
    with pytest.raises(ValueError, match="Multiple substrings"):
        _clean_extrap_arg(extrap="quadlinear")
    with pytest.raises(ValueError, match="If a string, it should contain"):
        _clean_extrap_arg(extrap="squared")

    res = _clean_extrap_arg("to_0_100_1e3")
    assert res[0] == "to 0"
    npt.assert_allclose(res[1], (100, 1000))

    res = _clean_extrap_arg(("to_0", 12.345, 67.89))
    assert res[0] == "to 0"
    npt.assert_allclose(res[1], (12.345, 67.89))
    with pytest.raises(ValueError, match="extrap argument not recognised"):
        _clean_extrap_arg(("to_0", 20, 10))  # 20 not < 10

@pytest.mark.parametrize("rng_seed", (0,))
def test_spline_coeffs_out_of_bounds(rng_seed):
    rng = np.random.default_rng(rng_seed)
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=-0.2)
    with pytest.raises(ValueError, match="extrap was specified as None"):
        spline_coeffs(x_int, x, invKD(x), extrap=None)

@pytest.mark.parametrize("rng_seed", (0, 1, 2))
def test_poly_extrap_linear(rng_seed):
    rng = np.random.default_rng(rng_seed)
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=-0.2)
    bc_type = ((1, rng.normal()), (1, rng.normal()))
    J = spline_coeffs(x_int, x, invKD(x, bc_type=bc_type), extrap="linear")
    y = J @ padded_yk
    low_idx = np.where(x_int < min(x))
    high_idx = np.where(x_int > max(x))
    assert np.isclose(y[low_idx], yk[0] + (x_int[low_idx] - min(x))*bc_type[0][1]).all()
    assert np.isclose(y[high_idx], yk[-1] + (x_int[high_idx] - max(x))*bc_type[1][1]).all()

@pytest.mark.parametrize("rng_seed", (0, 1, 2))
def test_poly_extrap_cubic(rng_seed):
    rng = np.random.default_rng(rng_seed)
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=-0.2)
    bc_type = ((1, rng.normal()), (1, rng.normal()))
    J = spline_coeffs(x_int, x, invKD(x, bc_type=bc_type), extrap="cubic")
    bayesn_ver = np.matmul(J, padded_yk)
    scipy_ver = CubicSpline(x, yk, bc_type=bc_type)(x_int)
    assert np.isclose(bayesn_ver, scipy_ver).all()

@pytest.mark.parametrize("rng_seed", (0, 1, 2))
def test_extrap_to_0(rng_seed):
    """ Extrapolation to 0 should be equivalent to a cubic spline with knots at
    (x0, 0) and (x[0], yk[0]) on the left and (x[-1], yk[-1]) and (xN1, 0) on the right
    with boundary conditions ((1, 0), (1, y_prime_left)) on the left and
    ((1, y_prime_right), (1, 0)) on the right.
    """
    rng = np.random.default_rng(rng_seed)
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=-0.3)
    y_prime_left, y_prime_right = rng.normal(size=2)
    x0, xN1 = x_int[0], x_int[-1]
    J = spline_coeffs(
        x_int, x, invKD(x, bc_type=((1, y_prime_left), (1, y_prime_right))),
        extrap=("to_0", x0, xN1)
    )
    # test left
    idx = np.where(x_int < x[0])[0]
    ref_x_int = x_int[idx]
    ref_x = np.array([x0, x[0]])
    ref_J = spline_coeffs(
        ref_x_int, ref_x, invKD(ref_x, bc_type=((1, 0), (1, y_prime_left)))
    )
    npt.assert_allclose((J @ padded_yk)[idx], ref_J @ np.array([1, 0, yk[0]]))
    # test right
    idx = np.where(x_int > x[-1])[0]
    ref_x_int = x_int[idx]
    ref_x = np.array([x[-1], xN1])
    ref_J = spline_coeffs(
        ref_x_int, ref_x, invKD(ref_x, bc_type=((1, y_prime_right), (1, 0)))
    )
    npt.assert_allclose((J @ padded_yk)[idx], ref_J @ np.array([1, yk[-1], 0]))


@pytest.mark.parametrize("bc_type", ("natural", "not-a-knot", "1", "2"))
@pytest.mark.parametrize("rng_seed", (0, 1, 2))
def test_spline_coeffs_step(rng_seed, bc_type):
    rng = np.random.default_rng(rng_seed)
    x, x_int, yk, padded_yk = init_spline(rng, range_mod=-0.2)
    spline_map = jax.jit(
        jax.vmap(spline_coeffs_step, in_axes=(0, None, None, None)),
        static_argnames=("extrap",)
    )
    if bc_type == "1":
        bc_type = ((1, rng.normal()), (1, rng.normal()))
    elif bc_type == "2":
        bc_type = ((2, rng.normal()), (2, rng.normal()))
    # with pytest.raises(jax.errors.TracerArrayConversionError):
    #     spline_map(x_int, x, invKD(x, bc_type=bc_type), 3)  # second and third args must be jax arrays
    J = spline_map(x_int, jnp.array(x), jnp.array(invKD(x, bc_type=bc_type)), 3)
    scipy_ver = CubicSpline(x, yk, bc_type=bc_type)(x_int)
    assert all(np.isclose(J @ padded_yk, scipy_ver))
