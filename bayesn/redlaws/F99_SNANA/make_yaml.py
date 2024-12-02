""" From SNANA issue 1422 https://github.com/RickKessler/SNANA/issues/1422

F99 was being approximated as the product of O94 and a polynomial.
This provided a reasonable fit for R_V=3.1, but became discrepant with other values.
"""
import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.polynomial.polynomial import polydiv as pdiv

# A(lambda)/A(V) = (a(x) + b(x) / Rv) * correction polynomial
N = 4
wns = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(N)]

# SNANA's hacky O94 -> F99 conversion uses an 10th order polynomial in kilo-angstrom space
# From degree 0 to 10, the coefficients are
f99 = P(
    [
        8.55929205e-02,
        1.91547833e00,
        -1.65101945e00,
        7.50611119e-01,
        -2.00041118e-01,
        3.30155576e-02,
        -3.46344458e-03,
        2.30741420e-04,
        -9.43018242e-06,
        2.14917977e-07,
        -2.08276810e-09,
    ]
)
# f(kiloangstroms) = np.polyval(f99_original_coeffs[::-1], kiloangstroms)
# Converting kilo-angstroms to units of inverse microns adds factors of 10
f99 = P(f99.coef * np.power(10, np.arange(11)))
# and makes the polynomials run from 0 to -10 rather than 0 to 10.
f99 = P(f99.coef[::-1])
exps -= 10
# This is what gets multiplied with the O94 polynomial + the rational bit

# for 0.3 <= x <= 1.1, same as CCM89
wns[0] = (0.3, 1.1)
# a(x) = 0.574*x**1.61 * x**(-10) * f99
# b(x) = -0.527*x**1.61 * x**(-10) * f99
exps[0] += 1.61
coeffs["A"]["poly"][0] = 0.574 * f99
coeffs["B"]["poly"][0] = -0.527 * f99

# for 1.1 <= x <= 3.3
wns[1] = (1.1, 3.3)
# y = x - 1.82
shift = P([-1.82, 1])
# The extinction package gives
# a[0] = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y + 0.701)*y - 0.609)*y + 0.104)*y + 1.0
# b[0] = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y - 3.989)*y + 2.908)*y + 1.952)*y
a_opt = P([1, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
b_opt = P([0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
coeffs["A"]["poly"][1] = a_opt(shift) * f99
coeffs["B"]["poly"][1] = b_opt(shift) * f99

# for 3.3 <= x <= 5.9 same as CCM89
wns[2] = (3.3, 5.9)
# a(x)_UV = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# b(x)_UV = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
for var, poly_coeffs, rem, div, div_shift in zip(
    "AB",
    ((1.752, -0.316), (-3.090, 1.825)),
    (-1.04, 1.206),
    (0.341, 0.263),
    (-4.67, -4.62),
):
    p = f99 * P(poly_coeffs)
    r = f99 * P([rem])
    d = P([div, 0, 1])(P([div_shift, 1]))
    quotient_coef, rem_coef = pdiv(r.coef, d.coef)
    coeffs[var]["poly"][2] = p + P(quotient_coef)
    coeffs[var]["rem"][2] = P(rem_coef)
    coeffs[var]["div"][2] = d
# for 5.9 <= x <= 8.0 same as CCM89
wns[3] = (5.9, 8.0)
# a(x)_FUV = a(x)_UV - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# b(x)_FUV = b(x)_UV + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
shift = P([-5.9, 1])
for var, quad, cubic in zip("AB", (-0.04473, 0.2130), (-0.009779, 0.1207)):
    coeffs[var]["poly"][3] = coeffs[var]["poly"][2] + f99 * P([0, 0, quad, cubic])(
        shift
    )
    coeffs[var]["rem"][3] = coeffs[var]["rem"][2]
    coeffs[var]["div"][3] = coeffs[var]["div"][2]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [0.286, 10.964]\n")
    f.write(
        f"RV_RANGE: [2.0, 6.0]\n"
    )  # F99 range from dust_extinction. Sample in paper [2.22, 5.83]
    # only replicates F99 at RV=3.1
    f.write(f"REGIME_EXP: [{', '.join(str(x) for x in exps)}]\n")
    for arr, name in zip(
        (
            wns,
            coeffs["A"]["poly"],
            coeffs["B"]["poly"],
            coeffs["A"]["rem"],
            coeffs["B"]["rem"],
            coeffs["A"]["div"],
            coeffs["B"]["div"],
        ),
        (
            "REGIMES",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
            "A_REMAINDER_COEFFS",
            "B_REMAINDER_COEFFS",
            "A_DIVISOR_COEFFS",
            "B_DIVISOR_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            if isinstance(coeffs, P):
                coeffs = coeffs.coef[::-1]
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
