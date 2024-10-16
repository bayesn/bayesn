import numpy as np
from numpy.polynomial import Polynomial

# A(lambda)/A(V) = (a(x) + b(x) / Rv) * correction polynomial
N = 4
wls = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# SNANA's hacky O94 -> F99 conversion uses an 10th order polynomial in kilo-angstrom space
# From degree 0 to 10, the coefficients are
f99_original_coeffs = np.array(
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
# Converting kilo-angstroms to units of inverse microns adds factors of 10 and makes the polynomials run from 0 to -10 rather than 0 to 10.
f99_coeffs = f99_original_coeffs * np.power(10, np.arange(11))
# factor out x**-10 and the coefficients are now in descending order from 10 to 0.
# f(x) = x**(-10)*np.polyval(f99_coeffs, x)
f99 = Polynomial(f99_coeffs[::-1])
# This gets multiplied with the O94 polynomial + the rational bit


# for 0.3 <= x <= 1.1, same as CCM89
# a(x) = 0.574*x**1.61 * x**(-10) * np.polyval(f99_coeffs, x)
# b(x) = -0.527*x**1.61 * x**(-10) * np.polyval(f99_coeffs, x)

wls[0] = [0.3, 1.1]
exps[0] = -8.39
coeffs["A"]["poly"][0] = 0.574 * f99_coeffs
coeffs["B"]["poly"][0] = -0.527 * f99_coeffs

# for 1.1 <= x <= 3.3
# y = x - 1.82
# The extinction package gives
# a[0] = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y + 0.701)*y - 0.609)*y + 0.104)*y + 1.0
# b[0] = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y - 3.989)*y + 2.908)*y + 1.952)*y
wls[1] = [1.1, 3.3]
exps[1] = -10
ap = Polynomial([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1][::-1])
bp = Polynomial([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0][::-1])
shift = Polynomial([-1.82, 1])
ap = ap(shift) * f99
bp = bp(shift) * f99
coeffs["A"]["poly"][1] = ap.coef[::-1]
coeffs["B"]["poly"][1] = bp.coef[::-1]


# for 3.3 <= x <= 5.9 same as CCM89
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
# b(x) = -3.090 + 1.825*x + 1/(0.829187*x**2 - 7.66169*x + 17.9166)
wls[2] = [3.3, 5.9]
exps[2] = -10
ap = Polynomial([1.752, -0.316])
ar = Polynomial([-0.104])
ad = Polynomial([4.67**2 + 0.341, -4.67 * 2, 1])
bp = Polynomial([-3.090, 1.825, -3.090])
br = Polynomial([1.206])
bd = Polynomial([4.62**2 + 0.263, -4.62 * 2, 1])
ap = f99 * ap + (f99 * ar) // ad
ar = (f99 * ar) % ad
bp = f99 * bp + (f99 * br) // bd
br = (f99 * br) % bd
coeffs["A"]["poly"][2] = ap.coef[::-1]
coeffs["B"]["poly"][2] = bp.coef[::-1]
coeffs["A"]["rem"][2] = ar.coef[::-1]
coeffs["B"]["rem"][2] = br.coef[::-1]
coeffs["A"]["div"][2] = ad.coef[::-1]
coeffs["B"]["div"][2] = bd.coef[::-1]

# for 5.9 <= x <= 8.0 same as CCM89
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# a(x) = 2.20335 - 0.809407*x + 0.128358*x**2 - 0.009779*x**3 - 0.104/((x-4.67)**2 + 0.341)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
# b(x) = -20.4647 + 11.9163*x - 1.92339*x**2 + 0.1207*x**3 + 1.206/((x-4.62)**2 + 0.263)
wls[3] = [5.9, 8.0]
exps[3] = -10
ap = Polynomial([2.20335, -0.809407, 0.128358, -0.009779])
ar = Polynomial([-0.104])
ad = Polynomial([4.67**2 + 0.341, -4.67 * 2, 1])
bp = Polynomial([-20.4647, 11.9163, -1.92339, 0.1207])
br = Polynomial([1.206])
bd = Polynomial([4.62**2 + 0.263, -4.62 * 2, 1])

ap = f99 * ap + (f99 * ar) // ad
ar = (f99 * ar) % ad
bp = f99 * bp + (f99 * br) // bd
br = (f99 * br) % bd

coeffs["A"]["poly"][3] = ap.coef[::-1]
coeffs["B"]["poly"][3] = bp.coef[::-1]
coeffs["A"]["rem"][3] = ar.coef[::-1]
coeffs["B"]["rem"][3] = br.coef[::-1]
coeffs["A"]["div"][3] = ad.coef[::-1]
coeffs["B"]["div"][3] = bd.coef[::-1]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"REGIME_EXP: [{', '.join(str(x) for x in exps)}]\n")
    for arr, name in zip(
        (
            wls,
            coeffs["A"]["poly"],
            coeffs["B"]["poly"],
            coeffs["A"]["rem"],
            coeffs["B"]["rem"],
            coeffs["A"]["div"],
            coeffs["B"]["div"],
        ),
        (
            "REGIME_WLS",
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
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
