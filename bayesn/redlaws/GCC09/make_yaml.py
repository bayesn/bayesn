import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 2
wls = np.zeros((N, 2))
offsets = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 3.3 <= x <= 5.9
# a(x) = 1.894 - 0.373*x - 0.0101 / ((x - 4.57)**2 + 0.0384)
# b(x) = -3.490 + 2.057 * x + 0.706 / ((x - 4.59) ** 2 + 0.169)
wls[0] = [3.3, 5.9]
coeffs["A"]["poly"][0] = [-0.373, 1.894]
coeffs["A"]["rem"][0] = [-0.0101]
coeffs["A"]["div"][0] = [1, -4.57 * 2, 4.57**2 + 0.0384]
coeffs["B"]["poly"][0] = [2.057, -3.49]
coeffs["B"]["rem"][0] = [0.706]
coeffs["B"]["div"][0] = [1, -4.59 * 2, 4.59**2 + 0.169]

# for 5.9 <= x <= 11.0
# a(x) = NUV a(x) + -0.110 * (x-5.9)**2 - 0.0100 * (x-5.9)**3
# b(x) = NUV b(x) + 0.531 * (x-5.9)**2 + 0.0544 * (x-5.9)**3
# -5.9 offset adds -0.373*(-5.9) to a(x) and 2.057*(-5.9) to b(x)
# Rational coefficients go from (x - const)**2 + other const to
# ((x - 5.9) + new const)**2 + other const
# new const = 5.9  - const
wls[1] = [5.9, 11.0]
ap_NUV = P(coeffs["A"]["poly"][0][::-1])
bp_NUV = P(coeffs["A"]["poly"][0][::-1])
ap = P([0, 0, -0.110, -0.0100])
bp = P([0, 0, 0.531, 0.0544])
shift = P([-5.9, 1])
coeffs["A"]["poly"][1] = (ap(shift) + ap_NUV).coef[::-1]
coeffs["B"]["poly"][1] = (bp(shift) + bp_NUV).coef[::-1]
coeffs["A"]["rem"][1] = [-0.0101]
coeffs["B"]["rem"][1] = [0.706]
coeffs["A"]["div"][1] = [1, -4.57 * 2, 4.57**2 + 0.0384]
coeffs["B"]["div"][1] = [1, -4.59 * 2, 4.59**2 + 0.169]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
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
