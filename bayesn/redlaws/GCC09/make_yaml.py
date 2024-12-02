""" From Gordon, Cartledge, & Clayton 2009ApJ...705.1320G
https://iopscience.iop.org/article/10.1088/0004-637X/705/2/1320
Average of 75 extinction curves with FUV and UV spectra.

Implementation based on dust_extinction
https://dust-extinction.readthedocs.io/en/stable/api/dust_extinction.parameter_averages.GCC09.html#dust_extinction.parameter_averages.GCC09
"""

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
        coeffs[var][component] = [P([0]) for _ in range(N)]

# for 3.3 <= x <= 5.9
wls[0] = (3.3, 5.9)
# a(x)_UV = 1.894 - 0.373*x - 0.0101 / ((x - 4.57)**2 + 0.0384)
# b(x)_UV = -3.490 + 2.057 * x + 0.706 / ((x - 4.59) ** 2 + 0.169)

# for 5.9 <= x <= 11.0
wls[1] = (5.9, 11.0)
# a(x)_FUV = a(x)_UV  + -0.110 * (x-5.9)**2 - 0.0100 * (x-5.9)**3
# b(x)_FUV = b(x)_UV  + 0.531 * (x-5.9)**2 + 0.0544 * (x-5.9)**3
for i in (0, 1):
    coeffs["A"]["poly"][i] = P([1.894, -0.373])
    coeffs["B"]["poly"][i] = P([-3.490, 2.057])
    coeffs["A"]["rem"][i] = P([-0.0101])
    coeffs["B"]["rem"][i] = P([0.706])
    coeffs["A"]["div"][i] = P([0.0384, 0, 1])(P([-4.57, 1]))
    coeffs["B"]["div"][i] = P([0.169, 0, 1])(P([-4.59, 1]))

shift = P([-5.9, 1])
a = P([0, 0, -0.110, -0.0100])
b = P([0, 0, 0.531, 0.0544])
coeffs["A"]["poly"][1] += a(shift)
coeffs["B"]["poly"][1] += b(shift)

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [3.3, 11]\n")
    f.write("RV_RANGE: [2.0, 6.0]\n")  # from dust_extinction. Sample in paper [2.5, 5.5]
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
