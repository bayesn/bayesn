""" From Valencic, Clayton, Gordon 2004ApJ...616..912V
https://iopscience.iop.org/article/10.1086/424922

Average of 417 UV extinction curves using IUE spectra
combined with 2MASS NIR photometry.

RV range [2.01 +/- 0.33 (HD 96042), 6.33 +/- 0.60 (HD 37020)]

Implementation based on dust_extinction
https://dust-extinction.readthedocs.io/en/stable/api/dust_extinction.parameter_averages.VCG04.html#dust_extinction.parameter_averages.VCG04
"""

import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 2
wns = np.zeros((N, 2))
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 3.3 <= x <= 5.9
wns[0] = [3.3, 5.9]
# a(x)_UV = 1.808 - 0.215 * x - 0.134 / ((x - 4.558) ** 2 + 0.566)
# b(x)_UV = -2.350 + 1.403 * x + 1.103 / ((x - 4.587) ** 2 + 0.263)
# for 5.9 <= x <= 8.0
wns[1] = (5.9, 8.0)
# a(x)_FUV = a(x)_UV + -0.0077 * (x-5.9)**2 - 0.0030 * (x-5.9)**3
# b(x)_FUV = b(x)_UV + 0.2060 * (x-5.9)**2 + 0.0550 * (x-5.9)**3
for i in (0, 1):
    coeffs["A"]["poly"][i] = P([1.808, -0.215])
    coeffs["B"]["poly"][i] = P([-2.35, 1.403])
    coeffs["A"]["rem"][i] = P([-0.134])
    coeffs["B"]["rem"][i] = P([1.103])
    coeffs["A"]["div"][i] = P([0.566, 0, 1])(P([-4.558, 1]))
    coeffs["B"]["div"][i] = P([0.263, 0, 1])(P([-4.587, 1]))

shift = P([-5.9, 1])
a = P([0, 0, -0.0077, -0.0030])
b = P([0, 0, 0.2060, 0.0550])
coeffs["A"]["poly"][1] += a(shift)
coeffs["B"]["poly"][1] += b(shift)

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [3.3, 8.0]\n")
    f.write(
        "RV_RANGE: [2.0, 6.0]\n"
    )  # from dust_extinction. Sample in paper [2.01, 6.33]
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
