""" From O'Donnell 1994ApJ...422..158O
https://ui.adsabs.harvard.edu/abs/1994ApJ...422..158O/abstract
Rederived CCM89 with uvby observations of 22 stars and Johnson filters,
IUE, and ANS extinction measurements.
Uses new coefficients the optical.

RV
min = 2.85, HD 14250
max = 5.6, HD 36982


Implementation based on extinction
https://extinction.readthedocs.io/en/latest/api/extinction.odonnell94.html
"""

import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 4
wns = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(N)]

# for 0.3 <= x <= 1.1, same as CCM89
# a(x) = 0.574*x**1.61
# b(x) = -0.527*x**1.61
wns[0] = (0.3, 1.1)
exps[0] = 1.61
coeffs["A"]["poly"][0] = P([0.574])
coeffs["B"]["poly"][0] = P([-0.527])

# for 1.1 <= x <= 3.3
wns[1] = (1.1, 3.3)
# y = x - 1.82
shift = P([-1.82, 1])
# The extinciton package gives
# a(x)_opt = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y + 0.701)*y - 0.609)*y + 0.104)*y + 1.0
# b(x)_opt = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y - 3.989)*y + 2.908)*y + 1.952)*y
a = P([1, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
b = P([0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])
coeffs["A"]["poly"][1] = a(shift)
coeffs["B"]["poly"][1] = b(shift)

# for 3.3 <= x <= 5.9 same as CCM89
wns[2] = [3.3, 5.9]
# a(x)_UV = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# b(x)_UV = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
coeffs["A"]["poly"][2] = P([1.752, -0.316])
coeffs["A"]["rem"][2] = P([-0.104])
coeffs["A"]["div"][2] = P([0.341, 0, 1])(P([-4.67, 1]))
coeffs["B"]["poly"][2] = P([-3.090, 1.825])
coeffs["B"]["rem"][2] = P([1.206])
coeffs["B"]["div"][2] = P([0.263, 0, 1])(P([-4.62, 1]))

# for 5.9 <= x <= 8.0 same as CCM89
wns[3] = [5.9, 8.0]
# a(x)_FUV = a(x)_UV - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# b(x)_FUV = b(x)_UV + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
shift = P([-5.9, 1])
coeffs["A"]["poly"][3] = coeffs["A"]["poly"][2] + P([0, 0, -0.04473, -0.009779])(shift)
coeffs["A"]["rem"][3] = coeffs["A"]["rem"][2]
coeffs["A"]["div"][3] = coeffs["A"]["div"][2]
coeffs["B"]["poly"][3] = coeffs["B"]["poly"][2] + P([0, 0, 0.2130, 0.1207])(shift)
coeffs["B"]["rem"][3] = coeffs["B"]["rem"][2]
coeffs["B"]["div"][3] = coeffs["B"]["div"][2]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [0.3, 8.0]\n")
    f.write("RV_RANGE: [2.0, 6.0]\n")  # from dust_extinction. Sample in paper [2.85, 5.6]
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
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
