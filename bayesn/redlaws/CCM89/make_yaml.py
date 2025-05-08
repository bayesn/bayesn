""" From Cardelli, Clayton, and Mathis 1989ApJ...345..245C
https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract
Fits parameterized extinction data from Fitzpatrick (1986) and Massa (1988) as a function
of one-parameter:$RV = A(V)/E(B-V).
"""

import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 5
wns = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 0.3 <= x <= 1.1
wns[0] = (0.3, 1.1)
# a(x)_NIR = 0.574*x**1.61
# b(x)_NIR = -0.527*x**1.61
exps[0] = 1.61
coeffs["A"]["poly"][0] = P([0.574])
coeffs["B"]["poly"][0] = P([-0.527])

# for 1.1 <= x <= 3.3
wns[1] = (1.1, 3.3)
# y = x - 1.82
# a(x)_optical = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
# b(x)_optical = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
a_opt = P([1, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
b_opt = P([0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])
shift = P([-1.82, 1])
coeffs["A"]["poly"][1] = a_opt(shift)
coeffs["B"]["poly"][1] = b_opt(shift)

# for 3.3 <= x <= 5.9
wns[2] = (3.3, 5.9)
# a(x)_UV = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# b(x)_UV = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
coeffs["A"]["poly"][2] = P([1.752, -0.316])
coeffs["A"]["rem"][2] = P([-0.104])
coeffs["A"]["div"][2] = P([0.341, 0, 1])(P([-4.67, 1]))
coeffs["B"]["poly"][2] = P([-3.090, 1.825])
coeffs["B"]["rem"][2] = P([1.206])
coeffs["B"]["div"][2] = P([0.263, 0, 1])(P([-4.62, 1]))

# for 5.9 <= x <= 8.0
wns[3] = (5.9, 8.0)
# a(x)_FUV = a(x)_UV - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# b(x)_FUV = b(x)_UV + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
shift = P([-5.9, 1])
coeffs["A"]["poly"][3] = coeffs["A"]["poly"][2] + P([0, 0, -0.04473, -0.009779])(shift)
coeffs["A"]["rem"][3] = coeffs["A"]["rem"][2]
coeffs["A"]["div"][3] = coeffs["A"]["div"][2]
coeffs["B"]["poly"][3] = coeffs["B"]["poly"][2] + P([0, 0, 0.2130, 0.1207])(shift)
coeffs["B"]["rem"][3] = coeffs["B"]["rem"][2]
coeffs["B"]["div"][3] = coeffs["B"]["div"][2]

# for 8.0 <= x <= 10.0
wns[4] = (8.0, 10.0)
# a(x)_FFUV = -1.073 - 0.628*(x - 8) + 0.137*(x - 8)**2 - 0.070*(x - 8)**3
# b(x)_FFUV = 13.670 + 4.257*(x - 8) - 0.420*(x - 8)**2 + 0.374*(x - 8)**3
shift = P([-8, 1])
a_FFUV = P([-1.073, -0.628, 0.137, -0.070])
b_FFUV = P([13.670, 4.257, -0.420, 0.374])
coeffs["A"]["poly"][4] = a_FFUV(shift)
coeffs["B"]["poly"][4] = b_FFUV(shift)

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [0.3, 10.0]\n")
    f.write(
        "RV_RANGE: [2.0, 6.0]\n"
    )  # from dust_extinction. Sample in paper [2.85, 5.6]
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
