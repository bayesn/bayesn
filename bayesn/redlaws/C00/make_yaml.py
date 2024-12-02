""" From Calzetti et al. 2000ApJ...533..682C
https://iopscience.iop.org/article/10.1086/308692
This is a fit of the Calzetti et al. 1994 "starburst reddening" curve (see also Calzetti 1997).
Derived from comparisons between stellar energy absorbed by dust (measured in FIR emission)
and a predicted value derived from UV-to-NIR SEDs of 5 galaxies.
Four galaxies have 0.12 - 2.2 um coverage, and the fifth has published 0.12 - 1 um coverage.

Implementation based on extinction
https://extinction.readthedocs.io/en/latest/api/extinction.calzetti00.html
"""

import numpy as np

# A(lambda)/A(V) = 1 + k / Rv
# a(x) is always 1, b(x) = k
N = 2
wns = np.zeros((N, 2))
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 1/2.2 <= x <= 1/0.63
# b(x) = 2.659 * (1.040*x - 1.857)
wns[0] = (1 / 2.2, 1 / 0.63)
coeffs["A"]["poly"][0] = [1]
coeffs["B"]["poly"][0] = [2.569 * 1.040, 2.659 * (-1.857)]

# for 1/0.63 < x <= 1/0.12
# b(x) = 2.659 * (((0.011*x - 0.198)*x + 1.509)*x - 2.156)
wns[1] = [1 / 0.63, 1 / 0.12]
coeffs["A"]["poly"][1] = [1]
coeffs["B"]["poly"][1] = [0.029249, -0.526482, 4.012431, -5.732804]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [0.455, 8.33]\n")
    for arr, name in zip(
        (wns, coeffs["A"]["poly"], coeffs["B"]["poly"]),
        (
            "REGIMES",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
