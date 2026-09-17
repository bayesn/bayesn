""" From Calzetti et al. 2000ApJ...533..682C Equation 4.
https://iopscience.iop.org/article/10.1086/308692
This is a fit of the Calzetti et al. 1994 "starburst reddening" curve (see also Calzetti 1997).
Derived from comparisons between stellar energy absorbed by dust (measured in FIR emission)
and a predicted value derived from UV-to-NIR SEDs of 5 galaxies.
Four galaxies have 0.12 - 2.2 um coverage, and the fifth has published 0.12 - 1 um coverage.

The piecewise function is not continuous.
It would be if the 0.63 micron breakpoint was at about 0.7807, 0.7227, or 0.0652 microns.
"""

import numpy as np
from numpy.polynomial import Polynomial as P

# A(\lambda)/A(V) = 1 + k / Rv
# a(x) is always 1, b(x) = k, sp(x) = 0
N = 2
wl_range = np.array([1200, 22000])
wn_range = 1e4/wl_range[::-1]
wns = np.zeros((N, 2))
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# from 0.63 microns to 2.2 microns
# b(x) = 2.659 * (1.040*x - 1.857)
wns[0] = (1 / 2.2, 1 / 0.63)
coeffs["A"]["poly"][0] = P([1])
coeffs["B"]["poly"][0] = 2.659*P([-1.857, 1.040])

# for 0.12 microns to 0.63 microns
# b(x) = 2.659 * (((0.011*x - 0.198)*x + 1.509)*x - 2.156)
wns[1] = (1 / 0.63, 1 / 0.12)
coeffs["A"]["poly"][1] = P([1])
coeffs["B"]["poly"][1] = 2.659*P([-2.156, 1.509, -0.198, 0.011])

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [{', '.join(wn_range.astype(str))}]\n")
    for arr, name in zip(
        (wns, coeffs["A"]["poly"], coeffs["B"]["poly"]),
        (
            "SUBDOMAINS",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            if isinstance(coeffs, P):
                coeffs = coeffs.coef[::-1]
            f.write(f"- [{', '.join(coeffs.astype(str))}]\n")
