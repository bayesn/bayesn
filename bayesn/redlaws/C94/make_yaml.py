""" From Calzetti et al. 1994ApJ...429..582C
https://ui.adsabs.harvard.edu/abs/1994ApJ...429..582C/abstract
The starburst extinction model is derived using UV and optical spectra of 39 galaxies.
The model is parameterized with Balmer optical depth and the effective extiction is
Q(x) = -2.156 + 1.509*x - 0.198 x**2 + 0.011*x**3
where x is in inverse microns.
It is related to other extinction models (k(x) = A(x)/E(B-V)) as
Q(x) = k(x)/(k(H_beta) - k(H_alpha))
where the quotient is roughly 1.16 (Seaton 1979)
For reference, in G23 the same quantity is 1.166.

Noticeably, this dust model does not include a UV feature at 2175 angstroms.
"""

import numpy as np

N = 1
wns = np.zeros((N, 2))
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 1/0.8 <= x < 1/0.125
wns[0] = (1 / 0.8, 1 / 0.125)
coeffs["A"]["poly"][0] = [0.011, -0.198, 1.509, -2.156]
coeffs["A"]["rem"][0] = [1.16]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write("WAVE_RANGE: [1.25, 8]\n")
    for arr, name in zip(
        (wns, coeffs["A"]["poly"], coeffs["A"]["rem"]),
        (
            "REGIMES",
            "A_POLY_COEFFS",
            "B_REMAINDER_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
