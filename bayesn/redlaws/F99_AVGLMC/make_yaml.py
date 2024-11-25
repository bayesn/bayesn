""" From Fitzpatrick 1999PASP..111...63F
https://iopscience.iop.org/article/10.1086/316293

Uses a cubic spline with anchors in the UV, optical, and NIR to match synthetic photometry
of an artificially reddened SED to observations in Johnson and Stromgren filters.
UV portion follows from Fitzpatrick & Massa 1990.
Uses FM90 sample - Orion nebula stars + HD210121 (RV=1/0.45 ~ 2.22 +/- 0.14)

Values taken from FMRCURVE.pro as seen on Nov 2024 at
https://universe.gsfc.nasa.gov/archive/idlastro/ftp/pro/astro/fm_unred.pro
With AVGLMC set and LMC2 unset.

The program states:
Parameterization is valid from the IR to the far-UV (3.5 microns to 0.1
microns).    UV extinction curve is extrapolated down to 912 Angstroms.
"""

import numpy as np
from numpy.polynomial import Polynomial as P

angstrom_knot_locations = np.array(
    [np.inf, 26500, 12200, 6000, 5470, 4670, 4110, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)
max_RV_coeff_len = 0

# Rational function coefficients (Rv^4 ... Rv^-1) for calculating spline values
spline_val_coeffs = [P([0]) for _ in range(N)]
# constant terms from Fitzpatrick 1999PASP..111...63F
C1_1 = -1.28
C1_2 = 0
C2_1 = 1.11
C2_2 = 0
C3 = 2.73
C4 = 0.64
C5 = 5.9
xo = 4.596
gamma = 0.91
FM90_quad = 0.5392
FM90_cubic = 0.05644
# A(lambda=inf) = 0
# yk[0] = -Rv
spline_val_coeffs[0] = P([0, -1])
# NIR
# yk[1] = 0.26469 * RV / 3.1 - RV
spline_val_coeffs[1] = P([0, 0.26469 / 3.1 - 1])
# yk[2] = 0.82925 * RV / 3.1 - RV
spline_val_coeffs[2] = P([0, 0.82925 / 3.1 - 1])
# yk[3] = -0.422809 + 1.00270 * RV + 2.13572e-4 * RV**2 - RV
spline_val_coeffs[3] = P([0, -0.422809, 1.00270 - 1, 2.13572e-4])
# yk[4] = -5.13540e-2 + 1.00216 * RV - 7.35778e-5 * RV**2 - RV
spline_val_coeffs[4] = P([0, -5.13540e-2, 1.00216 - 1, -7.35778e-5])
# yk[5] = 0.700127 + 1.00184 * RV - 3.32598e-5 * RV**2 - RV
spline_val_coeffs[5] = P([0, 0.700127, 1.00184 - 1, -3.32598e-5])
# yk[6] = 1.19456 + 1.01707 * RV - 5.46959e-3 * RV**2 + 7.97809e-4 * RV**3 - 4.45636e-5 * RV**4 - RV
spline_val_coeffs[6] = P(
    [0, 1.19456, 1.01707 - 1, -5.46959e-3, 7.97809e-4, -4.45636e-5]
)

# UV
# yk[7 or 8] = C1 + C2 * xk[7 or 8] + C3 * D where
# C2 = -0.824 + 4.717 / Rv
# original F99 C1-C2 correlation
# C1 = 2.030 - 3.007 * C2
# which is the just a constant term and a term that scales with 1/RV
for i in (7, 8):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )  # RV independent
    spline_val_coeffs[i] = P(
        [C2_2 * (C1_2 + xk[i]), C1_1 + C1_2 * C2_1 + C2_1 * xk[i] + C3 * D]
    )
    if xk[i] >= C5:
        spline_val_coeffs[i] += P(
            [0, C4 * (FM90_quad * (xk[i] - C5) ** 2 + FM90_cubic * (xk[i] - C5) ** 3)]
        )

for i in range(len(spline_val_coeffs)):
    max_RV_coeff_len = max(max_RV_coeff_len, len(spline_val_coeffs[i].coef))

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"NUM_KNOTS: {N}\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [0.286, 10.964]\n")
    f.write(
        f"RV_RANGE: [2.0, 6.0]\n"
    )  # from dust_extinction. Sample in paper [2.22, 5.83]
    f.write(f"MIN_ORDER: -1\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        if isinstance(coeffs, P):
            coeffs = coeffs.coef[::-1]
        diff = max_RV_coeff_len - len(coeffs)
        coeffs = np.append([0] * diff, coeffs)
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
