""" From Fitzpatrick & Massa 2007ApJ...663..320F
https://iopscience.iop.org/article/10.1086/518158
IR-through-UV extinction of normal, near main-sequence Galactic OB stars.
RV range = [2.33 (HD220057 (NGC 7654)), 6.42 (HD37022 (NGC 1977))]

The only RV sensitive knots are the first 5, which are >= 1 um.
For field stars without IR photometry, the authors assumed an RV of 3.1 and a
scale parameter k_IR of 1.11 based on k_IR = 0.63RV - 0.84 from F04.
This closely matches the 0.63RV - 0.83 found in this sample.
For cluster stars without IR photometry, the authors assumes the cluster averages.

Implementation based on extinction
https://extinction.readthedocs.io/en/latest/api/extinction.fm07.html
Constants from extinction
"""

import numpy as np
from numpy.polynomial import Polynomial as P

angstrom_knot_locations = np.array(
    [np.inf, 4e4, 2e4, 4e4 / 3, 1e4, 5530, 4000, 3300, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)
max_RV_coeff_len = 0

# Linear function of Rv calculating spline values
spline_val_coeffs = [P([0]) for _ in range(N)]
# constant terms
C1 = -0.175
C2 = 0.807
C3 = 2.991
C4 = 0.319
C5 = 6.097
xo = 4.592
gamma = 0.922


# yk[0:5] = (-0.83 + 0.63*Rv) * xk[0:5]**1.84 - Rv
#         = -0.83 * xk[0:5]**1.84 + Rv * (0.63 * xk[0:5]**1.84 - 1)
for i in range(0, 5):
    spline_val_coeffs[i] = P([-0.83 * xk[i] ** 1.84, 0.63 * xk[i] ** 1.84 - 1])
# yk[5:8] are constants, with yk[5] = 0
spline_val_coeffs[6] = P([1.322])
spline_val_coeffs[7] = P([2.055])
# yk[8:10] = C1 + C2 * xk[8:10] + C3 * D
for i in range(8, 10):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )  # RV independent
    spline_val_coeffs[i] = P([C1 + C2 * xk[i] + C3 * D])
    if xk[i] >= C5:
        spline_val_coeffs[i] += P([C4 * (xk[i] - C5) ** 2])

for i in range(len(spline_val_coeffs)):
    max_RV_coeff_len = max(max_RV_coeff_len, len(spline_val_coeffs[i].coef))

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [0.167, 10.989]\n")
    f.write(f"RV_RANGE: [2.33, 6.42]\n")
    f.write(f"MIN_ORDER: 0\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        if isinstance(coeffs, P):
            coeffs = coeffs.coef[::-1]
        diff = max_RV_coeff_len - len(coeffs)
        coeffs = np.append([0]*diff, coeffs)
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
