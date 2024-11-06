import numpy as np
from numpy.polynomial import Polynomial as P

angstrom_knot_locations = np.array(
    [np.inf, 2e4, 4e4 / 3, 1e4, 6000, 5470, 4670, 4110, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)

# Rational function coefficients (Rv^2 ... Rv^-1) for calculating spline values
spline_val_coeffs = [P([0]) for _ in range(N)]
# constant terms from F99, F04, and FM07
C1_1 = 2.18  # from F04
C1_2 = -2.91
C2_1 = -0.824  # from F99
C2_2 = 4.717
C3 = 2.991  # From FM07
C4 = 0.319
C5 = 5.9
xo = 4.592
gamma = 0.922

FM90_quad = 0.5392
FM90_cubic = 0.05644

# From F99 paper, values do not match FMRCURVE.pro
for i in range(4):
    spline_val_coeffs[i] = P([0, -0.84 * xk[i] ** 1.84, 0.63 * xk[i] ** 1.84 - 1])
spline_val_coeffs[4] = P([0, -0.426, 0.0044])
spline_val_coeffs[5] = P([0, -0.050, 0.0016])
spline_val_coeffs[6] = P([0, 0.701, 0.0016])
spline_val_coeffs[7] = P([0, 1.208, 0.0032, -0.00033])

# yk[7 or 8] = C1 + C2 * xk[7 or 8] + C3 * D where
for i in (8, 9):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )  # RV independent
    spline_val_coeffs[i] = P([C2_2 * (C1_2 + xk[i]), C1_1 + C1_2 * C2_1 + C2_1 * xk[i] + C3 * D])
    if xk[i] >= C5:
        spline_val_coeffs[i] += P([0, C4 * (FM90_quad * (xk[i] - C5) ** 2 + FM90_cubic * (xk[i] - C5) ** 3)])

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"NUM_KNOTS: {N}\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"MIN_ORDER: -1\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        if isinstance(coeffs, P):
            coeffs = coeffs.coef[::-1]
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
