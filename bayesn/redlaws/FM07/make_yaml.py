import numpy as np

angstrom_knot_locations = np.array(
    [np.inf, 4e4, 2e4, 4e4 / 3, 1e4, 5530, 4000, 3300, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)

# Linear function of Rv calculating spline values
spline_val_coeffs = np.zeros((N, 2))
# constant terms
C1 = -0.175
C2 = 0.807
C3 = 2.991
C4 = 0.319
C5 = 6.097
xo = 4.592
gamma = 0.922


# yk[0:5] = (-0.83 + 0.63*Rv) * xk[0:5]**1.84 - Rv
for i in range(0, 5):
    spline_val_coeffs[i, 0] = 0.63 * xk[i] ** 1.84 - 1
    spline_val_coeffs[i, 1] = -0.83 * xk[i] ** 1.84
# yk[5:8] are constants, with yk[5] = 0
spline_val_coeffs[6, 1] = 1.322
spline_val_coeffs[7, 1] = 2.055
# yk[8:10] = C1 + C2 * xk[8:10] + C3 * D
for i in range(8, 10):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )  # RV independent
    spline_val_coeffs[i, 1] = C1 + C2 * xk[i] + C3 * D
    if xk[i] >= C5:
        spline_val_coeffs[i, 1] += C4 * (xk[i] - C5) ** 2

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"NUM_KNOTS: {N}\n")
    f.write(f"KNOT_UNITS: inverse microns\n")
    f.write(f"MIN_ORDER: 0\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
