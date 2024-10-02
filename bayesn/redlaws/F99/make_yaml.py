import numpy as np

angstrom_knot_locations = np.array(
    [np.inf, 26500, 12200, 6000, 5470, 4670, 4110, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)

# Rational function coefficients (Rv^4 ... Rv^-1) for calculating spline values
spline_val_coeffs = np.zeros((N, 6))
# constant terms
C3 = 3.23
C4 = 0.41
C5 = 5.9
xo = 4.596
gamma = 0.99


# yk[0] = -Rv
spline_val_coeffs[0, 3] = -1
# yk[1] = 0.26469 * RV / 3.1 - RV
spline_val_coeffs[1, 3] = 0.26469 / 3.1 - 1
# yk[2] = 0.82925 * RV / 3.1 - RV
spline_val_coeffs[2, 3] = 0.82925 / 3.1 - 1
# yk[3] = -0.422809 + 1.00270 * RV + 2.13572e-4 * RV**2 - RV
spline_val_coeffs[3, 4] = -0.422809
spline_val_coeffs[3, 3] = 1.00270 - 1
spline_val_coeffs[3, 2] = 2.13572e-4
# yk[4] = -5.13540e-2 + 1.00216 * RV - 7.35778e-5 * RV**2 - RV
spline_val_coeffs[4, 4] = -5.13540e-2
spline_val_coeffs[4, 3] = 1.00216 - 1
spline_val_coeffs[4, 2] = -7.35778e-5
# yk[5] = 0.700127 + 1.00184 * RV - 3.32598e-5 * RV**2 - RV
spline_val_coeffs[5, 4] = 0.700127
spline_val_coeffs[5, 3] = 1.00184 - 1
spline_val_coeffs[5, 2] = -3.32598e-5
# yk[6] = 1.19456 + 1.01707 * RV - 5.46959e-3 * RV**2 + 7.97809e-4 * RV**3 - 4.45636e-5 * RV**4 - RV
spline_val_coeffs[6, 4] = 1.19456
spline_val_coeffs[6, 3] = 1.01707 - 1
spline_val_coeffs[6, 2] = -5.46959e-3
spline_val_coeffs[6, 1] = 7.97809e-4
spline_val_coeffs[6, 0] = -4.45636e-5

# yk[7,8] = C1 + C2 * xk[7,8] + C3 * D where
# C2 = -0.824 + 4.717 / Rv
# original F99 C1-C2 correlation
# C1 = 2.030 - 3.007 * C2
# which is the just a constant term and a term that scales with 1/RV
for i in (7, 8):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )  # RV independent
    spline_val_coeffs[i, 4] = 2.030 + 3.007 * 0.824 - 0.824 * xk[i] + C3 * D
    spline_val_coeffs[i, 5] = -3.007 * 4.717 + 4.717 * xk[i]

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"NUM_KNOTS: {N}\n")
    f.write(f"MIN_ORDER: -1\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
