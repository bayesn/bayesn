import numpy as np

angstrom_knot_locations = np.array(
    [np.inf, 26500, 12200, 6000, 5470, 4670, 4110, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)

# Rational function coefficients (Rv^4 ... Rv^-1) for calculating spline values
spline_val_coeffs = np.zeros((N, 6))
# constant terms from Fitzpatrick 1999PASP..111...63F
C1_1 = 2.030
C1_2 = -3.007
C2_1 = -0.824
C2_2 = 4.717
C3 = 3.23
C4 = 0.41
C5 = 5.9
xo = 4.596
gamma = 0.99


# values from https://universe.gsfc.nasa.gov/archive/idlastro/ftp/pro/astro/fm_unred.pro
# A(lambda=inf) = 0
# yk[0] = -Rv
spline_val_coeffs[0, 3] = -1
# NIR
# yk[1] = 0.26469 * RV / 3.1 - RV
spline_val_coeffs[1, 3] = 0.26469 / 3.1 - 1
# yk[2] = 0.82925 * RV / 3.1 - RV
spline_val_coeffs[2, 3] = 0.82925 / 3.1 - 1
# yk[3] = -0.422809 + 1.00270 * RV + 2.13572e-4 * RV**2 - RV
spline_val_coeffs[3] = [0, 0, 2.13572e-4, 1.00270 - 1, -0.422809, 0]
# yk[4] = -5.13540e-2 + 1.00216 * RV - 7.35778e-5 * RV**2 - RV
spline_val_coeffs[4] = [0, 0, -7.35778e-5, 1.00216 - 1, -5.13540e-2, 0]
# yk[5] = 0.700127 + 1.00184 * RV - 3.32598e-5 * RV**2 - RV
spline_val_coeffs[5] = [0, 0, -3.32598e-5, 1.00184 - 1, 0.700127, 0]
# yk[6] = 1.19456 + 1.01707 * RV - 5.46959e-3 * RV**2 + 7.97809e-4 * RV**3 - 4.45636e-5 * RV**4 - RV
spline_val_coeffs[6] = [-4.45636e-5, 7.97809e-4, -5.46959e-3, 1.01707 - 1, 1.19456, 0]

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
    spline_val_coeffs[i, -2] = C1_1 + C1_2 * C2_1 + C2_1 * xk[i] + C3 * D
    spline_val_coeffs[i, -1] = C2_2 * (C1_2 + xk[i])
    if xk[i] >= C5:
        spline_val_coeffs[i, -2] += C4 * (
            0.5392 * (xk[i] - C5) ** 2 + 0.05644 * (xk[i] - C5) ** 3
        )

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in inv_micron_knot_locations)}]\n")
    f.write(f"NUM_KNOTS: {N}\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"MIN_ORDER: -1\n")
    f.write("RV_COEFFS:\n")
    for coeffs in spline_val_coeffs:
        f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")