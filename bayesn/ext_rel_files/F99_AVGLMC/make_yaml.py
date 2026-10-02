""" From Fitzpatrick 1999PASP..111...63F
https://iopscience.iop.org/article/10.1086/316293

Uses a cubic spline with anchors in the UV, optical, and NIR to match synthetic photometry
of an artificially reddened SED to observations in Johnson and Stromgren filters.
UV portion follows from Fitzpatrick & Massa 1990.
Uses FM90 sample - Orion nebula stars + HD210121 (RV=1/0.45 ~ 2.22 +/- 0.14)

Values taken from FMRCURVE.pro as seen on Nov 2024 at
https://universe.gsfc.nasa.gov/archive/idlastro/ftp/pro/astro/fm_unred.pro
With AVGLMC and LMC2 both unset.

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

wave_range = np.array([912, 35000])
wn_range = 1e4/wave_range[::-1]
dummy_knots = np.linspace(*wn_range, len(xk))
L_knots = [xk, dummy_knots, dummy_knots]
# F99 is given as a spline in the NUV/optical/ir with set knot x values and
# y values given as polynomials of RV. The knot at 2700 A (xk[7]) is the transition,
# between NUV/opticall/IR and UV, but the spline also uses the knot at 2600 A.
n_subdomains = 3
wns = np.array([[wn_range[0], xk[7]], [xk[7], C5], [C5, wn_range[1]]])

coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(n_subdomains)]
coeffs["sp"] = [[P([0]) for _ in range(N)] for _ in range(n_subdomains)]

# NUV/Optical/IR is given as a spline with knots at set x values and y values given
# by a polynomial in RV. The knots in the UV include 1/RV scaling, so RV_exps = -1.
# in this subdomain.
RV_exps = np.array([-1, 0, 0])
# yk[0] = -Rv
coeffs["sp"][0][0] = P([0, 0, 0])
# NIR
# yk[1] = 0.26469 * RV / 3.1
coeffs["sp"][0][1] = P([0, 0, 0.26469 / 3.1])
# yk[2] = 0.82925 * RV / 3.1
coeffs["sp"][0][2] = P([0, 0, 0.82925 / 3.1])
# yk[3] = -0.422809 + 1.00270 * RV + 2.13572e-4 * RV**2
coeffs["sp"][0][3] = P([0, -0.422809, 1.00270, 2.13572e-4])
# yk[4] = -5.13540e-2 + 1.00216 * RV - 7.35778e-5 * RV**2
coeffs["sp"][0][4] = P([0, -5.13540e-2, 1.00216, -7.35778e-5])
# yk[5] = 0.700127 + 1.00184 * RV - 3.32598e-5 * RV**2
coeffs["sp"][0][5] = P([0, 0.700127, 1.00184, -3.32598e-5])
# yk[6] = 1.19456 + 1.01707 * RV - 5.46959e-3 * RV**2 + 7.97809e-4 * RV**3 - 4.45636e-5 * RV**4
coeffs["sp"][0][6] = P([0, 1.19456, 1.01707, -5.46959e-3, 7.97809e-4, -4.45636e-5])
# yk[7 or 8] = C1 + C2 * xk[7 or 8] + C3 * D + RV where
# C2 = C2_1 + C2_2/RV = -0.824 + 4.717 / RV
# C1 = C1_1 + C1_2*C2 = 2.030 - 3.007*C2
# D = C3*x**2/((x**2-x0**2)**2 + (x*gamma)**2)
# In local variables and with x = xk[7 or 8]
# yk[7 or 8] = C1_1 + C1_2*C2_1 + C1_2*C2_2/RV + C2_1*x + C2_2*x/RV + RV
for i in (7, 8):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )
    coeffs["sp"][0][i] = P(
        [C2_2 * (C1_2 + xk[i]), C1_1 + C1_2 * C2_1 + C2_1 * xk[i] + C3 * D, 1]
    )

# The UV and FUV subdomains have the same formulation as yk[7 or 8], but scales with x so
# is included through AB coeffs. The term scaling with 1/RV is accounted for
# with B coeffs. The +RV at the end is accounted for with sp(x).
for i in (1, 2):
    coeffs["A"]["poly"][i] = P([C1_1 + C1_2*C2_1, C2_1])
    coeffs["B"]["poly"][i] = P([C1_2*C2_2, C2_2])
    coeffs["A"]["rem"][i] = P([0, 0, C3])
    coeffs["A"]["div"][i] = P([xo**4, 0, gamma**2 - 2*xo**2, 0, 1])
    # RV_exps[1 or 2] = 0 instead of -1, so P([0, 1]) instead of P([0, 0, 1])
    for j in range(len(dummy_knots)):
        coeffs["sp"][i][j] = P([0, 1])

# The FUV region has an additional C4 term.
shift = P([-5.9, 1])
coeffs["A"]["poly"][2] += P([0, 0, C4*FM90_quad, C4*FM90_cubic])(shift)

for subdomain_coeffs in coeffs["sp"]:
    for sp_coeffs in subdomain_coeffs:
        max_RV_coeff_len = max(max_RV_coeff_len, len(sp_coeffs.coef))

with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS:\n")
    for knots in L_knots:
        f.write(f"- [{', '.join(str(x) for x in knots)}]\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [{', '.join(wn_range.astype(str))}]\n")
    f.write(
        f"RV_RANGE: [2.0, 6.0]\n"
    )  # from dust_extinction. Sample in paper [2.22, 5.83]
    f.write("OUTPUT_TYPE: axebv\n")
    f.write("SPLINE_BC_TYPE: [natural, None, None]\n")
    f.write(f"RV_EXP: [{', '.join(RV_exps.astype(str))}]\n")
    f.write("RV_COEFFS:\n")
    for subdomain_coeffs in coeffs["sp"]:
        nested_list_start = True
        for sp_coeffs in subdomain_coeffs:
            if isinstance(sp_coeffs, P):
                sp_coeffs = sp_coeffs.coef[::-1]
            diff = max_RV_coeff_len - len(sp_coeffs)
            sp_coeffs = np.append(np.zeros(diff), sp_coeffs)
            if nested_list_start:
                f.write("- ")
            else:
                f.write("  ")
            f.write(f"- [{', '.join(sp_coeffs.astype(str))}]\n")
            nested_list_start = False
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
            "SUBDOMAINS",
            *np.array([[f"{var}_{term}_COEFFS" for var in "AB"] for term in ("POLY", "REMAINDER", "DIVISOR")]).flatten(),
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            if isinstance(coeffs, P):
                coeffs = coeffs.coef[::-1]
            f.write(f"- [{', '.join(coeffs.astype(str))}]\n")
