""" From Fitzpatrick 2004ASPC..309...33F
https://aspbooks.org/custom/publications/paper/309-0033.html
Similar to F99, but with some updated values (C1, C3, C4, xo, gamma)
and an RV dependent NIR power law.

Implementation based on dust_extinction
https://dust-extinction.readthedocs.io/en/stable/api/dust_extinction.parameter_averages.F04.html#dust_extinction.parameter_averages.F04
"""

import numpy as np
from numpy.polynomial import Polynomial as P

angstrom_knot_locations = np.array(
    [np.inf, 2e4, 4e4 / 3, 1e4, 6000, 5470, 4670, 4110, 2700, 2600]
)
xk = inv_micron_knot_locations = 1e4 / angstrom_knot_locations
N = len(xk)
max_RV_coeff_len = 0

# Rational function coefficients (Rv^2 ... Rv^-1) for calculating spline values
spline_val_coeffs = [P([0]) for _ in range(N)]
# constant terms from F99, F04, and FM07
C1_1 = 2.18  # from F04
C1_2 = -2.91
C2_1 = -0.824  # from F99
C2_2 = 4.717
C3 = 2.991  # from FM07
C4 = 0.319
C5 = 5.9  # from F99
xo = 4.592
gamma = 0.922
FM90_quad = 0.5392
FM90_cubic = 0.05644

wave_range = np.array([912, 35000])
wn_range = 1e4/wave_range[::-1]
dummy_knots = np.linspace(*wn_range, len(xk))
L_knots = [xk, dummy_knots, dummy_knots]
# F04 is like F99, which is a spline in the NUV/optical/ir with set knot x values and
# y values given as polynomials of RV. The knot at 2700 A (xk[8]) is the transition,
# between NUV/opticall/IR and UV, but the spline also uses the knot at 2600 A.
n_subdomains = 3
wns = np.array([[wn_range[0], xk[8]], [xk[8], C5], [C5, wn_range[1]]])

coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(n_subdomains)]
coeffs["sp"] = [[P([0]) for _ in range(N)] for _ in range(n_subdomains)]

RV_exps = np.array([-1, 0, 0])
# Updated F04 NIR curve
for i in range(4):
    coeffs["sp"][0][i] = P([0, -0.84 * xk[i] ** 1.84, 0.63 * xk[i] ** 1.84])
coeffs["sp"][0][4] = P([0, -0.426, 1.0044])
coeffs["sp"][0][5] = P([0, -0.050, 1.0016])
coeffs["sp"][0][6] = P([0, 0.701, 1.0016])
coeffs["sp"][0][7] = P([0, 1.208, 1.0032, -0.00033])
# yk[8 or 9] = C1 + C2 * xk[8 or 9] + C3 * D + RV where
# C2 = C2_1 + C2_2/RV = -0.824 + 4.717 / RV
# C1 = C1_1 + C1_2*C2 = 2.030 - 3.007*C2
# D = C3*x**2/((x**2-x0**2)**2 + (x*gamma)**2)
# In local variables and with x = xk[8 or 9]
# yk[8 or 9] = C1_1 + C1_2*C2_1 + C1_2*C2_2/RV + C2_1*x + C2_2*x/RV + RV
for i in (8, 9):
    D = xk[i] ** 2 / (
        (xk[i] ** 2 - xo**2) ** 2 + (gamma * xk[i]) ** 2
    )
    coeffs["sp"][0][i] = P(
        [C2_2 * (C1_2 + xk[i]), C1_1 + C1_2 * C2_1 + C2_1 * xk[i] + C3 * D, 1]
    )
    if xk[i] >= C5:
        coeffs["sp"][0][i] += P(
            [0, C4 * (FM90_quad * (xk[i] - C5) ** 2 + FM90_cubic * (xk[i] - C5) ** 3)]
        )

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
            f.write(f"- [{', '.join(str(x) for x in sp_coeffs)}]\n")
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
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
