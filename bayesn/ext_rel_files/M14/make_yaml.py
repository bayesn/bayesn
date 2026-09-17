""" From Maíz Apellániz et al. 2014
https://www.aanda.org/articles/aa/full_html/2014/04/aa23439-14/aa23439-14.html
Code adapted from IDL code in Table 1.
"""

import numpy as np
from numpy.polynomial import Polynomial as P

n_subdomains = 6
wns = np.zeros((n_subdomains, 2))
wn_range = np.array([0.3, 10])
AB_exps = np.zeros(n_subdomains)
RV_exps = np.zeros(n_subdomains)

coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(n_subdomains)]
coeffs["sp"] = [[P([0]) for _ in range(11)] for _ in range(n_subdomains)]
idx = 0

# IR is given as 0.574*x**1.61 - 0.527*x**1.61 / RV
wns[idx] = np.array([0.3, 1])
coeffs["A"]["poly"][idx] = P([0.574])
coeffs["B"]["poly"][idx] = P([-0.527])
AB_exps[0] = 1.61
idx += 1

# optical is given as a spline with knots at xk with values given as rational
# functions of x. The rational function varies between three sets of knots (x{123}).
# Although this subdomain produces functions matching a(x) + b(x)/R_V, the spline interp
# means it has to be done through sp(x) using RV**{-1} and RV**0. Furthermore, the
# boundary conditions are provided as distinct first derivatives, so need to be
# provided with two separate subdomains. The spline for a(x) will be provided as
# the second subdomain (idx=1) and b(x) with the third (idx=2).
wns[idx] = wns[idx+1] = np.array([1, 4.2])
RV_exps[idx+1] = -1
x1 = np.array([1.0])
x2 = np.array([1.15, 1.81984, 2.1, 2.27015, 2.7])
x3 = np.array([3.5, 3.9, 4.0, 4.1, 4.2])
xk = np.concatenate([x1, x2, x3])
a_extra = [0, 0, 0, -0.011, 0, 0, 0.442, 0.341, 0.130, 0.020, 0]
b_extra = [0, 0, 0,  0.091, 0, 0, 1.256, 1.021, 0.416, 0.064, 0]
L_knots = np.zeros((n_subdomains, len(xk)))
L_knots[1] = L_knots[2] = xk
a_bcs, b_bcs = [[[1, 0], [1, 0]] for _ in range(2)]
# first set of knots
coeffs["sp"][idx][0]   = P([ 0.574*x1[0]**1.61])  # a1v
coeffs["sp"][idx+1][0] = P([-0.527*x1[0]**1.61])  # b1v
a_bcs[0][1] =  0.574*1.61*x1[0]**0.61
b_bcs[0][1] = -0.527*1.61*x1[0]**0.61
# second set of knots is messy
shift_2v = P([-1.82, 1])
for i in range(len(x1), len(x1)+len(x2)):
    a2v_poly = P([1+a_extra[i], 0.17699, -0.50447, -0.02427,  0.72085,  0.01979, -0.77530,  0.32999])
    b2v_poly = P([0+b_extra[i], 1.41338,  2.28305,  1.07233, -5.38434, -0.62251,  5.30260, -2.09002])
    coeffs["sp"][idx][i]   = P([a2v_poly(shift_2v(xk[i]))])
    coeffs["sp"][idx+1][i] = P([b2v_poly(shift_2v(xk[i]))])
# third set of knots
a_dx3 = 4.67
b_dx3 = 4.62
for i in range(len(x1)+len(x2), len(xk)):
    coeffs["sp"][idx][i]   = P([ 1.752 - 0.316*xk[i] - 0.104/((xk[i] - a_dx3)**2 + 0.341) + a_extra[i]])  # a3v
    coeffs["sp"][idx+1][i] = P([-3.090 + 1.825*xk[i] + 1.206/((xk[i] - b_dx3)**2 + 0.263) - b_extra[i]])  # b3v
a_bcs[1][1] = -0.316 + 0.104*2*(x3[-1]-4.67)/((x3[-1] - a_dx3)**2 + 0.341)**2
b_bcs[1][1] =  1.825 - 1.206*2*(x3[-1]-4.62)/((x3[-1] - b_dx3)**2 + 0.263)**2
idx += 2

# The UV subdomain is actually split into two subdomains, NUV and UV, the latter having
# an extra polynomial component.
wns[idx] = np.array([4.2, 5.9])
for sub_idx in range(idx, idx+2):
    coeffs["A"]["poly"][sub_idx] = P([1.752, -0.316])
    coeffs["A"]["rem"][sub_idx]  = P([-0.104])
    coeffs["A"]["div"][sub_idx]  = P([0.341, 0, 1])(P([-a_dx3, 1]))
    coeffs["B"]["poly"][sub_idx] = P([-3.090, 1.825])
    coeffs["B"]["rem"][sub_idx]  = P([1.206])
    coeffs["B"]["div"][sub_idx]  = P([0.263, 0, 1])(P([-b_dx3, 1]))
idx += 1
# UV
wns[idx] = np.array([5.9, 8])
shift_uv = P([-5.9, 1])
coeffs["A"]["poly"][4] += P([0, 0, -0.04473, -0.009779])(shift_uv)  # fa
coeffs["B"]["poly"][4] += P([0, 0,  0.2130,   0.1207  ])(shift_uv)  # fb
idx += 1
# FUV
wns[idx] = np.array([8, 10])
shift_fuv = P([-8, 1])
coeffs["A"]["poly"][5] = P([-1.073, -0.628,  0.137, -0.070])(shift_fuv)  # au
coeffs["B"]["poly"][5] = P([13.670,  4.257, -0.420,  0.374])(shift_fuv)  # bu

max_RV_coeff_len = 0
for subdomain_coeffs in coeffs["sp"]:
    for sp_coeffs in subdomain_coeffs:
        max_RV_coeff_len = max(max_RV_coeff_len, len(sp_coeffs.coef))


with open("BAYESN.YAML", "w") as f:
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [{', '.join(wn_range.astype(str))}]\n")
    f.write(
        f"RV_RANGE: [2.0, 6.0]\n"
    )  # from dust_extinction. Sample in paper [2.22, 5.83]
    f.write(f"L_KNOTS:\n")
    for knots in L_knots:
        f.write(f"- [{', '.join(str(x) for x in knots)}]\n")
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
    f.write(f"SPLINE_BC_TYPE: [None, {a_bcs}, {b_bcs}, None, None, None]\n")
    f.write(f"A_EXP: [{', '.join(str(exp) for exp in AB_exps)}]\n")
    f.write(f"B_EXP: [{', '.join(str(exp) for exp in AB_exps)}]\n")
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
