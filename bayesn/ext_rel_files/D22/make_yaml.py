""" From Decleir et al. 2022ApJ...930...15D
https://iopscience.iop.org/article/10.3847/1538-4357/ac5dbe
Compares NIR (0.8 - 5.5 um) spectra of 15 comparison and 25 reddened MW OB stars.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from astropy.modeling.models import PowerLaw1D

d22 = pd.read_csv(
    "D22_Rv_slope.dat",
    delimiter="\s+",
    header=None,
    skiprows=1,
    names=["wl", "slope", "std"],
)
wls = [[0.8, 4]]
A_poly_coeffs = [[0.377]]
B_poly_coeffs = [[0.]]
A_exps = [-1.78]
# A(\lambda)/A(V) = 0.377*\lambda**1.78 + sp(\lambda)(1/Rv-1/3.1)
# where sp(\lambda) is the spline interpolation of knots at d22['wl'] and values at
# d22['slope']
with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS:\n- [{', '.join(str(x) for x in d22['wl'].values)}]\n")
    f.write(f"UNITS: microns\n")
    f.write(f"WAVE_RANGE: [0.8, 4.0]\n")
    f.write("SPLINE_BC_TYPE: [not-a-knot]\n")
    f.write(
        "RV_RANGE: [2.5, 5.5]\n"
    )  # from dust_extinction. Sample in paper [2.43, 5.33]
    f.write(f"RV_EXP: [-1]\n")
    f.write("RV_COEFFS:\n")
    first = True
    for const, inv in zip(
        -d22["slope"].values / 3.1,
        d22["slope"].values,
    ):
        f.write(f"{'-' if first else ' '} - [{const}, {inv}]\n")
        first = False
    f.write(f"A_EXP: [{', '.join([str(x) for x in A_exps])}]\n")
    for arr, name in zip(
        (wls, A_poly_coeffs, B_poly_coeffs),
        ("SUBDOMAINS", "A_POLY_COEFFS", "B_POLY_COEFFS"),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
