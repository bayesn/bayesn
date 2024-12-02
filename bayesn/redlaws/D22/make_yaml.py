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
exps = [-1.78]
# with knots at d22['wl'] and values at d22['slope']
# A(x)/A(V) = a + b(3.1-Rv)/3.1/Rv
# where b is the spline interpolation.
# Instead, knots values should be at (3.1 - Rv)/3.1 * d22['slope']
# which preserves A(x)/A(V) = a + b/Rv
with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in d22['wl'].values)}]\n")
    f.write(f"UNITS: microns\n")
    f.write(f"WAVE_RANGE: [0.8, 5.0]\n")
    f.write(
        "RV_RANGE: [2.5, 5.5]\n"
    )  # from dust_extinction. Sample in paper [2.43, 5.33]
    f.write(f"MIN_ORDER: 0\n")
    f.write(f"REGIME_EXP: [{', '.join([str(x) for x in exps])}]\n")
    f.write("RV_COEFFS:\n")
    for lin, const in zip(
        -d22["slope"].values / 3.1,
        d22["slope"].values,
    ):
        f.write(f"- [{lin}, {const}]\n")
    for arr, name in zip(
        (wls, A_poly_coeffs),
        ("REGIMES", "A_POLY_COEFFS"),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
