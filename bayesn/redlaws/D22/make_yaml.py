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
A_poly_coeffs = [[0.377, 0]]
exps = [1.78]
# with knots at d22['wl'] and values at d22['slope']
# A(x)/A(V) = a + b(3.1-Rv)/3.1/Rv
# where b is the spline interpolation.
# Instead, knots values should be at (3.1 - Rv)/3.1 * d22['slope']
# which preserves A(x)/A(V) = a + b/Rv
with open("BAYESN.YAML", "w") as f:
    f.write("TYPE: SPLINE\n")
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in d22['wl'].values)}]\n")
    f.write(f"NUM_KNOTS: {len(d22['wl'])}\n")
    f.write(f"KNOT_UNITS: microns\n")
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
        ("REGIME_WLS", "A_POLY_COEFFS"),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
