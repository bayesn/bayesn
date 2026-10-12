""" From Fitzpatrick 2019ApJ...886..108F
https://iopscience.iop.org/article/10.3847/1538-4357/ab4c3a

Combines new HST/STIS optical spectrophotometry with existing IUE UV spectrophotometry
and 2MASS NIR photometry to make 72 extinction curves with gapless coverage from NIR through UV.

There is some scatter around the R-dependent curves beyond what individual
uncertainties would suggest, which could imply a mode of variation beyond R_V.


The provided file gives k(\lambda - 55)_0 as k_3.02 and s(\lambda - 55) as deltak
which can be used to infer k(\lambda - 55) for any R(V)
k(\lambda - 55) = k_3.02 + deltak [R(V) - 3.10] \alpha
To normalize against A(V) rather than A(550 nm), use
k(\lambda - 55) = \alpha k(\lambda - V) + \beta
A(\lambda)/A(V) = k(\lambda - V)/R_V + 1
"""
import pandas as pd
import numpy as np

f19 = pd.read_csv("F19_tabulated.dat", delimiter='\s+')
alpha = 0.990
beta = 0.049
AV_norm = False
with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS:\n- [{', '.join(str(x) for x in f19['x'].values)}]\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [0.3, 8.7]\n")
    f.write("OUTPUT_TYPE: exvebv\n")
    f.write("SPLINE_BC_TYPE: [not-a-knot]\n")
    f.write(f"RV_RANGE: [2.5, 6.0]\n")
    f.write(f"A_POLY_COEFFS:\n- [0]\n")
    f.write(f"B_POLY_COEFFS:\n- [0]\n")
    f.write(f"RV_EXP: [0]\n")
    f.write("RV_COEFFS:\n")
    first = True
    for lin, const in zip(
        f19["deltak"].values * alpha,
        f19["k_3.02"].values - 3.10 * f19["deltak"].values * alpha,
    ):
        if AV_norm:
            lin = (lin - beta)/alpha
            const = (const - beta)/alpha
        f.write(f"{'-' if first else ' '} - [{np.round(lin, 6):.6f}, {np.round(const, 6):.6f}]\n")
        first = False
