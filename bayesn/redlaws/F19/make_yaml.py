""" From Fitzpatrick 2019ApJ...886..108F
https://iopscience.iop.org/article/10.3847/1538-4357/ab4c3a

Combines new HST/STIS optical spectrophotometry with existing IUE UV spectrophotometry
and 2MASS NIR photometry to make 72 extinction curves with gapless coverage from NIR through UV.

There is some scatter around the R-dependent curves beyond what individual
uncertainties would suggest, which could imply a mode of variation beyond R_V.
"""
import pandas as pd
import numpy as np

f19 = pd.read_csv("F19_tabulated.dat", delimiter='\s+')
with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in f19['x'].values)}]\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [0.3, 8.7]\n")
    f.write(f"RV_RANGE: [2.5, 6.0]\n")
    f.write(f"MIN_ORDER: 0\n")
    f.write("RV_COEFFS:\n")
    for lin, const in zip(
        f19["deltak"].values * 0.99,
        f19["k_3.02"].values - 3.10 * f19["deltak"].values * 0.99,
    ):
        f.write(f"- [{np.round(lin, 6):.6f}, {np.round(const, 6):.6f}]\n")
