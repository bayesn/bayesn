import pandas as pd
import numpy as np

f19 = pd.read_csv("F19_tabulated.dat", delimiter='\s+')
with open("BAYESN.YAML", "w") as f:
    f.write(f"L_KNOTS: [{', '.join(str(x) for x in f19['x'].values)}]\n")
    f.write(f"UNITS: inverse microns\n")
    f.write(f"MIN_ORDER: 0\n")
    f.write("RV_COEFFS:\n")
    for lin, const in zip(
        f19["deltak"].values * 0.99,
        f19["k_3.02"].values - 3.10 * f19["deltak"].values * 0.99,
    ):
        f.write(f"- [{np.round(lin, 6):.6f}, {np.round(const, 6):.6f}]\n")
