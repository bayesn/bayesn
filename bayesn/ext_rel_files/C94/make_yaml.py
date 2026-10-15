r""" From Calzetti et al. 1994ApJ...429..582C
https://ui.adsabs.harvard.edu/abs/1994ApJ...429..582C/abstract
The starburst extinction model is derived using UV and optical spectra of 39 galaxies.
The model is parameterized with Balmer optical depth and the effective extinction is
Q(x) = -2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3
where x is in inverse microns and the authors chose an arbitrary zero point such that
Q(\lambda=5500) \approx 0. Q(\lambda) is thus an effective colour excess normalised
to V \approx 5500 Angstroms, Q(x) = A(x) - A(V). This yields
A(x)/A(V) = 1+Q(x)/A(V) = 1+Q(x)/[R(V)E(B-V)] = 1+Q(x)/[R(V)Q(B)]
In BayeSN's a(x) + b(x)/R(V) + sp(x) format, a(x)=1 and b(x) = Q(x)/Q(B)
Calzetti 1997 uses 1/Q(B) = 2.656 (\lambda_B \approx 4408.57 Angstroms).
Calzetti et al. 2000 uses   2.659 (\lambda_B \approx 4409.62 Angstroms).
We will use this latter value for Q(B).

Noticeably, this dust model does not include a UV feature at 2175 angstroms.
"""

import numpy as np
from numpy.polynomial import Polynomial as P

wl_range = (1250, 8000)
wn_range = 1e4/np.array(wl_range[::-1])
A_poly = P([1])
B_poly = 2.659*P([-2.156, 1.509, -0.198, 0.011])

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [{', '.join(wn_range.astype(str))}]\n")
    f.write("OUTPUT_TYPE: axav\n")
    for var, coeffs in zip("AB", (A_poly, B_poly)):
        f.write(f"{var}_POLY_COEFFS:\n")
        f.write(f"- [{', '.join(str(x) for x in coeffs.coef[::-1])}]\n")
