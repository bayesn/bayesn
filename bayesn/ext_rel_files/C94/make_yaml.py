""" From Calzetti et al. 1994ApJ...429..582C
https://ui.adsabs.harvard.edu/abs/1994ApJ...429..582C/abstract
The starburst extinction model is derived using UV and optical spectra of 39 galaxies.
The model is parameterized with Balmer optical depth and the effective extinction is
Q(x) = -2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3
where x is in inverse microns and the authors chose an arbitrary zero point such that
Q(5500) \approx 0. It is related to other standard measures of extinction through
Q(x) = A(x)/E(B-V)/ [k(H_beta) - k(H_alpha)]
where k(H_beta) - k(H_alpha) is roughly 1.16 (Seaton 1979)
For reference, in G23 the same quantity is 1.166.

The arbitrary zero-point allows for negative extinction values redward of 5500 A.
While not rigorous, if one treats Q(5500) as Q(V) then Q(x) - Q(V) = E(x-V)/E(B-V)/1.16.
Thus 1.16*Q(x) = E(x-V)/E(B-V)

Noticeably, this dust model does not include a UV feature at 2175 angstroms.
"""

import numpy as np
from numpy.polynomial import Polynomial as P

wl_range = (1250, 8000)
wn_range = 1e4/np.array(wl_range[::-1])
A_poly = 1.16*P([-2.156, 1.509, -0.198, 0.011])

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"WAVE_RANGE: [{', '.join(wn_range.astype(str))}]\n")
    f.write("OUTPUT_TYPE: exvebv\n")
    f.write("A_POLY_COEFFS:\n")
    f.write(f"- [{', '.join(str(x) for x in A_poly.coef[::-1])}]\n")
