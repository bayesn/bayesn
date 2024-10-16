import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 2
wls = np.zeros((N, 2))
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 3.3 <= x <= 5.9
# a(x) = 1.808 - 0.215 * x - 0.134 / ((x - 4.558) ** 2 + 0.566)
# b(x) = -2.350 + 1.403 * x + 1.103 / ((x - 4.587) ** 2 + 0.263)
wls[0] = [3.3, 5.9]
coeffs['A']['poly'][0] = [-0.215, 1.808]
coeffs['A']['rem'][0] = [-0.134]
coeffs['A']['div'][0] = [1, -4.558 * 2, 4.558**2 + 0.566]
coeffs['B']['poly'][0] = [1.403, -2.35]
coeffs['B']['rem'][0] = [1.103]
coeffs['B']['div'][0] = [1, -4.587 * 2, 4.587**2 + 0.263]

# for 5.9 <= x <= 8.0
# a(x) = NUV a(x) + -0.0077 * (x-5.9)**2 - 0.0030 * (x-5.9)**3
# b(x) = NUV b(x) + 0.2060 * (x-5.9)**2 + 0.0550 * (x-5.9)**3
wls[1] = [5.9, 8.0]
ap_NUV = P(coeffs['A']['poly'][0][::-1])
bp_NUV = P(coeffs['A']['poly'][0][::-1])
ap = P([0, 0, -0.0077, -0.0030])
bp = P([0, 0, 0.2060, 0.0550])
shift = P([-5.9, 1])
coeffs['A']['poly'][1] = (ap(shift) + ap_NUV).coef[::-1]
coeffs['A']['rem'][1] = [-0.134]
coeffs['A']['div'][0] = [1, -4.558 * 2, 4.558**2 + 0.566]
coeffs['B']['poly'][1] = (bp(shift) + bp_NUV).coef[::-1]
coeffs['B']['rem'][1] = [1.103]
coeffs['B']['div'][0] = [1, -4.587 * 2, 4.587**2 + 0.263]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    for arr, name in zip(
        (
            wls,
            coeffs["A"]["poly"],
            coeffs["B"]["poly"],
            coeffs["A"]["rem"],
            coeffs["B"]["rem"],
            coeffs["A"]["div"],
            coeffs["B"]["div"],
        ),
        (
            "REGIME_WLS",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
            "A_REMAINDER_COEFFS",
            "B_REMAINDER_COEFFS",
            "A_DIVISOR_COEFFS",
            "B_DIVISOR_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
