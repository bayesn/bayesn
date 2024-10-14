import numpy as np

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 2
wls = np.zeros((N, 2))
offsets = np.zeros(N)
A_poly_coeffs = np.zeros((N, 4))
A_rational_coeffs = np.zeros((N, 3))
B_poly_coeffs = np.zeros((N, 4))
B_rational_coeffs = np.zeros((N, 3))

# for 3.3 <= x <= 5.9
# a(x) = 1.808 - 0.215 * x - 0.134 / ((x - 4.558) ** 2 + 0.566)
# b(x) = -2.350 + 1.403 * x + 1.103 / ((x - 4.587) ** 2 + 0.263)
wls[0] = [3.3, 5.9]
A_poly_coeffs[0] = [0, 0, -0.215, 1.808]
B_poly_coeffs[0] = [0, 0, 1.403, -2.35]
A_rational_coeffs[0] = np.array([1, -4.558 * 2, 4.558**2 + 0.566]) / (-0.134)
B_rational_coeffs[0] = np.array([1, -4.587 * 2, 4.587**2 + 0.263]) / 1.103

# for 5.9 <= x <= 8.0
# a(x) = NUV a(x) + -0.0077 * (x-5.9)**2 - 0.0030 * (x-5.9)**3
# b(x) = NUV b(x) + 0.2060 * (x-5.9)**2 + 0.0550 * (x-5.9)**3
# -5.9 offset adds -0.215*(-5.9) to a(x) and 1.403*(-5.9) to b(x)
# Rational coefficients go from (x - const)**2 + other const to
# ((x - 5.9) + new const)**2 + other const
# new const = 5.9  - const
wls[1] = [5.9, 8.0]
offsets[1] = -5.9
A_poly_coeffs[1] = [-0.003, -0.0077, -0.215, 1.808 -  A_poly_coeffs[0][-2] * offsets[1]]
B_poly_coeffs[1] = [0.055, 0.206, 1.403, -2.35 - B_poly_coeffs[0][-2]*offsets[1]]
A_rational_coeffs[1] = np.array([1, (5.9 - 4.558) * 2, (5.9 - 4.558) ** 2 + 0.566]) / (
    -0.134
)
B_rational_coeffs[1] = (
    np.array([1, (5.9 - 4.587) * 2, (5.9 - 4.587) ** 2 + 0.263]) / 1.103
)

with open("BAYESN.YAML", "w") as f:
    f.write("TYPE: LINEAR\n")
    f.write(f"NUM_REGIMES: {len(wls)}\n")
    f.write(f"REGIME_OFFSET: [{', '.join(str(x) for x in offsets)}]\n")
    for arr, name in zip(
        (wls, A_poly_coeffs, B_poly_coeffs, A_rational_coeffs, B_rational_coeffs),
        (
            "REGIME_WLS",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
            "A_RATIONAL_COEFFS",
            "B_RATIONAL_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
