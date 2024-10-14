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
# a(x) = 1.894 - 0.373*x - 0.0101 / ((x - 4.57)**2 + 0.0384)
# b(x) = -3.490 + 2.057 * x + 0.706 / ((x - 4.59) ** 2 + 0.169)
wls[0] = [3.3, 5.9]
A_poly_coeffs[0] = [0, 0, -0.373, 1.894]
B_poly_coeffs[0] = [0, 0, 2.057, -3.49]
A_rational_coeffs[0] = np.array([1, -4.57 * 2, 4.57**2 + 0.0384]) / (-0.0101)
B_rational_coeffs[0] = np.array([1, -4.59 * 2, 4.59**2 + 0.169]) / 0.706

# for 5.9 <= x <= 11.0
# a(x) = NUV a(x) + -0.110 * (x-5.9)**2 - 0.0100 * (x-5.9)**3
# b(x) = NUV b(x) + 0.531 * (x-5.9)**2 + 0.0544 * (x-5.9)**3
# -5.9 offset adds -0.373*(-5.9) to a(x) and 2.057*(-5.9) to b(x)
# Rational coefficients go from (x - const)**2 + other const to
# ((x - 5.9) + new const)**2 + other const
# new const = 5.9  - const
wls[1] = [5.9, 11.0]
offsets[1] = -5.9
A_poly_coeffs[0] = [-0.0100, -0.110, -0.373, 1.894 - A_poly_coeffs[0][-2] * offsets[1]]
B_poly_coeffs[0] = [0.0544, 0.531, 2.057, -3.49 - B_poly_coeffs[0][-2] * offsets[1]]
A_rational_coeffs[1] = np.array(
    [1, (offsets[1] - 4.57) * 2, (offsets[1] - 4.57) ** 2 + 0.566]
) / (-0.0101)
B_rational_coeffs[1] = (
    np.array([1, (offsets[1] - 4.59) * 2, (offsets[1] - 4.59) ** 2 + 0.263]) / 0.706
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
