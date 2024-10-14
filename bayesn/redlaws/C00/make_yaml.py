import numpy as np

# A(lambda)/A(V) = 1 + k / Rv
# a(x) is always 1, b(x) = k
N = 2
min_order = 0
wls = np.zeros((N, 2))
A_poly_coeffs = np.zeros((N, 4))
B_poly_coeffs = np.zeros((N, 4))

# for 0.4545... <= x <= 1.587301
# b(x) = 2.659 * (1.040*x - 1.857)
wls[0] = [1 / 2.2, 1 / 0.63]
A_poly_coeffs[0][-1] = 1
B_poly_coeffs[0][2] = 2.659 * 1.040
B_poly_coeffs[0][3] = 2.659 * (-1.857)

# for 1.587301 < x <= 8.33...
# b(x) = 2.659 * (((0.011*x - 0.198)*x + 1.509)*x - 2.156)
wls[1] = [1 / 0.63, 1 / 0.12]
A_poly_coeffs[1][-1] = 1
B_poly_coeffs[1] = [0.029249, -0.526482, 4.012431, -5.732804]

with open("BAYESN.YAML", "w") as f:
    f.write("TYPE: LINEAR\n")
    f.write(f"NUM_REGIMES: {N}\n")
    for arr, name in zip(
        (wls, A_poly_coeffs, B_poly_coeffs),
        (
            "REGIME_WLS",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
