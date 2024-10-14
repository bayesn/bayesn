import numpy as np

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 5
wls = np.zeros((N, 2))
exps = np.ones(N)
offsets = np.zeros(N)
A_poly_coeffs = np.zeros((N, 8))
A_rational_coeffs = np.zeros((N, 3))
B_poly_coeffs = np.zeros((N, 8))
B_rational_coeffs = np.zeros((N, 3))

# for 0.3 <= x <= 1.1
# a(x) = 0.574*x**1.61
# b(x) = -0.527*x**1.61
wls[0] = [0.3, 1.1]
exps[0] = 1.61
A_poly_coeffs[0][6] = 0.574
B_poly_coeffs[0][6] = -0.527

# for 1.1 <= x <= 3.3
# y = x - 1.82
# a(x) = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
# b(x) = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
wls[1] = [1.1, 3.3]
offsets[1] = -1.82
A_poly_coeffs[1] = [0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1]
B_poly_coeffs[1] = [-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0]


# for 3.3 <= x <= 5.9
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# a(x) = 1.752 - 0.316*x + 1/(-9.61538*x**2 + 89.8077*x - 212.98)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
# b(x) = -3.090 + 1.825*x + 1/(0.829187*x**2 - 7.66169*x + 17.9166)
wls[2] = [3.3, 5.9]
A_poly_coeffs[2] = [0, 0, 0, 0, 0, 0, -0.316, 1.752]
B_poly_coeffs[2] = [0, 0, 0, 0, 0, 0, 1.825, -3.090]
A_rational_coeffs[2] = [-9.61538, 89.8077, -212.98]
B_rational_coeffs[2] = [0.829187, -7.66169, 17.9166]

# for 5.9 <= x <= 8.0
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# a(x) = 2.20335 - 0.809407*x + 0.128358*x**2 - 0.009779*x**3+ 1/(-9.61538*x**2 + 89.8077*x - 212.98)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
# b(x) = -20.4647 + 11.9163*x - 1.92339*x**2 + 0.1207*x**3 + 1/(0.829187*x**2 - 7.66169*x + 17.9166)
wls[3] = [5.9, 8.0]
A_poly_coeffs[3] = [0, 0, 0, 0, -0.009779, 0.128358, -0.809407, 2.20335]
B_poly_coeffs[3] = [0, 0, 0, 0, 0.1207, -1.92339, 11.9163, -20.4647]
A_rational_coeffs[3] = [-9.61538, 89.8077, -212.98]
B_rational_coeffs[3] = [0.829187, -7.66169, 17.9166]

# for 8.0 <= x <= 10.0
# a(x) = -1.073 - 0.628*(x - 8) + 0.137*(x - 8)**2 - 0.070*(x - 8)**3
# b(x) = 13.670 + 4.257*(x - 8) - 0.420*(x - 8)**2 + 0.374*(x - 8)**3
wls[4] = [8.0, 10.0]
offsets[4] = -8
A_poly_coeffs[4] = [0, 0, 0, 0, -0.070, 0.137, -0.628, -1.073]
B_poly_coeffs[4] = [0, 0, 0, 0, 0.374, -0.420, 4.257, 13.670]

with open("BAYESN.YAML", "w") as f:
    f.write("TYPE: LINEAR\n")
    f.write(f"NUM_REGIMES: {len(wls)}\n")
    f.write(f"REGIME_EXP: [{', '.join(str(x) for x in exps)}]\n")
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
