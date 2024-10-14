import numpy as np

# A(lambda)/A(V) = a(x) + b(x) / Rv
wls = np.zeros((4, 2))
exps = np.ones(4)
offsets = np.zeros(4)
A_poly_coeffs = np.zeros((4, 9))
A_rational_coeffs = np.zeros((4, 3))
B_poly_coeffs = np.zeros((4, 9))
B_rational_coeffs = np.zeros((4, 3))

# for 0.3 <= x <= 1.1, same as CCM89
# a(x) = 0.574*x**1.61
# b(x) = -0.527*x**1.61
wls[0] = [0.3, 1.1]
exps[0] = 1.61
A_poly_coeffs[0][7] = 0.574
B_poly_coeffs[0][7] = -0.527

# for 1.1 <= x <= 3.3
# y = x - 1.82
# The extinciton package gives
# a[0] = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y + 0.701)*y - 0.609)*y + 0.104)*y + 1.0
#      = -0.505*y**8 + 1.647*y**7 - 0.827*y**6 - 1.718*y**5 + 1.137*y**4 + 0.701*y**3 - 0.609*y**2 + 0.104*y + 1
# b[0] = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y - 3.989)*y + 2.908)*y + 1.952)*y
#      = 3.347*y**8 - 10.905*y**7 + 5.491*y**6 + 11.102*y**5 - 7.985*y**4 - 3.989*y**3 + 2.901*y**2 + 1.952*y + 0
wls[1] = [1.1, 3.3]
offsets[1] = -1.82
A_poly_coeffs[1] = [-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609, 0.104, 1]
B_poly_coeffs[1] = [3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908, 1.952, 0]


# for 3.3 <= x <= 5.9 same as CCM89
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# a(x) = 1.752 - 0.316*x + 1/(-9.61538*x**2 + 89.8077*x - 212.98)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
# b(x) = -3.090 + 1.825*x + 1/(0.829187*x**2 - 7.66169*x + 17.9166)
wls[2] = [3.3, 5.9]
A_poly_coeffs[2] = [0, 0, 0, 0, 0, 0, 0, -0.316, 1.752]
B_poly_coeffs[2] = [0, 0, 0, 0, 0, 0, 0, 1.825, -3.090]
A_rational_coeffs[2] = [-9.61538, 89.8077, -212.98]
B_rational_coeffs[2] = [0.829187, -7.66169, 17.9166]

# for 5.9 <= x <= 8.0 same as CCM89
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# a(x) = 2.20335 - 0.809407*x + 0.128358*x**2 - 0.009779*x**3+ 1/(-9.61538*x**2 + 89.8077*x - 212.98)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
# b(x) = -20.4647 + 11.9163*x - 1.92339*x**2 + 0.1207*x**3 + 1/(0.829187*x**2 - 7.66169*x + 17.9166)
wls[3] = [5.9, 8.0]
A_poly_coeffs[3] = [0, 0, 0, 0, 0, -0.009779, 0.128358, -0.809407, 2.20335]
B_poly_coeffs[3] = [0, 0, 0, 0, 0, 0.1207, -1.92339, 11.9163, -20.4647]
A_rational_coeffs[3] = [-9.61538, 89.8077, -212.98]
B_rational_coeffs[3] = [0.829187, -7.66169, 17.9166]

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
