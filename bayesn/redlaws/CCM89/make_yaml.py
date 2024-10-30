import numpy as np
from numpy.polynomial import Polynomial as P

# A(lambda)/A(V) = a(x) + b(x) / Rv
N = 5
wls = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [[0] for _ in range(N)]

# for 0.3 <= x <= 1.1
# a(x) = 0.574*x**1.61 = (0.574)*x**1.61
# b(x) = -0.527*x**1.61 = (-0.527)**1.61
wls[0] = [0.3, 1.1]
exps[0] = 1.61
coeffs["A"]["poly"][0] = [0.574]
coeffs["B"]["poly"][0] = [-0.527]

# for 1.1 <= x <= 3.3
# y = x - 1.82
# a(x) = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
# b(x) = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
wls[1] = [1.1, 3.3]
a = P([1, 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999])
b = P([0, 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])
shift = P([-1.82, 1])
coeffs["A"]["poly"][1] = a(shift).coef[::-1]
coeffs["B"]["poly"][1] = b(shift).coef[::-1]


# for 3.3 <= x <= 5.9
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263)
wls[2] = [3.3, 5.9]
coeffs["A"]["poly"][2] = [-0.316, 1.752]
coeffs["A"]["rem"][2] = [-0.104]
coeffs["A"]["div"][2] = [1, -4.67 * 2, 4.67**2 + 0.341]
coeffs["B"]["poly"][2] = [1.825, -3.090]
coeffs["B"]["rem"][2] = [1.206]
coeffs["B"]["div"][2] = [1, -4.62 * 2, 4.62**2 + 0.263]

# for 5.9 <= x <= 8.0
# a(x) = 1.752 - 0.316*x - 0.104/((x-4.67)**2 + 0.341) - 0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
# a(x) = 2.20335 - 0.809407*x + 0.128358*x**2 - 0.009779*x**3 - 0.104/((x-4.67)**2 + 0.341)
# b(x) = -3.090 + 1.825*x + 1.206/((x-4.62)**2 + 0.263) + 0.2130*(x-5.9)**2 + 0.1207*(x-5.9)**3
# b(x) = -20.4647 + 11.9163*x - 1.92339*x**2 + 0.1207*x**3 + 1.206/((x-4.62)**2 + 0.263)
wls[3] = [5.9, 8.0]
coeffs["A"]["poly"][3] = [-0.009779, 0.128358, -0.809407, 2.20335]
coeffs["A"]["rem"][3] = [-0.104]
coeffs["A"]["div"][3] = [1, -4.67 * 2, 4.67**2 + 0.341]
coeffs["B"]["poly"][3] = [0.1207, -1.92339, 11.9163, -20.4647]
coeffs["B"]["rem"][3] = [1.206]
coeffs["B"]["div"][3] = [1, -4.62 * 2, 4.62**2 + 0.263]

# for 8.0 <= x <= 10.0
# a(x) = -1.073 - 0.628*(x - 8) + 0.137*(x - 8)**2 - 0.070*(x - 8)**3
# b(x) = 13.670 + 4.257*(x - 8) - 0.420*(x - 8)**2 + 0.374*(x - 8)**3
wls[4] = [8.0, 10.0]
a = P([-1.073, -0.628, 0.137, -0.070])
b = P([13.670, 4.257, -0.420, 0.374])
shift = P([-8, 1])
coeffs["A"]["poly"][4] = a(shift).coef[::-1]
coeffs["B"]["poly"][4] = b(shift).coef[::-1]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"REGIME_EXP: [{', '.join(str(x) for x in exps)}]\n")
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
