import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.polynomial.polynomial import polydiv as pdiv
from scipy.special import comb
from math import prod

# A(lambda)/A(V) = a(x) + b(x)*(1/Rv - 1/3.1)
N = 23
wns = np.zeros((N, 2))
exps = np.zeros(N)
coeffs = {}
for var in "AB":
    coeffs[var] = {}
    for component in ("poly", "rem", "div"):
        coeffs[var][component] = [P([0]) for _ in range(N)]
    coeffs[var]["drude"] = [P([0] * 4) for _ in range(N)]


################################### Model Constants  ###################################
b_nir_scale = -1.01251
b_nir_exp = -1.06099
scale, alpha_1, alpha_2, swave, swidth = [0.38526, 1.68467, 0.78791, 4.30578, 4.78338]
ir_drude_a = [0.06652, 9.8434, 2.21205, -0.24703]
ir_drude_b = [0.0267, 19.58294, 17.0, -0.27]

optical_poly_a = [-0.35848, 0.7122, 0.08746, -0.05403, 0.00674]
optical_poly_b = [0.12354, -2.68335, 2.01901, -0.39299, 0.03355]
optical_drudes_a = (
    (0.03893, 2.288, 0.243),
    (0.02965, 2.054, 0.179),
    (0.01747, 1.587, 0.243),
)
optical_drudes_b = (
    (0.18453, 2.288, 0.243),
    (0.19728, 2.054, 0.179),
    (0.1713, 1.587, 0.243),
)

uv_F90_a = [0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99]
uv_F90_b = [-2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99]
F90_quad = 0.5392
F90_cubic = 0.05644

# regimes are split into IR, NIR, optical, UV, FUV, with overlaps between
# adjacent regions except for UV/FUV.
# In increasing wavelength (microns), each regime starts at:
blue_end = 0.09
UV = 1 / 5.9  # ~ 0.17
optical_UV = 0.30
optical = 0.33
NIR_optical = 0.9
NIR = 1.1
IR_NIR = swave - swidth / 2  # ~1.91
IR = swave + swidth / 2  # ~6.70
red_end = 32

################################### IR ###################################
wns[0] = (1 / red_end, 1 / NIR)
# b(x) = b_nir_scale * x ** b_nir_exp
coeffs["B"]["poly"][0] = P([b_nir_scale])
exps[0] = b_nir_exp

wns[1] = (1 / red_end, 1 / IR)
# a(x) = scale*x**alpha_2*swave**(alpha_2 - alpha_1)
coeffs["A"]["poly"][1] = P([scale * swave ** (alpha_2 - alpha_1)])
exps[1] = alpha_2


################################### IR/NIR ###################################
# a(x) = scale*x**alpha_1*(1-weights) + scale*x**alpha_2*swave**(alpha_2 - alpha_1)*weights
# weights is a piecewise function that's 1 when x < 1/IR, 0 when x > 1/IR_NIR and
# smoothly transitions between the two. This logic gets used in multiple places
def get_weights(x_min, x_max, N=3, complement=False):
    """given x_min and x_max in microns, returns the weight as a function of inverse microns.
    The N argument refers to how many polynomials are used to define the weight function.

    weights = 3*y**2 - 2*y**3 where y(x') = (x' - x_min)/(x_max - x_min) with x' in microns.
    To switch to x in inverse microns,
    y(x) = (1/x - x_min)/(x_max - x_min) = (1 - x_min x)/(x*(x_max - x_min)) = rem(x)/div(x)
    weights = w_numerator/w_div = (3 * rem(x)**2 * div(x) - 2 * rem(x)**3)/div(x)**3


    if complement:
        returns the functions for 1 - weights instead
    if N = 3:
        returns a w_poly, w_rem, w_div such that weights = w_poly + w_rem / w_div
    if N = 2:
        returns a w_numerator, w_div such that weights = w_numerator / w_div
    """
    y_rem = P([1, -x_min])
    y_div = P([0, x_max - x_min])
    w_div = y_div**3
    w_poly_coef, w_rem_coef = pdiv(
        (3 * y_rem**2 * y_div - 2 * y_rem**3).coef, w_div.coef
    )
    if complement:
        w_poly_coef *= -1
        w_rem_coef *= -1
        w_poly_coef[0] += 1
    if N == 3:
        return P(w_poly_coef), P(w_rem_coef), w_div
    elif N == 2:
        return P(w_poly_coef) * w_div + P(w_rem_coef), w_div


# Need to do two sets of polynomials since the x**alpha_1 and x**alpha_2 terms can't be combined.
wns[2] = wns[3] = (1 / IR, 1 / IR_NIR)
# Regime 2 adds scale*x**alpha_1*(1-weights)
# Regime 3 adds scale*x**alpha_2*swave**(alpha_2 - alpha_1)*weights
for regime, s, exp, complement in zip(
    (2, 3),
    (scale, scale * swave ** (alpha_2 - alpha_1)),
    (alpha_1, alpha_2),
    (True, False),
):
    w_poly, w_rem, w_div = get_weights(IR_NIR, IR, N=3, complement=complement)
    coeffs["A"]["poly"][regime] = s * w_poly
    coeffs["A"]["rem"][regime] = s * w_rem
    coeffs["A"]["div"][regime] = w_div
    exps[regime] = exp

################################### NIR ###################################
# Between 1/IR_NIR <= x < NIR
wns[4] = (1 / IR_NIR, 1 / NIR)
# a(x) = scale*x**alpha_1
coeffs["A"]["poly"][4] = P([scale])
exps[4] = alpha_1

# actually, a(x) should also include two modified drude profiles for silicate features
# These cannot be exactly mapped to rational functions, so the functional form is hard-coded.
# The order of the parameters is amplitude, center, fwhm, asym
wns[5] = wns[6] = (1 / red_end, 1 / NIR)
coeffs["A"]["poly"][5] = [1]
coeffs["A"]["drude"][5] = ir_drude_a
coeffs["A"]["poly"][6] = [1]
coeffs["A"]["drude"][6] = ir_drude_b


################################### OPTICAL ###################################
# NIR/optical overlap is a smooth transition between the NIR and optical
# {a,b}(x) = weights * {a,b}(x)_NIR + (1-weights) * {a,b}(x)_optical
# First filling out the optical only portion
# {a,b}(x)_optical are 4th order polynomials (in inverse microns) and three Drude profiles
wns[7] = (1 / NIR_optical, 1 / optical)


# The Drude profiles are a special case of the modified version; asym=0
# However, when asym is 0, the Drude profile is a rational function of x' (microns)
# D(x') = amp * fwhm**2 * x'**2 / (x'**4 + (fwhm**2 - 2*center**2)* x'**2 + center**4)
# in terms of x (inverse microns)
# D(x) = amp * fwhm**2 * x**2 / (1 + (fwhm**2 - 2*center**2)* x**2 + center**4*x**4)
# The G23 law seems to call the Drude profile with the functional form for wavespace
# but parameters calibrated for inverse microns, so this function is the first D(x')
def drude_rational(amp, center, fwhm):
    rem = P([0, 0, amp * fwhm**2])
    div = P([center**4, 0, fwhm**2 - 2 * center**2, 0, 1])
    return rem, div


# This lets us add them in one wavelength regime rather than iterating for each Drude
for var, drude_params, optical_poly_coefs in zip(
    "AB", (optical_drudes_a, optical_drudes_b), (optical_poly_a, optical_poly_b)
):
    poly = P(optical_poly_coefs)
    drude_remdiv = np.zeros((3, 2), dtype=object)
    for i in range(3):
        drude_remdiv[i] = drude_rational(*drude_params[i])
    d_rems = drude_remdiv.T[0]
    d_divs = drude_remdiv.T[1]

    d_poly_coef, d_final_rem_coef = pdiv(
        (
            (
                d_rems[0] * d_divs[1] * d_divs[2]
                + d_rems[1] * d_divs[0] * d_divs[2]
                + d_rems[2] * d_divs[0] * d_divs[1]
            )
        ).coef,
        prod(d_divs).coef,
    )
    coeffs[var]["poly"][7] = poly + P(d_poly_coef)
    coeffs[var]["rem"][7] = P(d_final_rem_coef)
    coeffs[var]["div"][7] = prod(d_divs)

################################### NIR/OPTICAL ###################################
# Back to the NIR-optical overlap
# a(x)_NIR = scale*x**alpha_1 + modified drudes
# b(x)_NIR = b_nir_scale*x**b_nir_exp
# {a,b}(x)_optical = opt_poly + opt_rem / opt_div
# {a,b}(x) = weights * {a,b}(x)_NIR + (1-weights) * {a,b}(x)_optical
# need to do 5 passes, three for a(x)_NIR and its two modified Drudes
# one for b(x)_NIR because of the distinct non-integer exponent
# and one for {a,b}(x)_optical.
for i in range(8, 13):
    wns[i] = (1 / NIR, 1 / NIR_optical)
w_poly, w_rem, w_div = get_weights(NIR_optical, NIR, N=3)
for var, regime, s, exp in zip(
    "AB", (8, 9), (scale, b_nir_scale), (alpha_1, b_nir_exp)
):
    coeffs[var]["poly"][regime] = s * w_poly
    coeffs[var]["rem"][regime] = s * w_rem
    coeffs[var]["div"][regime] = w_div
    exps[regime] = exp
for i, drude_params in zip((10, 11), (ir_drude_a, ir_drude_b)):
    coeffs["A"]["poly"][i] = w_poly
    coeffs["A"]["rem"][i] = w_rem
    coeffs["A"]["div"][i] = w_div
    coeffs["A"]["drude"][i] = drude_params

# The (1-weights) * {a,b}(x)_optical is messy
# It's easiest to remove the cross terms by using polynomials for the numerator and
# denominator rather than a polynomial + remainder / divisor
w_numerator, w_div = get_weights(NIR_optical, NIR, N=2, complement=True)
for var in "AB":
    opt_poly = coeffs[var]["poly"][7]
    opt_rem = coeffs[var]["rem"][7]
    opt_div = coeffs[var]["div"][7]
    opt_numerator = opt_div * opt_poly + opt_rem
    quotient_coef, rem_coef = pdiv(
        (w_numerator * opt_numerator).coef, (w_div * opt_div).coef
    )
    coeffs[var]["poly"][12] = P(quotient_coef)
    coeffs[var]["rem"][12] = P(rem_coef)
    coeffs[var]["div"][12] = w_div * opt_div

################################### UV+FUV ###################################
# Once again handling the UV region before the UV-optical overlap
# Based on Fitzpatrick 1990
# {a,b}(x) = C1 + C2 * x + C3*(x**2 / ((x**2 - x_o**2)**2 + x**2 * gamma**2))
wns[13] = (1 / optical_UV, 1 / UV)
wns[14] = (1 / UV, 1 / blue_end)
for var, params in zip("AB", (uv_F90_a, uv_F90_b)):
    C1, C2, C3, C4, xo, gamma = params
    coeffs[var]["poly"][13] = P([C1, C2])
    coeffs[var]["rem"][13] = coeffs[var]["rem"][14] = P([0, 0, C3])
    coeffs[var]["div"][13] = coeffs[var]["div"][14] = P(
        [xo**4, 0, gamma**2 - 2 * xo**2, 0, 1]
    )
    # if x >= 5.9 (wns[12])
    # exv/ebv += C4*(F90_quad*(x-5.9)**2 + F90_cubic*(x-5.9)**3)
    fuv = C4 * P([0, 0, F90_quad, F90_cubic])(P([-5.9, 1])) + P([C1, C2])
    coeffs[var]["poly"][14] = fuv

################################### optical/UV ###################################
# {a,b}(x) = weights*{a,b}(x)_optical + (1-weights)*{a,b}(x)_UV
# There are no non-integer exponents, so the polynomials can be combined
# However, this magnifies floating point errors more than two regimes would
wns[15] = wns[16] = (1 / optical, 1 / optical_UV)
for var in "AB":
    w_numerator, w_div = get_weights(optical_UV, optical, N=2, complement=False)
    opt_poly = coeffs[var]["poly"][7]
    opt_rem = coeffs[var]["rem"][7]
    opt_div = coeffs[var]["div"][7]
    opt_numerator = opt_div * opt_poly + opt_rem

    w_numerator2 = w_div - w_numerator  # numerator for (1 - weights)
    uv_poly = coeffs[var]["poly"][13]
    uv_rem = coeffs[var]["rem"][13]
    uv_div = coeffs[var]["div"][13]
    uv_numerator = uv_div * uv_poly + uv_rem

    quotient, rem = pdiv(
        (
            uv_div * opt_numerator * w_numerator
            + opt_div * uv_numerator * w_numerator2
        ).coef,
        (uv_div * opt_div * w_div).coef,
    )
    coeffs[var]["poly"][15] = P(quotient)
    coeffs[var]["rem"][15] = P(rem)
    coeffs[var]["div"][15] = uv_div * opt_div * w_div

# Was testing polynomial math
# if method == 2:
#     for var in "AB":
#         w_numerator, w_div = get_weights(optical_UV, optical, N=2, complement=False)
#         opt_poly = coeffs[var]["poly"][7]
#         opt_rem = coeffs[var]["rem"][7]
#         opt_div = coeffs[var]["div"][7]
#         opt_numerator = opt_div * opt_poly + opt_rem
# 
#         w_numerator2 = w_div - w_numerator  # numerator for (1 - weights)
#         uv_poly = coeffs[var]["poly"][13]
#         uv_rem = coeffs[var]["rem"][13]
#         uv_div = coeffs[var]["div"][13]
#         uv_numerator = uv_div * uv_poly + uv_rem
# 
#         coeffs[var]["rem"][15] = (
#             uv_div * opt_numerator * w_numerator + opt_div * uv_numerator * w_numerator2
#         )
#         coeffs[var]["div"][15] = uv_div * opt_div * w_div
# 
# if method == 3:
#     for var in "AB":
#         w_numerator, w_div = get_weights(optical_UV, optical, N=2, complement=False)
#         opt_poly = coeffs[var]["poly"][7]
#         opt_rem = coeffs[var]["rem"][7]
#         opt_div = coeffs[var]["div"][7]
#         opt_numerator = opt_div * opt_poly + opt_rem
#         coeffs[var]["rem"][15] = w_numerator * opt_numerator
#         coeffs[var]["div"][15] = w_div * opt_div
# 
#         w_numerator2 = w_div - w_numerator  # numerator for (1 - weights)
#         uv_poly = coeffs[var]["poly"][13]
#         uv_rem = coeffs[var]["rem"][13]
#         uv_div = coeffs[var]["div"][13]
#         uv_numerator = uv_div * uv_poly + uv_rem
#         coeffs[var]["rem"][16] = w_numerator2 * uv_numerator
#         coeffs[var]["div"][16] = w_div * uv_div
# 
# if method == 4:
#     for var in "AB":
#         w_numerator, w_div = get_weights(optical_UV, optical, N=2, complement=False)
#         opt_poly = coeffs[var]["poly"][7]
#         opt_rem = coeffs[var]["rem"][7]
#         opt_div = coeffs[var]["div"][7]
#         opt_numerator = opt_div * opt_poly + opt_rem
#         opt_quotient_coef, opt_rem_coef = pdiv(
#             (w_numerator * opt_numerator).coef, (w_div * opt_div).coef
#         )
#         coeffs[var]["poly"][15] = P(opt_quotient_coef)
#         coeffs[var]["rem"][15] = P(opt_rem_coef)
#         coeffs[var]["div"][15] = w_div * opt_div
# 
#         w_numerator2 = w_div - w_numerator  # numerator for (1 - weights)
#         uv_poly = coeffs[var]["poly"][13]
#         uv_rem = coeffs[var]["rem"][13]
#         uv_div = coeffs[var]["div"][13]
#         uv_numerator = uv_div * uv_poly + uv_rem
#         uv_quotient_coef, uv_rem_coef = pdiv(
#             (w_numerator2 * uv_numerator).coef, (w_div * uv_div).coef
#         )
#         coeffs[var]["poly"][16] = P(uv_quotient_coef)
#         coeffs[var]["rem"][16] = P(uv_rem_coef)
#         coeffs[var]["div"][16] = w_div * uv_div

# G23 is parametrized as A(x)/A(V) = a(x) + b(x)*(1/RV - 1/3.1)
# but BayeSN want a(x) + b(x)/Rv.
# Thus, a(x) should be decreased by b(x)/3.1 in each regime
# If b(x) is defined in a regime where a(x) is not, the empty a(x) can be set to -b(x)/3.1.
# If a(x) is already populated, it's easier to make a new regime than to combine rational fns.
for a_regime, b_regime in zip(
    (0, 17, 9, 18, 19, 20, 21, 22), (0, 7, 9, 12, 13, 14, 15, 16)
):
    coeffs["A"]["poly"][a_regime] = -coeffs["B"]["poly"][b_regime] / 3.1
    coeffs["A"]["rem"][a_regime] = -coeffs["B"]["rem"][b_regime] / 3.1
    coeffs["A"]["div"][a_regime] = coeffs["B"]["div"][b_regime]
    wns[a_regime] = wns[b_regime]
    exps[a_regime] = exps[b_regime]

with open("BAYESN.YAML", "w") as f:
    f.write("UNITS: inverse microns\n")
    f.write(f"REGIME_EXP: [{', '.join(str(x) for x in exps)}]\n")
    for arr, name in zip(
        (
            wns,
            coeffs["A"]["poly"],
            coeffs["B"]["poly"],
            coeffs["A"]["rem"],
            coeffs["B"]["rem"],
            coeffs["A"]["div"],
            coeffs["B"]["div"],
            coeffs["A"]["drude"],
            coeffs["B"]["drude"],
        ),
        (
            "REGIME_WLS",
            "A_POLY_COEFFS",
            "B_POLY_COEFFS",
            "A_REMAINDER_COEFFS",
            "B_REMAINDER_COEFFS",
            "A_DIVISOR_COEFFS",
            "B_DIVISOR_COEFFS",
            "A_DRUDE_PARAMS",
            "B_DRUDE_PARAMS",
        ),
    ):
        f.write(f"{name}:\n")
        for coeffs in arr:
            if isinstance(coeffs, P):
                coeffs = coeffs.coef[::-1]
            f.write(f"- [{', '.join(str(x) for x in coeffs)}]\n")
