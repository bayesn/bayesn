import os
from ruamel.yaml import YAML
from warnings import warn

import jax.numpy as jnp
from jax import Array
from jax.lax import cond
from jax.typing import ArrayLike
import numpy as np

from .spline_utils import invKD, spline_coeffs
from .utils import convert_wavelengths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_REL_DIR = os.path.join(BASE_DIR, "ext_rel_files")
yaml = YAML(typ="safe")

class DustExtRel:
    """
    Parametric dust extinction relation for optical and NIR wavelengths.

    Implements analytic/spline models for dust extinction laws (e.g. CCM89, F99, G23)
    allowing dynamic evaluation across wavelengths as a function of total-to-selective
    extinction ratio R(V).
    """

    def __init__(
        self,
        name: str = "G23",
        x_in: ArrayLike | str = "default",
        x_units: str = "angstroms",
        default_min_wave: float | int = 3500.0,
        default_max_wave: float | int = 9500.0,
        default_wave_bins: int = 300,
        verbose: bool = True,
    ) -> None:
        """
        Initialise and load a dust extinction relation.

        Parameters
        ----------
        name
            Name of built-in relation (e.g. 'G23', 'F99', 'CCM89') or path to YAML definition.
        x_in : (N_wl,) or str
            Wavelength array or preset: 'default' (log-spaced over default range) or 'full_range'.
        x_units
            Units of input wavelengths ('angstroms', 'nm', or 'microns').
        default_min_wave
            Short-wavelength boundary for default grid (in `x_units`).
        default_max_wave
            Long-wavelength boundary for default grid (in `x_units`).
        default_wave_bins
            Number of wavelength bins for default grid.
        verbose
            Whether to print loading information.
        """
        self.default_min_wave = default_min_wave
        self.default_max_wave = default_max_wave
        self.default_wave_bins = default_wave_bins
        self.load(name=name, x_in=x_in, x_units=x_units, verbose=verbose)

    def __str__(self):
        return f"DustExtRel: {self.name}"

    def load_params(self, name: str, verbose: bool = True) -> dict:
        built_in_DERs = next(os.walk(EXT_REL_DIR))[1]
        if os.path.exists(name):
            if verbose:
                print(f"Loading custom dust extinction relation at {name}")
            with open(name, "r") as file:
                params = yaml.load(file)
        elif name in built_in_DERs:
            if verbose:
                print(f"Loading built-in dust extinction relation {name}")
            with open(os.path.join(EXT_REL_DIR, name, "BAYESN.YAML"), "r") as file:
                params = yaml.load(file)
        else:
            raise FileNotFoundError(
                f"Specified dust extinction relation {name} does not exist and does "
                f"not correspond to one of the built-in model {built_in_DERs}"
            )
        self.n_subdomains = len(params.get("SUBDOMAINS", [1]))
        self.name = name
        self.range = jnp.array(params["WAVE_RANGE"])
        self.rv_range = jnp.array(params.get("RV_RANGE", [2, 6]))
        self.units = params.get("UNITS", "inverse microns").lower()
        self.output_type = params.get("OUTPUT_TYPE", "axav")
        self.spline_bc_type = params.get("SPLINE_BC_TYPE", "natural")
        self.xk = jnp.array(params.get("L_KNOTS", jnp.zeros((self.n_subdomains, 1))))
        self.rv_coeffs = jnp.array(params.get("RV_COEFFS", jnp.zeros((1, 1, 1))))
        self.rv_exp = jnp.array(params.get("RV_EXP", jnp.zeros(1)))
        self._validate_params()
        return params

    def _validate_params(self):
        msg = [f"Something is wrong with the DustExtRel parameters {self.name}:"]
        if not isinstance(self.n_subdomains, int):
            msg.append(f"n_subdomains {self.n_subdomains} is not an integer.")
        if not self.range[0] < self.range[1]:
            msg.append(f"range {self.range} is not increasing.")
        if not any(x in self.units for x in ["micron", "um", "nanomet", "nm", "angstrom", "aa"]):
            msg.append(
                f"units {self.units} is not recognised, use (optionally inverse) "
                "microns, nanometers, or angstroms."
            )
        if not self.xk.shape[0] == self.n_subdomains:
            msg.append(f"xk shape {self.xk.shape} does not start with n_subdomains {self.n_subdomains}.")
        if not len(self.rv_coeffs.shape) == 3:
            msg.append(f"rv_coeffs should have three dimensions, but shape is {self.rv_coeffs.shape}.")
        if self.output_type not in ("axav", "axebv", "exvebv"):
            msg.append(
                f"output_type {self.output_type} is not supported, use 'axav' for "
                "A(x)/A(V), 'axebv', for A(x)/E(B-V), or 'exvebv' for E(x-V)/E(B-V)."
            )
        if len(msg) > 1:
            raise ValueError("\n    ".join(msg))

    def set_x(self, x_in: ArrayLike | str = "default", x_units: str = "angstroms") -> None:
        if not isinstance(x_in, str):
            self.x = convert_wavelengths(x_in, units_in=x_units, units_out=self.units)
        elif isinstance(x_in, str) and x_in == "default":
            log_range = jnp.log10(jnp.array([self.default_min_wave, self.default_max_wave]))
            log_wave = jnp.power(10, jnp.linspace(*log_range, self.default_wave_bins))
            self.x = convert_wavelengths(log_wave, units_in=x_units, units_out=self.units)
        elif isinstance(x_in, str) and x_in == "full_range":
            self.x = np.linspace(*self.range, self.default_wave_bins)
        else:
            raise ValueError(
                f"x_in={x_in} not recognised. It should be an ArrayLike, 'default', "
                "or 'full_range'."
            )

    def load(
        self,
        name: str,
        x_in: ArrayLike | str = "default",
        x_units: str = "angstroms",
        verbose: bool = True,
    ) -> None:
        """
        Load a dust extinction relation from a YAML definition.

        The relation is defined as a piecewise function:

        .. math::

            a(x) + b(x)/R_V + {\\rm sp}(x)

        parametrised in terms of A(x)/A(V), A(x)/E(B-V), or E(x-V)/E(B-V)
        depending on the OUTPUT_TYPE specified in the YAML model file.
        Downstream evaluation via :meth:`_get_axav` coerces this to A(x)/A(V).

        Parameters
        ----------
        name
            Name of dust extinction relation to load. Built-in options include:
            'CCM89', 'C94', 'O94', 'F99', 'F99_AVGLMC', 'F99_LMC2', 'F99_SNANA',
            'C00', 'F04', 'VCG04', 'GCC09', 'F19', 'D22', 'G23'.
        x_in : (N_wl,) or str
            Wavelength array or preset: 'default' or 'full_range'.
        x_units
            Units of `x_in` ('angstroms', 'nm', or 'microns').
        verbose
            Whether to print loading details and warnings.
        """
        params = self.load_params(name=name, verbose=verbose)
        self.set_x(x_in=x_in, x_units=x_units)
        x = np.asarray(self.x)
        zeros = np.zeros((self.n_subdomains, 1))

        if verbose and (min(x) < min(self.range) or max(x) > max(self.range)):
            range_aa = sorted(convert_wavelengths(self.range, units_in=self.units, units_out="angstroms"))
            x_aa = sorted(convert_wavelengths([min(x), max(x)], units_in=self.units, units_out="angstroms"))
            warn(UserWarning(
                f"The {self.name} dust extinction relation is only valid from from "
                f"{range_aa[0]} to {range_aa[1]} angstroms but was loaded from "
                f"{x_aa[0]} to {x_aa[1]}. No extinction will be applied outside the "
                "supported range."
            ))

        der_A = np.zeros_like(x)
        der_B = np.zeros_like(x)
        der_J = np.zeros((self.n_subdomains, len(x), self.rv_coeffs.shape[1] + 1))
        # last dimension gets +1 for spline interpolation padding.
        for i in range(self.n_subdomains):
            wl_range = params.get("SUBDOMAINS", [self.range])[i]
            idx = np.where((wl_range[0] <= x) & (x < wl_range[1]))[0]
            # Include last element if sub-domain edge coincides with domain edge.
            if wl_range[1] == self.range[1] and wl_range[1] in x:
                idx = np.append(idx, np.where(x == wl_range[1])[0])
            if not idx.shape[0]:
                continue
            mod_x = x[idx]
            for var in "AB":
                exp_term = mod_x ** params.get(f"{var}_EXP", zeros)[i]
                poly = np.array(params.get(f"{var}_POLY_COEFFS", zeros)[i], dtype=np.float64)
                rem = np.array(params.get(f"{var}_REMAINDER_COEFFS", zeros)[i], dtype=np.float64)
                div = np.array(params.get(f"{var}_DIVISOR_COEFFS", zeros)[i], dtype=np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rational_term = np.polyval(poly, mod_x) + np.nan_to_num(
                        np.polyval(rem, mod_x) / np.polyval(div, mod_x), posinf=0, neginf=0
                    )
                    # Asymmetric Drude profile, symmetric profiles converted to polynomials
                    amp, center, fwhm, asym = np.array(
                        params.get(f"{var}_DRUDE_PARAMS", np.zeros((self.n_subdomains, 4)))[
                            i
                        ],
                        dtype=np.float64,
                    )
                    gamma = 2 * fwhm / (1 + np.exp(asym * (1 / mod_x - center)))
                    drude_divisor = ((1 / (mod_x * center) - mod_x * center) ** 2 + (gamma / center) ** 2)
                    drude_term = np.nan_to_num(amp * (gamma / center) ** 2 / drude_divisor, nan=1, posinf=0, neginf=0)
                term = exp_term * rational_term * drude_term
                if var == "A":
                    der_A[idx] += term
                else:
                    der_B[idx] += term
            xk_i = np.asarray(self.xk[i])
            if np.count_nonzero(xk_i):
                bc_type = params.get("SPLINE_BC_TYPE", ["natural" for _ in range(self.n_subdomains)])[i]
                der_J[i, idx] += np.asarray(spline_coeffs(
                    mod_x, xk_i, invKD(xk_i, bc_type=bc_type)
                ))
        self.ax = jnp.asarray(der_A)
        self.bx = jnp.asarray(der_B)
        self.Jx = jnp.asarray(der_J)

    def _get_axav(self, RV: ArrayLike) -> Array:
        """
        Calculate A(x)/A(V) using loaded dust extinction relation parameters.

        JIT-compilable method evaluating the extinction relation:
        :math:`a(x) + b(x)/R_V + {\\rm sp}(x)`.
        If the relation was defined in terms of A(x)/E(B-V) or E(x-V)/E(B-V),
        it is coerced to A(x)/A(V) so that the output is always A(x)/A(V).

        Parameters
        ----------
        RV : (N_sn,)
            R(V) values used for the calculation.

        Returns
        -------
        ax_av : (N_sn, N_wl)
            Extinction law values A(x)/A(V).
        """
        max_len_coeffs = self.rv_coeffs.shape[2]
        RV_powers = jnp.linspace(self.rv_exp, self.rv_exp+max_len_coeffs-1, max_len_coeffs)[::-1].T
        RV_matrix = jnp.power(RV[None, None, :], RV_powers[:, :, None])
        yk = jnp.matmul(self.rv_coeffs, RV_matrix)
        padding = jnp.power(RV[None, None, :], self.rv_exp[:, None, None])
        padded_yk = jnp.concatenate([padding, yk], axis=1)
        Jx = jnp.matmul(self.Jx, padded_yk).sum(axis=0).T
        ret_val = self.ax[None, :] + self.bx[None, :] / RV[:, None] + Jx

        no_op = lambda v: v
        ret_val = cond(self.output_type == "axav", no_op, lambda v: v / RV[:, None], ret_val)
        return cond(self.output_type == "exvebv", lambda v: v + 1, no_op, ret_val)

    def get_axav(self, RV: ArrayLike, verbose: bool = True) -> Array:
        """
        Calculate A(x)/A(V) with validation and shape coercion.

        Non-JIT wrapper for :meth:`_get_axav`. Prints a warning if any R(V) values are
        outside the valid range and coerces inputs to 1D arrays. Intended for interactive use.

        Parameters
        ----------
        RV : (N_sn,) or (1, N_sn) or scalar
            R(V) values used for the calculation.
        verbose : bool, default=True
            Whether to print warnings for out-of-range R(V) or shape issues.

        Returns
        -------
        ax_av : (N_sn, N_wl)
            Extinction law values A(x)/A(V) evaluated across wavelength grid.
        """
        shaped_RV = jnp.atleast_1d(jnp.squeeze(RV)).astype(float)
        if verbose and len(shaped_RV.shape) > 1:
            raise ValueError(
                f"RV should be a constant, 1D-array, or ArrayLike that can be "
                "jnp.squeezed into a 1D-array. The current squeezed shape is "
                f"{shaped_RV.shape}, which is not supported."
            )
        if verbose and (
            (shaped_RV < self.rv_range[0]).any() or (shaped_RV > self.rv_range[1]).any()
        ):
            warn(UserWarning(
                f"WARNING: The {self.name} dust extinction relation is only valid with "
                f"RV in the interval {self.rv_range}. RV={shaped_RV} will require "
                "extrapolation beyond the data used to define the model."
            ))
        return self._get_axav(RV=shaped_RV)
