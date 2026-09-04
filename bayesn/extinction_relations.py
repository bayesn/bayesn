import os
from ruamel.yaml import YAML
from warnings import warn

import jax.numpy as jnp
from jax import Array
from jax.lax import cond
from jax.typing import ArrayLike
import numpy as np

from .spline_utils import invKD, spline_coeffs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_REL_DIR = os.path.join(BASE_DIR, "ext_rel_files")
yaml = YAML(typ="safe")

class DustExtRel:
    def __init__(
        self,
        name: str = "G23",
        x_in: ArrayLike | str = "default",
        x_units: str = "angstroms",
        default_min_wave: int | float = 3500.,
        default_max_wave: int | float = 9500.,
        default_wave_bins: int = 300,
        verbose: bool = True
    ) -> None:
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
        if self.output_type not in ("axav", "axebv", "exvebv"):
            raise ValueError(
                f"The OUTPUT_TYPE yaml key was given as {self.output_type}, which is "
                "not supported. Valid options are 'axav' for A(x)/A(V), 'axebv', for "
                "A(x)/E(B-V), or 'exvebv' for E(x-V)/E(B-V)."
            )
        self.xk = jnp.array(params.get("L_KNOTS", jnp.zeros((self.n_subdomains, 1))))
        self.rv_coeffs = jnp.array(params.get("RV_COEFFS", jnp.zeros((1, 1, 1))))
        self.rv_exp = jnp.array(params.get("RV_EXP", jnp.zeros(1)))
        return params

    def set_x(self, x_in: ArrayLike | str = "default", x_units: str = "angstroms") -> None:
        x_exp = 1
        x_mult = [1, 1]
        error_name = ("extinction relation 'UNIT'", "x_units")
        for i, unit_str in enumerate((self.units, x_units)):
            if "inv" in unit_str:
                x_exp *= -1
            match unit_str.lower():
                case units if "micron" in units or "um" in units:
                    x_mult[i] = 1e4
                case units if "nanomet" in units or "nm" in units:
                    x_mult[i] = 10
                case units if "angstrom" in units or "aa" in units:
                    x_mult[i] = 1
                case _:
                    raise ValueError(
                        f"Unit string {error_name[i]}={units} not recognised as a "
                        "wavelength unit because it does not contain one of the "
                        "supported substrings 'micron', 'um', 'nanomet', 'nm', "
                        "'angstrom', or 'aa'."
                    )
        if isinstance(x_in, str) and x_in == "default":
            log_range = jnp.log10(jnp.array([self.default_min_wave, self.default_max_wave]))
            log_wave = jnp.power(10, jnp.linspace(*log_range, self.default_wave_bins))
            self.x = jnp.power(x_mult[1]/x_mult[0] * log_wave, x_exp)
        elif isinstance(x_in, str) and x_in == "full_range":
            self.x = np.linspace(*self.range, self.default_wave_bins)
        else:
            self.x = (x_mult[1]/x_mult[0] * x_in) ** x_exp

    def get_undefined_intervals(self, x):
        undefined_intervals = []
        if min(x) < min(self.range):
            undefined_intervals.append(str((min(x), min(self.range))))
        if max(x) > max(self.range):
            undefined_intervals.append(str((max(self.range), max(x))))
        return undefined_intervals

    def load(self, name: str, x_in: ArrayLike | str = "default", x_units: str = "angstroms", verbose: bool = True) -> None:
        """
        Loads a dust extinction relation from a yaml file.

        The redlaws/README file has technical information, but to summarize: each
        extinction relation is specified as a piecewise function a(x) + b(x)/RV + sp(x).
        And returns either A(x)/A(V), A(x)/E(B-V), or E(x-V)/E(B-V) as indicated by the
        OUTPUT_TYPE keyword axav, axebv, or exvebv (default axav).

        Within any given subdomain i,
        {a,b}(x) = x**{A,B}_EXP * (P_{a,b}(x) + R_{a,b}(x)/D_{a,b}(x))
        where P_{a,b}, R_{a,b}, and D_{a,b} are all standard polynomials.
        sp(x) has knots at L_KNOTS[i] with values given by RV**RV_EXP*P(RV).

        RV can be varied during sampling, but everything that can be pre-calculated is
        stored in the attributes aw_ax, bx, Jx, rv_coeffs,
        RV_exps, and output_type, which are used by the _get_axav method.

        The recognized YAML keywords and their descriptions are described in the
        redlaws/README file.

        Parameters
        ----------
        name:
            Name of dust extinction relation to load. Options are
            ``CCM89``:
            ``C94``:
            ``O94``:
            ``F99``:
            ``F99_AVGLMC``:
            ``F99_LMC2``:
            ``F99_SNANA``:
            ``C00``:
            ``F04``:
            ``VCG04``:
            ``GCC09``:
            ``F19``:
            ``D22``:
            ``G23``:
        x:
            Wavelengths used to calculate reddening in _get_axav.
            To use SEDmodel.model_wave as the wavelength grid, use x="default".
            To use linearly-spaced wavelengths in inverse microns spanning the range
            permitted by the extinction relation (i.e. for testing), use x="full_range".
        verbose:

        Attributes
        ----------
        """
        params = self.load_params(name=name, verbose=verbose)
        self.set_x(x_in=x_in, x_units=x_units)
        x = np.asarray(self.x)
        zeros = np.zeros((self.n_subdomains, 1))

        undefined_intervals = self.get_undefined_intervals(x)
        if undefined_intervals != [] and verbose:
            warn(UserWarning(
                f"WARNING: The {self.name} dust extinction relation is only valid "
                f"from {min(self.range)} to {max(self.range)} {self.units}. There will "
                f"be no extinction applied in {' and '.join(undefined_intervals)} "
                f"{self.units}."
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
        Parameters
        ----------
        RV: ArrayLike shape (N_sn,) or (1, N_sn,)
            R(V) values used for the calculation.

        Returns
        -------
        ax_av: ArrayLike shape (N_sn, N_wl)
            A set of A(x)/A(V) values calculated with the loaded dust extinction relation.
        """

        RV = jnp.atleast_1d(jnp.squeeze(RV)).astype(float)
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
        """ A wrapper for _get_axav that prints out a warning if the RV is out of the
        specified RV_range.

        Parameters
        ----------
        RV: ArrayLike shape (N_sn,) or (1, N_sn,)
            R(V) values used for the calculation.

        Returns
        -------
        ax_av: ArrayLike shape (N_sn, N_wl)
            A set of A(x)/A(V) values calculated with the loaded extinction relation.
        """
        shaped_RV = jnp.atleast_1d(jnp.squeeze(RV))
        if verbose and len(shaped_RV.shape) > 1:
            raise ValueError(
                f"RV should be a constant, 1D-array, or ArrayLike that can be "
                "jnp.squeezed into a 1D-array. The current squeezed shape is "
                f"{shaped_RV.shape}, which is not supported."
            )
        if verbose and (
            shaped_RV < self.rv_range[0]).any() or (shaped_RV > self.rv_range[1]
        ).any():
            warn(UserWarning(
                f"WARNING: The {self.name} dust extinction relation is only valid with "
                f"RV in the interval {self.rv_range}. RV={RV} will require "
                "extrapolation beyond the data used to define the model."
            ))
        return self._get_axav(RV=RV)
