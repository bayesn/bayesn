import functools
from numbers import Number
import os
from pathlib import Path
import subprocess

from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import numpyro.distributions as dist
from numpyro.handlers import substitute, trace
from numpyro.infer.util import log_density, _unconstrain_reparam
from ruamel.yaml import YAML
try:
    from sfdmap2 import sfdmap
    sfdmap_loaded = True
except ModuleNotFoundError:
    sfdmap_loaded = False

from bayesn import constants

def convert_z(
    z: Number,
    ra: Number,
    dec: Number,
    z_in_type: str = "hel",
    z_err: ArrayLike | None = None,
) -> float | tuple[float, float]:
    """
    Convert redshift between heliocentric and CMB reference frames.

    Given a heliocentric or CMB-frame redshift, coordinates, and optionally an
    uncertainty, convert to the other reference frame. Assumes no correlation between
    z_err and CMB_V_ERR, and ignores uncertainty in the CMB dipole location:

    .. math::

        1 + z_{\\rm cmb} = (1 + z_{\\rm hel}) (1 + z_{\\rm pv})

    Parameters
    ----------
    z
        Input redshift value.
    ra
        Right ascension in degrees (ICRS).
    dec
        Declination in degrees (ICRS).
    z_in_type
        Input frame, starting with 'hel' for heliocentric or 'cmb' for CMB frame.
    z_err : () or scalar
        Optional uncertainty on input redshift.

    Returns
    -------
    converted_z : float
        Redshift converted to the destination frame.
    converted_z_err : float
        Converted uncertainty, returned if `z_err` is provided.
    """
    c = constants.C_LIGHT
    sc = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    cmb_sc = SkyCoord(
        l=constants.CMB_L, b=constants.CMB_B, frame="galactic", unit="deg"
    )
    ang_sep = cmb_sc.separation(sc).value * np.pi / 180
    cmb_pv = constants.CMB_V * np.cos(ang_sep)
    if z_in_type.lower().startswith("hel"):
        hel_to_cmb = 1
    elif z_in_type.lower().startswith("cmb"):
        hel_to_cmb = -1
    converted_z = (1+z) * (1 + cmb_pv / c)**hel_to_cmb - 1
    if z_err is None:
        return converted_z
    cmb_pv_err = constants.CMB_V_ERR * np.cos(ang_sep)
    pv_dz = z_err * (1 + cmb_pv / c)**hel_to_cmb
    z_dpv = hel_to_cmb * (1+z) * (1 + cmb_pv/c)**(hel_to_cmb-1) * cmb_pv_err/c
    converted_z_err = np.sqrt(pv_dz**2 + z_dpv**2)
    return converted_z, converted_z_err

def convert_wavelengths(
    x: Number | ArrayLike,
    units_in: str,
    units_out: str = "angstroms",
) -> float | np.ndarray:
    """
    Convert wavelength values between physical and inverse units.

    Supported unit strings include 'micron', 'um', 'nanomet', 'nm', 'angstrom', 'aa',
    and their inverse forms (e.g. 'inv_micron').

    Parameters
    ----------
    x : (...) or scalar
        Input wavelength or wavenumber value(s).
    units_in
        Input unit description string.
    units_out
        Output unit description string. Defaults to 'angstroms'.

    Returns
    -------
    (...) or scalar
        Converted wavelength values in `units_out`.
    """
    x_mult = [1, 1]
    is_scalar = isinstance(x, Number)
    x = np.array(x)
    x = 1/x if "inv" in units_in else x
    for i, unit_str in enumerate((units_in, units_out)):
        match unit_str.lower():
            case units if "micron" in units or "um" in units:
                x_mult[i] = 1e4
            case units if "nanomet" in units or "nm" in units:
                x_mult[i] = 10
            case units if "angstrom" in units or "aa" in units:
                x_mult[i] = 1
            case _:
                raise ValueError(
                    f"Unit string {['units_in', 'units_out'][i]}={unit_str} not "
                    "recognised as a wavelength unit because it does not contain one "
                    "of the supported substrings 'micron', 'um', 'nanomet', 'nm', "
                    "'angstrom', or 'aa'."
                )
    x = x * (x_mult[0]/x_mult[1])
    x = 1/x if "inv" in units_out else x
    return float(x) if is_scalar else x

def _predict(model, args, kwargs, z_unc):
    """Run ``model`` at unconstrained latents z_unc and return the obs-site
    distribution loc (predicted flux), scale, observed value, and a 0/1 mask (1 for
    valid observations, 0 for padded/masked). The Gauss/Newton solver builds its
    residuals from these.
    """
    # map the unconstrained latents z_unc into the model's constrained parameters
    sub_fn = functools.partial(_unconstrain_reparam, z_unc)
    substituted = substitute(model, substitute_fn=sub_fn)
    with trace() as tr:
        substituted(*args, **kwargs)
    obs_site = tr["obs"]
    obs_fn = obs_site["fn"]
    if isinstance(obs_fn, dist.MaskedDistribution):
        base = obs_fn.base_dist
        mask = obs_fn._mask.astype(base.loc.dtype)
    else:
        base = obs_fn
        mask = jnp.ones_like(base.loc)
    return base.loc, base.scale, obs_site["value"], mask


def _prior_pot(model, args, kwargs, z_unc):
    """Prior contribution to -log p(z) in unconstrained space including the bijector
    log-det. Used by the Gauss-Newton solver for the prior parts of H and the gradient.
    """
    sub_fn = functools.partial(_unconstrain_reparam, z_unc)
    substituted = substitute(model, substitute_fn=sub_fn)
    # prior_only=True makes the model skip the likelihood, so log_density is the prior alone
    log_joint, _ = log_density(substituted, args, {**kwargs, "prior_only": True}, {})
    return -log_joint
def _dl_dustmaps(dustmaps_dir: Path) -> None:
    print("-------------")
    print("SFD dust maps not present, downloading them now for use in BayeSN. This only needs to happen once")
    print("-------------")
    dustmaps_dir.mkdir()
    tar_path = Path(dustmaps_dir, "sfdmap.tar.gz")
    # dustmaps.sfd.fetch()
    subprocess.run(f"wget https://github.com/kbarbary/sfddata/archive/master.tar.gz -O {tar_path}", shell=True)
    subprocess.run(f"tar -xzf {tar_path} -C {dustmaps_dir}", shell=True)
    tar_path.unlink()

def get_MWEBV(
    ra: Number,
    dec: Number,
    dustmaps_dir: Path | str | None = None,
    *args,
    **kwargs,
) -> float:
    """
    Get Milky Way E(B-V) extinction from SFD dust maps.

    Parameters
    ----------
    ra
        Right ascension in degrees (ICRS).
    dec
        Declination in degrees (ICRS).
    dustmaps_dir
        Path to directory containing SFD dust maps. If None, uses default package path.

    Returns
    -------
    float
        Milky Way E(B-V) color excess.
    """
    if not sfdmap_loaded:
        raise ImportError("Could not import sfdmap2.")
    if dustmaps_dir is None:
        dustmaps_dir = Path(Path(__file__).parent, "data", "dust_maps")
    elif isinstance(dustmaps_dir, str):
        dustmaps_dir = Path(dustmaps_dir)
    if not dustmaps_dir.exists():
        _dl_dustmaps(dustmaps_dir)
    sfd = sfdmap.SFDMap(Path(dustmaps_dir, "sfddata-master"))
    return float(sfd.ebv(ra, dec, *args, **kwargs))

def flux_to_mag(
    flux: Number | ArrayLike,
    flux_err: Number | ArrayLike,
    zp: Number = 27.5,
    nan_val: Number = np.nan,
    approx: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert fluxes and uncertainties to Pogson magnitudes given a zeropoint.

    Parameters
    ----------
    flux : (...) or scalar
        Observed flux value(s).
    flux_err : (...) or scalar
        Flux uncertainty value(s).
    zp
        Magnitude zeropoint. Defaults to 27.5.
    nan_val
        Value assigned to unphysical non-positive fluxes. Defaults to np.nan.
    approx
        If False, mag_err = 2.5*log10(1+flux_err/flux)
        If True, mag_err = 2.5/ln(10)*flux_err/flux, which assumes ln(1+flux_err/flux)
        is roughly flux_err/flux.

    Returns
    -------
    mag : (...)
        Calculated magnitudes.
    mag_err : (...)
        Calculated magnitude uncertainties.
    """
    flux = jnp.atleast_1d(flux)
    flux_err = jnp.atleast_1d(flux_err)
    if (flux_err < 0).any():
        raise ValueError("Negative flux errors are not supported.")
    mag = np.where(flux > 0, zp - 2.5 * np.log10(flux), nan_val)
    if approx:
        mag_err = np.where(flux > 0, 2.5 / np.log(10) * flux_err / flux, nan_val)
    else:
        mag_err = np.where(flux > 0, 2.5 * np.log10(1 + flux_err / flux), nan_val)
    return mag, mag_err

def mag_to_flux(
    mag: Number | ArrayLike,
    mag_err: Number | ArrayLike,
    zp: Number = 27.5,
    approx: bool = False
) -> tuple[Array, Array]:
    """
    Convert magnitudes and uncertainties to fluxes given a zeropoint.

    Parameters
    ----------
    mag : (...) or scalar
        Observed magnitude value(s).
    mag_err : (...) or scalar
        Magnitude uncertainty value(s).
    zp
        Magnitude zeropoint. Defaults to 27.5.
    approx
        If False, the flux_err = flux * (10^(mag_err / 2.5) - 1)
        If True, the flux_err = flux * mag_err * ln(10) / 2.5, which assumes
        ln(1+flux_err/flux) is roughly flux_err/flux.

    Returns
    -------
    flux : (...)
        Calculated fluxes.
    flux_err : (...)
        Calculated flux uncertainties.
    """
    mag = jnp.atleast_1d(mag)
    mag_err = jnp.atleast_1d(mag_err)
    if (mag_err <= 0).any():
        raise ValueError("Please do not provide unphysical non-positve errors.")
    flux = 10 ** (0.4 * (zp - mag))
    if approx:
        flux_err = flux * mag_err * np.log(10) / 2.5
    else:
        flux_err = flux * (10 ** (mag_err / 2.5) - 1)
    return flux, flux_err

def SNR_power_weighted_ave(
    arr: ArrayLike,
    power: Number = 2,
    SNR: ArrayLike | None = None,
    flux: ArrayLike | None = None,
    flux_err: ArrayLike | None = None,
    mag: ArrayLike | None = None,
    mag_err: ArrayLike | None = None,
) -> float:
    """
    Calculate power-weighted average using signal-to-noise ratio weights.

    Supports three channels for calculating SNR: direct `SNR`, `(flux, flux_err)`,
    or `(mag, mag_err)`. Exactly one of these three channels must be provided.

    Parameters
    ----------
    arr : (N,)
        Array of values to average.
    power
        Exponent applied to SNR for weighting. Defaults to 2 (variance weighting).
    SNR : (N,)
        Direct SNR values.
    flux : (N,)
        Flux measurements.
    flux_err : (N,)
        Flux uncertainties.
    mag : (N,)
        Magnitude measurements.
    mag_err : (N,)
        Magnitude uncertainties.

    Returns
    -------
    float
        Weighted average value.
    """
    # Figure out which channel is being provided through the args.
    SNR_channel = SNR is not None
    flux_channel = flux is not None and flux_err is not None
    mag_channel = mag is not None and mag_err is not None
    if SNR_channel + flux_channel + mag_channel > 1:
        raise ValueError(
            "SNR can be provided as the argument, or calculated through the arguments "
            "flux and flux_err or mag and mag_err. Only one of these channels should "
            "be specified through the arguments to avoid ambiguity."
        )

    # Check for positive errors
    if mag_channel:  # mag_to_flux checks for positive mag_errs.
        flux, flux_err = mag_to_flux(mag, mag_err)  # zp does not matter for SNR
    if not SNR_channel:  # need to check for positive flux_errs here
        if (flux_err <= 0).any():
            raise ValueError("Please do not provide unphysical non-positive errors")
        SNR = np.abs(flux/flux_err)
    return np.average(arr, weights=SNR**power)
def where_logic(arr: ArrayLike, val: Number, logic: str) -> tuple[np.ndarray, ...]:
    """
    Apply np.where with logic based on the string argument.

    Parameters
    ----------
    arr : (...)
        Array of arbitrary shape to filter.
    val
        A single numeric value used to filter arr.
    logic
        The logical operator in the where method. Supported operators:
        * (=, ==, eq)
        * (!=, neq)
        * (<, lt)
        * (<=, lte, leq)
        * (>, gt)
        * (>=, gte, geq)

    Returns
    -------
    tuple of np.ndarray
        Indices of `arr` where `arr logic val` evaluates to True.

    Examples
    --------
    >>> where_logic(np.arange(5, 10), 7, '<=')
    (array([0, 1, 2]),)
    """
    arr = np.array(arr)
    match logic:
        case "=" | "==" | "eq":
            return np.where(arr == val)
        case "!=" | "neq":
            return np.where(arr != val)
        case "<" | "lt":
            return np.where(arr < val)
        case "<=" | "lte" | "leq":
            return np.where(arr <= val)
        case ">" | "gt":
            return np.where(arr > val)
        case ">=" | "gte" | "geq":
            return np.where(arr >= val)
        case _:
            raise ValueError(f"Unsupported logic {logic}. Use =, <, <=, >, or >=.")

def assert_dicts_match(
    d1: dict,
    d2: dict,
    flag_missing_data: bool = False,
    flag_nans: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Check two dictionaries for discrepancies in keys and array values.

    Parameters
    ----------
    d1
        First dictionary to compare.
    d2
        Second dictionary to compare.
    flag_missing_data
        If True, missing keys or Nones in only one dict raise an AssertionError.
    flag_nans
        If True, NaNs are treated as non-matching.
    rtol
        Relative tolerance for numeric comparison.
    atol
        Absolute tolerance for numeric comparison.

    Raises
    ------
    AssertionError
        If dictionaries differ in keys, shapes, or values beyond tolerance.
    """
    keys_in_only_one = set(d1).symmetric_difference(d2)
    if keys_in_only_one and flag_missing_data:
        raise AssertionError(f"The key(s) {keys_in_only_one} appear in only one dict.")
    for key in set(d1).intersection(d2):
        val1, val2 = [np.atleast_1d(d.get(key)) for d in (d1, d2)]
        if not val1.shape == val2.shape:
            raise AssertionError(
                "The dictionaries have different shapes for key {key}.\n"
                f"d1[{key}].shape={val1.shape}, d1[{key}]={val1}\n"
                f"d2[{key}].shape={val2.shape}, d2[{key}]={val2}\n"
            )
        # Getting indices of Nones to see if there is missing data
        nones1, nones2 = [
                set(tuple(idx) for idx in np.array(np.where(v == None)).T)
                for v in (val1, val2)
        ]
        xor_idx = nones1.symmetric_difference(nones2)
        if xor_idx and flag_missing_data:
            raise AssertionError(
                f"One dict contains Nones in the {key} key where another has data. "
                f"The indices where they differ are {xor_idx}, and the values at those "
                f"indices in d1 are {[val1[idx] for idx in xor_idx]} and in d2 they "
                f"are {[val2[idx] for idx in xor_idx]}."
            )
        comp_idx = np.where((val1 != None) & (val2 != None))
        # np dtype may still be object if there were Nones, try to cast as numeric
        # for isclose comparison, or else test with ==.
        try:
            good = np.isclose(
                np.array(val1[comp_idx], dtype=float),
                np.array(val2[comp_idx], dtype=float),
                rtol=rtol, atol=atol,
                equal_nan=not flag_nans,
            )
            bad_idx = tuple(idx[~good] for idx in comp_idx)
            if not good.all():
                raise AssertionError(
                    f"The dictionaries contain numeric data in key {key} that are not "
                    f"close when using rtol={rtol} and atol={atol}. The discrepant "
                    f"indices are {[list(idx) for idx in np.array(bad_idx).T]}, with "
                    f"the values in d1[{key}]={[val1[bad_idx]]} and in d2[{key}]="
                    f"{[val2[bad_idx]]}"
                )
        except ValueError:  # From trying to get a float array of non-numbers
            if (val1[comp_idx] == val2[comp_idx]).all():
                continue
            raise AssertionError(
                f"The dictionaries contain discrepant non-numeric data in key {key}."
            )

def find_data_dir_in_SNANA(data_dir: str, private_data_path: list[str]) -> tuple[str, dict]:
    """
    Locate SNANA photometry directory across standard and private paths.

    Parameters
    ----------
    private_data_path
        List of private directory paths to search.

    Returns
    -------
    data_dir : str
        Found data directory path.
    survey_dict : dict
        Dictionary of survey definitions from SURVEY.DEF.
    """
    # Assuming you're using SNANA running on Perlmutter or a similar cluster
    # Look in standard public repositories for real data/simulations
    if os.environ.get("SNDATA_ROOT") is None:
        raise ValueError(
            "The environment variable SNDATA_ROOT is not set. This does not seem to be "
            "the expected cluster, so it is not clear where to find the data."
        )
    dir_list = ["SNDATA_ROOT/lcmerge", "SNDATA_ROOT/SIM"]
    sim_list = np.loadtxt(
        os.path.join(
            os.environ.get("SNDATA_ROOT"), "SIM", "PATH_SNDATA_SIM.LIST"
        ),
        dtype=str,
    )
    dir_list = dir_list + list([sim_dir[1:] for sim_dir in sim_list])
    pdp = [
        path[1:] if path[0] == "$" else path
        for path in private_data_path
    ]
    dir_list = dir_list + pdp  # Add any private data directories
    found_in = []
    for directory in dir_list:
        root_split = directory.split("/")
        root, remainder = root_split[0], "".join(root_split[1:])
        if not os.path.isabs(directory):
            root = os.environ.get(root, "NULL")
        if os.path.exists(os.path.join(root, remainder, data_dir)):
            found_in.append(os.path.join(root, remainder, data_dir))
    if len(found_in) == 0:
        raise ValueError(
            f"Requested photometry {data_dir} was not found in any of the "
            "usual public locations, maybe you need to specify an additional "
            "private data location."
        )
    elif len(found_in) > 1:
        raise ValueError(
            f"Requested photometry {data_dir} was found in multiple locations, "
            "please remove duplicates and ensure the one you want to use "
            "remains."
        )
    data_dir = found_in[0]
    # Load up SNANA survey definitions file
    survey_def_path = os.path.join(
        os.environ.get("SNDATA_ROOT"), "SURVEY.DEF"
    )
    survey_dict = {}
    with open(survey_def_path) as fp:
        for line in fp:
            if line[: line.find(":")] == "SURVEY":
                split = line.split()
                survey_dict[split[1]] = split[2]
    return data_dir, survey_dict

def get_yaml_comments(yaml_path: Path | str) -> dict[str, str]:
    """
    Extract comment descriptions preceding mapping keys in a YAML file.

    Parameters
    ----------
    yaml_path
        Path to the YAML file to parse comments from.

    Returns
    -------
    dict of str: str
        Dictionary mapping YAML keys to their preceding comment text.
    """
    yaml = YAML()
    with open(yaml_path, "r") as f:
        data = yaml.load(f)

    # 1. Collect all comment tokens from data.ca
    tokens = []
    if data.ca.comment and len(data.ca.comment) > 1 and data.ca.comment[1]:
        tokens.extend(data.ca.comment[1])
    for val in data.ca.items.values():
        if not val:
            continue
        for item in val:
            if item is None:
                continue
            if isinstance(item, list):
                tokens.extend(item)
            else:
                tokens.append(item)

    # 2. Get line numbers for all keys and sort them in file order
    key_lines = {k: data.lc.key(k)[0] for k in data.keys()}
    sorted_keys = sorted(data.keys(), key=lambda k: key_lines[k])

    # 3. Associate tokens between the previous key line and the current key line
    comments = {}
    for i, k in enumerate(sorted_keys):
        k_line = key_lines[k]
        prev_line = key_lines[sorted_keys[i - 1]] if i > 0 else -1
        matching_tokens = [
            t for t in tokens if prev_line < t.start_mark.line < k_line
        ]
        if matching_tokens:
            raw_text = "".join(t.value for t in matching_tokens)
            lines = [line.strip().lstrip("#").strip() for line in raw_text.splitlines()]
            cleaned = " ".join(line for line in lines if line)
            if cleaned:
                comments[k] = cleaned

    return comments

def get_model_params_from_chains(
    chains: dict,
    N_knots_sig: int = 24,
    N_W: int = 36,
) -> np.ndarray:
    """
    Collate scalar model parameters from posterior chain samples.

    Extracts and flattens parameters in the order: L_Sigma, W0, W1, sigma0, tauA,
    followed by either (RV) or (mu_R, sigma_R) depending on whether RV is fixed or sampled.

    Parameters
    ----------
    chains
        Dictionary of MCMC chains containing keys 'Ds', 'sigmaepsilon', 'L_Omega',
        'W0', 'W1', 'sigma0', 'tauA', and 'RV' or ('mu_R', 'sigma_R').
    N_knots_sig
        Number of knots in the sigma surface. Defaults to 24.
    N_W
        Number of knots in W surface. Defaults to 36.

    Returns
    -------
    (N_params, N_chains * N_samples)
        Collated matrix of model parameters across all samples and chains.
    """
    N_chains, N_samples = chains["Ds"].shape[:2]
    N_L = int(N_knots_sig*(N_knots_sig+1)/2)
    N_params = N_L + 2*N_W + 3 + int("sigma_R" in chains)
    model_params = np.zeros((N_params, N_chains*N_samples))
    sigeps_diag = np.zeros((N_chains, N_samples, N_knots_sig, N_knots_sig))
    for i in range(N_knots_sig):
        sigeps_diag[:,:,i,i] = chains["sigmaepsilon"][:,:,i]
    L_Sigma = np.matmul(sigeps_diag, chains["L_Omega"])
    k = 0
    for i in range(N_knots_sig):
        for j in range(N_knots_sig):
            if j > i:
                continue
            model_params[k] = L_Sigma[:,:,i,j].flatten()
            k += 1
    for key in ("W0", "W1"):
        for i in range(N_W):
            model_params[k] = chains[key][:,:,i].flatten()
            k += 1
    for param in ("sigma0", "tauA"):
        model_params[k] = chains[param].flatten()
        k += 1
    if "mu_R" in chains:
        model_params[k] = chains["mu_R"].flatten()
        model_params[k+1] = chains["sigma_R"].flatten()
    else:
        model_params[k] = chains["RV"].flatten()
    return model_params
