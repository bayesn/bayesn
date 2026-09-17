import functools
from numbers import Number
from pathlib import Path

from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import numpy as np
from numpyro.handlers import substitute, trace
from numpyro.infer.util import log_density, _unconstrain_reparam
from jax.typing import ArrayLike
import jax.numpy as jnp
try:
    from sfdmap2 import sfdmap
    sfdmap_loaded = True
except:
    sfdmap_loaded = False

from bayesn import constants

def convert_z(
    z: Number,
    ra: Number,
    dec: Number,
    z_in_type: str = "hel",
    z_err: None | ArrayLike = None
) -> float | tuple[float, float]:
    """ Given a heliocentric or CMB-frame redshift, coordinates, and optionally an
    uncertainty, convert to the other reference frame. Assume no correlation between
    z_err and CMB_V_ERR, and ignore uncertainty in the CMB dipole location.

    1 + z_cmb = (1 + z_hel) * (1 + z_pv)

    Parameters
    ----------
    z:
    ra:
    dec:
    z_in_type:
    z_err:

    Returns
    -------
    converted_z:
    converted_z_err:
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
    dustmaps_dir: None | str = None,
    *args,
    **kwargs
) -> Number:
    if not sfdmap_loaded:
        raise ImportError("Could not import sfdmap2.")
    if dustmaps_dir is None:
        dustmaps_dir = Path(Path(__file__).parent, "data", "dust_maps")
    if not dustmaps_dir.exists():
        _dl_dustmaps(dustmaps_dir)
    sfd = sfdmap.SFDMap(Path(dustmaps_dir, "sfddata-master"))
    return sfd.ebv(ra, dec, *args, **kwargs)
def flux_to_mag(
    flux: Number | ArrayLike,
    flux_err: Number | ArrayLike,
    zp: Number = 27.5,
    nan_val: Number = np.nan,
):
    """ Convert fluxes and errors to magnitudes and errors given a zeropoint.

    Parameters
    ----------
    flux:
    flux_err:
    zp:
    nan_val:

    Returns
    -------
    mag:
    mag_err:
    """
    flux = jnp.atleast_1d(flux)
    flux_err = jnp.atleast_1d(flux_err)
    if (flux_err < 0).any():
        raise ValueError("Negative flux errors are not supported.")
    mag = np.where(flux > 0, zp - 2.5 * np.log10(flux), nan_val)
    mag_err = np.where(flux > 0, 2.5 * np.log10(1 + flux_err / flux), nan_val)
    return mag, mag_err

def mag_to_flux(
    mag: Number | ArrayLike,
    mag_err: Number | ArrayLike,
    zp: Number = 27.5
) -> tuple[float, float]:
    """ Convert magnitudes errors to fluxes and errors given a zeropoint.

    Parameters
    ----------
    mag:
    mag_err:
    zp:

    Returns
    -------
    flux:
    flux_err:
    """
    mag = jnp.atleast_1d(mag)
    mag_err = jnp.atleast_1d(mag_err)
    if (mag_err <= 0).any():
        raise ValueError("Please do not provide unphysical non-positve errors.")
    flux = 10 ** (0.4 * (zp - mag))
    flux_err = flux * (10 ** (mag_err / 2.5) - 1)
    return flux, flux_err

def SNR_power_weighted_ave(
    arr: ArrayLike,
    power: Number = 2,
    SNR: None | ArrayLike = None,
    flux: None | ArrayLike = None,
    flux_err: None | ArrayLike = None,
    mag: None | ArrayLike = None,
    mag_err: None | ArrayLike = None,
) -> float:
    """ Supports three channels for calculating SNR, either the SNR argument, the flux
    and flux_err arguments, or the mag and mag_err arguments.
    Only one of these channels should not be Nones.

    Parameters
    ----------
    arr:
    power:
    SNR:
    flux:
    flux_err:
    mag:
    mag_err:

    Returns
    -------
    ave_val:
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
    """ Apply np.where with logic based on the string argument.

    Parameters
    ----------
    arr:
        An ArrayLike (can be cast as a np.array) of arbitrary shape.
    val:
        A single numeric value used to filter arr
    logic:
        The logical operator in the where method. Supported operators are the strings:
        (=, ==, eq), (!=, neq), (<, lt), (<=, lte, leq), (>, gt), (>=, gte, geq),
        where equivalent operators are grouped by parentheses.

    Returns
    -------
    out:
        An array with the indices of arr where `arr logic val` is True.

    Example
    -------
    where_logic(np.arange(5, 10), 7, '<=') will return (array([0, 1]),), which is the
    output from np.where(np.arange(5, 10) <= 7)
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
def assert_dicts_match(d1: dict, d2: dict, flag_missing_data=False, rtol=0, atol=0):
    """ Given two dictionaries, check to see if there are any discrepancies that may
    indicate inaccurate data. If any discrepancies are found, raise a ValueError.
    Missing data in one dict does not count as a discrepancy unless the
    flag_missing_data arg is True.
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
                rtol=rtol, atol=atol
            )
            bad_idx = tuple(idx[~good] for idx in comp_idx)
            if not good.all():
                raise AssertionError(
                    f"The dictionaries contain numeric data in key {key} that are not "
                    f"close when using rtol={rtol} and atol={atol}. The discrepant "
                    f"indices are {list(idx) for idx in np.array(bad_idx).T}, with the "
                    f"values in d1[{key}]={[val1[bad_idx]]} and in d2[{key}]="
                    f"{[val2[bad_idx]]}"
                )
        except ValueError:  # From trying to get a float array of non-numbers
            if (val1[comp_idx] == val2[comp_idx]).all():
                continue
            raise AssertionError(
                f"The dictionaries contain discrepant non-numeric data in key {key}."
            )
def find_data_dir_in_SNANA(private_data_path) -> tuple[str, dict]:
    # Assuming you're using SNANA running on Perlmutter or a similar cluster
    # Look in standard public repositories for real data/simulations
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
    with open(survey_def_path) as fp:
        for line in fp:
            if line[: line.find(":")] == "SURVEY":
                split = line.split()
                survey_dict[split[1]] = split[2]
    return data_dir, survey_dict
