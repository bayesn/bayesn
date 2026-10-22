"""
Class for BayeSN datasets comprising one or more SNe.
This module provides a standard data format, methods for performing astronomical
transformations to the data, applying data quality cuts, and other domain-specific
tasks.
Jax transformations will be applied in bayesn_model.py, when the dataset is moving to
numpyro sampling. This module should stick with pythonic and numpy data structures.
The class's factory methods use the read_* methods from io.py, which convert data in
various formats to a standard intermediate representation.
The modules are separated to avoid bloat and improve unit testing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from io import StringIO
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import Any
from warnings import warn

from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import NamedTuple

from bayesn import io
from bayesn import utils, constants

########################
### Global constants ###
########################
meta_names: dict[str, list[str]] = {
    "str": ["snid", "field", "survey", "cutflag_snana"],
    "num": [
        "ra", "dec", "peak_mjd", "sn_type",
        "z_helio", "z_helio_err",
        "z_cmb", "z_cmb_err",
        "z_hubble", "z_hubble_err",
        "mwebv", "mwebv_err",
        "host_logmass", "host_logmass_err",
        "vpec", "vpec_err"],
    "sim": [f"sim_{s}s" for s in
        ("gentype", "template_id", "libid", "redshift_cmb", "vpec", "dlmag",
        "peakmjd", "theta", "AV", "RV")]
}
all_meta_names: list[str] = (
    meta_names["str"] + meta_names["num"] + meta_names["sim"]
)
# These defaults should not be used to instantiate SNDataset attributes, missing data
# should be represented with python Nones, both for string lists and np.arrays. The
# default_values are for making data products compatible with SNANA.
default_values: dict[str, str | Number] = {
    "field": "VOID",
    "survey": "NULL",
    "cutflag_snana": "NULL",
    "sn_type": 0,
    "z_helio_err": 5e-4,
    "z_cmb_err": 5e-4,
    "z_hubble_err": 5e-4,
    "mwebv": 0.0,
    "mwebv_err": 0.03,
    "host_logmass": -9.0,
    "host_logmass_err": -9.0,
    "vpec": 0.0,
    "vpec_err": 150.0,
}

# needs to be list for pandas column indexing.
req_phot_cols = ("snid", "flt", "mjd")


def default_arr_str() -> np.ndarray:
    return np.zeros(shape=(0,), dtype=float)


def default_arr_float() -> np.ndarray:
    return np.zeros(shape=(0,), dtype=float)


######################
### Global methods ###
######################
def get_standard_name(name: str) -> str:
    """
    Standarise a metadata or photometry column name.

    Maps SNANA, SNooPy, and legacy BayeSN column and metadata keys to canonical
    BayeSN names.

    Parameters
    ----------
    name
        Input key or column name to standarise.

    Returns
    -------
    Canonical BayeSN name.
    """
    if name in all_meta_names:
        return name
    # SNANA names are largely the same but capitalised though there are a few
    # exceptions which are handled explicitly.
    name = name.lower()
    if name in all_meta_names:
        return name
    misc = {
        # metadata names
        "zhel":                "z_hubble",
        "zhel_err":            "z_hubble_err",
        "zhd":                 "z_hubble",
        "zhd_err":             "z_hubble_err",
        "redshift_final":      "z_cmb",
        "redshift_final_err":  "z_cmb_err",
        "right_ascension":     "ra",
        "decl":                "dec",
        "declination":         "dec",
        "search_peakmjd":      "peak_mjd",
        "pkmjd":               "peak_mjd",
        "peakmjd":             "peak_mjd",
        "tmax":                "peak_mjd",
        "hostgal_logmass":     "host_logmass",
        "hostgal_logmass_err": "host_logmass_err",
        "vpecerr":             "vpec_err",
        "nepoch":              "N_obs",
        "nobs":                "N_obs",
        "n_obs":               "N_obs",
        "sim_dlmu":            "sim_dlmags",
        "sim_template_index":  "sim_template_ids",
        "sim_av":              "sim_AVs",
        "sim_rv":              "sim_RVs",
        "trestmin":            "phase_min",
        "trestmax":            "phase_max",
        # photometry columns
        "fluxcal":             "flux",
        "fluxcalerr":          "flux_err",
        "fluxcal_err":         "flux_err",
        "magnitude":           "mag",
        "magerr":              "mag_err",
        "band":                "flt",
    }
    if name in misc:
        return misc[name]
    if name.replace("redshift_", "z_") in all_meta_names:
        return name.replace("redshift_", "z_")
    if name.startswith("sim_") and name+"s" in all_meta_names:
        return name+"s"
    return name


def get_SNANA_name(name: str) -> str:
    """
    Map a canonical BayeSN name to the corresponding SNANA format name.

    Parameters
    ----------
    name
        Canonical BayeSN key or column name.

    Returns
    -------
    SNANA uppercase name.
    """
    name = name.upper()
    misc = {
        # metadata names
        "Z_CMB":            "REDSHIFT_FINAL",
        "Z_CMB_ERR":        "REDSHIFT_FINAL_ERR",
        "PEAK_MJD":         "SEARCH_PEAKMJD",
        "DEC":              "DECL",
        "HOST_LOGMASS":     "HOSTGAL_LOGMASS",
        "HOST_LOGMASS_ERR": "HOSTGAL_LOGMASS_ERR",
        "SIM_DLMAGS":       "SIM_DLMU",
        "SIM_TEMPLATE_IDS": "SIM_TEMPLATE_INDEX",
        # photometry columns
        "FLUX":             "FLUXCAL",
        "FLUX_ERR":         "FLUXCALERR",
        "MAG_ERR":          "MAGERR",
    }
    if name in misc:
        return misc[name]
    if name.startswith("Z_"):
        return name.replace("Z", "REDSHIFT")
    if name.startswith("SIM_"):
        return name.rstrip("S")
    if name not in [x.upper() for x in all_meta_names]:
        warn(UserWarning(f"Not sure what SNANA key {name} refers to, returning input."))
    return name


def clean_peakmjd_key(
    sn_dict: dict[str, str | Number | ArrayLike],
    peakmjd_key: str | list[str] | None = None,
) -> dict:
    """
    Standarise the peak MJD key in a metadata dictionary.

    The default method searches for standard keys ('search_peakmjd', 'peakmjd',
    'pkmjd', 'tmax'). Additional custom keys can be specified via `peakmjd_key`.
    The matched key is replaced with ``'peak_mjd'``.

    Parameters
    ----------
    sn_dict
        Metadata dictionary.
    peakmjd_key
        Custom key(s) to check for peak MJD in addition to defaults.

    Returns
    -------
    Dictionary with peak MJD standarised to key ``'peak_mjd'``.
    """
    if isinstance(peakmjd_key, str):
        peakmjd_key = [peakmjd_key]
    elif peakmjd_key is None:
        peakmjd_key = []
    default_keys = ["search_peakmjd", "peakmjd", "pkmjd", "tmax"]
    peakmjd_key = [key.lower() for key in peakmjd_key] + default_keys
    found_keys = [key for key in sn_dict if key.lower() in peakmjd_key]
    if len(found_keys) > 1:
        raise ValueError(
            f"Multiple peakmjd_keys found. The sn_dict contains {found_keys}, only "
            "one of which can be interpreted as the value for peak_mjd."
        )
    elif len(found_keys) == 1:
        sn_dict["peak_mjd"] = sn_dict.pop(found_keys[0])
    return sn_dict


def clean_sn_dict(sn_dict: dict[str, str | Number | ArrayLike]) -> dict:
    """
    Standardise metadata dictionary keys and coerce scalar values to 1D arrays.

    Parameters
    ----------
    sn_dict
        Metadata dictionary such as those produced by :mod:`bayesn.io` read functions.

    Returns
    -------
    Cleaned metadata dictionary with canonical keys and 1D numpy array values.
    """
    # REDSHIFT_CMB and REDSHIFT_FINAL (and errs) both map to z_cmb.
    # This can create a race condition. Instead, prioritise FINAL.
    if all(sn_dict.get(f"REDSHIFT_{key}") is not None for key in ("FINAL", "CMB")):
        sn_dict.pop("REDSHIFT_CMB")
    if all(sn_dict.get(f"REDSHIFT_{key}_ERR") is not None for key in ("FINAL", "CMB")):
        sn_dict.pop("REDSHIFT_CMB_ERR")
    # Standardising keys
    for key in list(sn_dict.keys()):
        sn_dict[get_standard_name(key)] = sn_dict.pop(key)
    # Sanitising 0D values.
    for key in sn_dict:
        if key in meta_names["str"] and (isinstance(sn_dict[key], str) or not hasattr(sn_dict[key], "__iter__")):
            sn_dict[key] = np.atleast_1d(sn_dict[key]).astype(str)
        elif key not in meta_names["str"]:
            sn_dict[key] = np.atleast_1d(sn_dict[key])
    # Adding standard keys as Nones if needed.
    keys_to_check = meta_names["str"] + meta_names["num"]
    if any([key.startswith("sim_") for key in sn_dict]):
        keys_to_check = all_meta_names
    for key in set(keys_to_check).difference(sn_dict.keys()):
        sn_dict[key] = np.full(len(sn_dict["snid"]), None)
    return sn_dict


def clean_obs_df(
    obs_df: pd.DataFrame,
    snids: str | ArrayLike | None = None,
    phot_idx: ArrayLike | None = None,
) -> pd.DataFrame:
    """
    Standardise column names and order in a photometry DataFrame.

    Ensures standard columns ('snid', 'flt', 'mjd', etc.) are present and properly typed.
    If the 'snid' column is missing, it is constructed using `snids` and `phot_idx`.

    Parameters
    ----------
    obs_df
        Input photometry DataFrame.
    snids : (N_sn,) or str, optional
        Supernova identifier(s).
    phot_idx : (N_sn + 1,), optional
        Photometry partition indices separating individual SNe.

    Returns
    -------
    Cleaned and sorted photometry DataFrame.
    """
    snids = np.atleast_1d(snids)
    if len(snids) == 0 or len(obs_df) == 0:
        return pd.DataFrame(columns=req_phot_cols)
    if "snid" not in obs_df:
        if len(snids) == 1 and phot_idx is None:
            phot_idx = np.array([0, len(obs_df)])
        elif phot_idx is None or len(phot_idx) == 1:
            raise TypeError("phot_idx cannot be inferred for multiple SNe.")
            assert len(phot_idx) - 1 == len(snids), "phot_idx length"
        N_obs = np.diff(phot_idx)
        snid_col = []
        for i,snid in enumerate(snids):
            snid_col.extend(np.full(N_obs[i], snids[i]))
        obs_df["snid"] = snid_col

    obs_df.columns = [get_standard_name(col) for col in obs_df.columns]
    data_cols, other_cols = [[] for _ in range(2)]
    for col in sorted(obs_df.columns):
        col = get_standard_name(col)
        if col in ("flux", "flux_err", "mag", "mag_err"):
            data_cols.append(col)
        elif col not in req_phot_cols:
            other_cols.append(col)
    for col in ("mjd", *data_cols):
        obs_df[col] = obs_df[col].astype(float)

    # Sorting
    if snids is not None:
        # Use ordering of snids rather than alphabetical.
        obs_df["snid"] = obs_df["snid"].astype("category").cat.set_categories(snids)
    obs_df = obs_df.sort_values(list(req_phot_cols)).reset_index(drop=True)
    return obs_df[[*req_phot_cols, *data_cols, *other_cols]]


class ObsData(NamedTuple):
    """
    Lightweight container for supernova observation data used by BayeSN.

    Parameters
    ----------
    host_logmass : (N_sn,)
        Host galaxy stellar mass in :math:`\\log_{10}(M_*/M_\\odot)`.
    z_hel : (N_sn,)
        Heliocentric redshift.
    z_hel_err : (N_sn,)
        Uncertainty in heliocentric redshift.
    muhat : (N_sn,)
        Apparent distance modulus from cosmology prior.
    MWEBV : (N_sn,)
        Milky Way E(B-V) color excess.
    mjd : (N_obs_max, N_sn)
        Modified Julian Dates (or rest-frame phases) of observations.
    flux : (N_obs_max, N_sn)
        Observed fluxes (or magnitudes).
    flux_err : (N_obs_max, N_sn)
        Flux (or magnitude) uncertainties.
    band_indices : (N_obs_max, N_sn)
        Integer bandpass index corresponding to each observation.
    mask : (N_obs_max, N_sn)
        Boolean mask where 1 indicates valid observed data and 0 indicates padding.
    """
    host_logmass: ArrayLike
    z_hel: ArrayLike
    z_hel_err: ArrayLike
    muhat: ArrayLike
    MWEBV: ArrayLike
    mjd: ArrayLike
    flux: ArrayLike
    flux_err: ArrayLike
    band_indices: ArrayLike
    mask: ArrayLike

@dataclass(eq=False)
class SNDataset:
    """
    Container for supernova light curve photometry and metadata.

    Supports reading, manipulating, filtering, and exporting datasets of one or more
    supernovae across multiple filters. Handles conversions between heliocentric and
    CMB frame redshifts, peculiar velocities, and rest-frame phases.

    Attributes
    ----------
    N_sn : int
        Number of supernovae in the dataset.
    fluxcal_zpt : float
        Standard photometric zeropoint (default 27.5).
    sim : bool
        Whether the dataset contains simulated supernova metadata.
    photometry : pd.DataFrame
        Table of photometric observations containing columns such as 'snid', 'flt',
        'mjd', 'flux', 'flux_err', 'mag', 'mag_err', and 'phase'.
    """
    #################################
    ### Attributes and Properties ###
    #################################
    # constants
    N_sn: int = 0
    fluxcal_zpt: float = 27.5
    sim: bool = False

    # string metadata arrays
    snid:          np.ndarray = dataclasses_field(default_factory=default_arr_str)
    # For SNANA compatibility
    field:         np.ndarray = dataclasses_field(default_factory=default_arr_str)
    survey:      np.ndarray = dataclasses_field(default_factory=default_arr_str)
    cutflag_snana: np.ndarray = dataclasses_field(default_factory=default_arr_str)

    # numeric metadata arrays
    ra:                np.ndarray = dataclasses_field(default_factory=default_arr_float)
    dec:               np.ndarray = dataclasses_field(default_factory=default_arr_float)
    peak_mjd:          np.ndarray = dataclasses_field(default_factory=default_arr_float)
    sn_type:           np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_helio:           np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_cmb:             np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_hubble:          np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_helio_err:       np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_cmb_err:         np.ndarray = dataclasses_field(default_factory=default_arr_float)
    z_hubble_err:      np.ndarray = dataclasses_field(default_factory=default_arr_float)
    mwebv:             np.ndarray = dataclasses_field(default_factory=default_arr_float)
    mwebv_err:         np.ndarray = dataclasses_field(default_factory=default_arr_float)
    host_logmass:      np.ndarray = dataclasses_field(default_factory=default_arr_float)
    host_logmass_err:  np.ndarray = dataclasses_field(default_factory=default_arr_float)
    vpec:              np.ndarray = dataclasses_field(default_factory=default_arr_float)
    vpec_err:          np.ndarray = dataclasses_field(default_factory=default_arr_float)
    # For SNANA compatibility
    sim_gentypes:      np.ndarray | None = None
    sim_template_ids:  np.ndarray | None = None
    sim_libids:        np.ndarray | None = None
    sim_redshift_cmbs: np.ndarray | None = None
    sim_vpecs:         np.ndarray | None = None
    sim_dlmags:        np.ndarray | None = None
    sim_peakmjds:      np.ndarray | None = None
    sim_thetas:        np.ndarray | None = None
    sim_AVs:           np.ndarray | None = None
    sim_RVs:           np.ndarray | None = None

    # Any additional metadata
    other_metadata: dict[str, ArrayLike] = dataclasses_field(default_factory=dict)

    # Expected columns: snid, flt, mjd, flux, flux_err, mag, mag_err, maybe t
    photometry: pd.DataFrame = dataclasses_field(default_factory=pd.DataFrame)

    def __eq__(self, other):
        try:
            for names in meta_names.values():
                for attr in names:
                    np.testing.assert_equal(getattr(self, attr), getattr(other, attr))
            pd.testing.assert_frame_equal(self.photometry, other.photometry)
            return True
        except:
            return False

    @property
    def meta_str(self) -> list[np.ndarray]:
        return [getattr(self, attr) for attr in meta_names["str"]]

    @property
    def meta_num(self) -> list[np.ndarray]:
        return [getattr(self, attr) for attr in meta_names["num"]]

    @property
    def meta_sim(self) -> list[np.ndarray | None]:
        return [getattr(self, attr) for attr in meta_names["sim"]]

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        meta_dict = dict(zip(
            meta_names["str"] + meta_names["num"], self.meta_str + self.meta_num
        ))
        if self.sim:
            meta_dict.update(dict(zip(meta_names["sim"], self.meta_sim)))
        meta_dict.update(self.other_metadata)
        return meta_dict

    @property
    def unique_bands(self) -> np.ndarray:
        return self.photometry["flt"].unique()

    @property
    def N_obs(self) -> np.ndarray:
        if not len(self.photometry):
            return np.zeros(self.N_sn)
        counts = self.photometry["snid"].value_counts()
        return np.array([counts.get(s, 0) for s in self.snid], dtype=int)

    @property
    def phot_idx(self) -> np.ndarray:
        if not len(self.photometry):
            return np.array([0])
        return np.append(0, np.cumsum(self.N_obs))

    @property
    def phase_min(self) -> np.ndarray:
        if "phase" not in self.photometry.columns:
            return np.full(self.N_sn, None)
        return self.photometry.groupby("snid", observed=True)["phase"].transform("min").unique()

    @property
    def phase_max(self) -> np.ndarray:
        if "phase" not in self.photometry.columns:
            return np.full(self.N_sn, None)
        return self.photometry.groupby("snid", observed=True)["phase"].transform("max").unique()

    ######################
    ### Initialisation ###
    ######################
    def __post_init__(self) -> None:
        self._clean_0d_metadata()
        self._clean_photometry()
        self._validate_dtypes()
        self._validate_lengths()
        self._validate_photometry()

    def _clean_0d_metadata(self):
        for attr in all_meta_names:
            val = getattr(self, attr)
            if val is None and attr in meta_names["sim"] and not self.sim:
                continue  # None is different from np.array([]) and np.array([None])
            if isinstance(val, None | str | Number):
                setattr(self, attr, np.full(self.N_sn, val))

    def _clean_photometry(self):
        self.photometry = clean_obs_df(self.photometry, self.snid, self.phot_idx)

    def _validate_dtypes(self) -> None:
        for attr in all_meta_names:
            if not self.sim and attr in meta_names["sim"]:
                continue
            assert isinstance(getattr(self, attr), np.ndarray), f"{attr} dtype"
        assert isinstance(self.photometry, pd.DataFrame), "photometry dtype"
        for key, val in self.other_metadata.items():
            assert isinstance(val, np.ndarray), f"other_metadat[{key}] dtype"

    def _validate_lengths(self) -> None:
        for attr in all_meta_names:
            if not self.sim and attr in meta_names["sim"]:
                continue
            assert getattr(self, attr).shape[0] == self.N_sn, f"{attr} length"
        assert len(self.N_obs) == self.N_sn, "N_obs length"
        assert len(self.photometry) == sum(self.N_obs), f"photometry length"
        for key, val in self.other_metadata.items():
            assert len(val) == self.N_sn, f"other_metadata[{key}] length"

    def _validate_photometry(self) -> None:
        phot = self.photometry
        dups = phot.duplicated(["flt", "mjd"], keep=False)
        assert not any(dups), phot[dups]
        if "flux_err" in phot:
            assert all(phot["flux_err"] > 0)
        if "mag_err" in phot:
            assert all(phot["mag_err"] > 0)
        if "mag" in phot and "flux" in phot:
            mask = phot["flux"] > 0
            data_zps = phot["mag"][mask] + 2.5*np.log10(phot["flux"][mask])
            np.testing.assert_allclose(self.fluxcal_zpt, data_zps)

    def _validate_other_dtypes(self, sn_dict: None | dict[str, ArrayLike] = None, obs_df: None | pd.DataFrame = None):
        if sn_dict is not None:
            for attr in sn_dict:
                assert isinstance(sn_dict[attr], np.ndarray), f"{attr} dtype"
        if obs_df is not None:
            assert isinstance(obs_df, pd.DataFrame), "photometry dtype"

    def _validate_other_lengths(self, sn_dict: dict[str, ArrayLike], obs_df: None | pd.DataFrame = None):
        N_sn = len(sn_dict["snid"])
        for attr in sn_dict:
            assert len(sn_dict[attr]) == N_sn, attr
        if obs_df is not None:
            assert len(obs_df["snid"].unique()) == N_sn

    ######################
    ### Getter methods ###
    ######################
    def get_idx(self, snid: str | ArrayLike) -> int | np.ndarray:
        """
        Get the integer index or indices corresponding to one or more supernova IDs.

        Parameters
        ----------
        snid : (N_query,) or str
            Supernova identifier(s) to locate in the dataset.

        Returns
        -------
        Integer index for a single SNID, or numpy array of indices for multiple SNIDs.
        """
        if isinstance(snid, str):
            idx = np.where(self.snid == snid)[0]
            if not len(idx):
                raise ValueError(f"snid {snid} not found.")
            return idx[0]
        elif hasattr(snid, "__iter__"):
            # Recursive for ArrayLike snids, could probably be more efficient...
            return np.array([self.get_idx(s) for s in snid])
        else:
            raise TypeError(
                f"snid of type {type(snid)} is not supported. Please pass a string or "
                "ArrayLike argument."
            )

    def _parse_snid_idx_args(
        self,
        snid: str | ArrayLike | None = None,
        idx: int | ArrayLike | None = None,
    ) -> int | np.ndarray:
        """
        Sanitise and unify SNID or integer index arguments.

        Parameters
        ----------
        snid : (N_query,) or str, optional
            Supernova identifier(s).
        idx : (N_query,) or int, optional
            Integer supernova index or indices.

        Returns
        -------
        Sanitised supernova index or array of indices.
        """
        if snid is None and idx is None:
            raise ValueError("Either snid or idx should be specified.")
        elif snid is None and idx is not None:
            return idx
        elif snid is not None and idx is None:
            return self.get_idx(snid)
        else:
            if self.get_idx(snid) == idx:
                return idx
            raise ValueError("Either snid or idx should be specified, not both.")

    def get_metadata_subset(
        self,
        snid: str | ArrayLike | None = None,
        idx: int | ArrayLike | None = None,
        use_defaults: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Get metadata for a subset of supernovae.

        Parameters
        ----------
        snid : (N_query,) or str, optional
            Supernova identifier(s).
        idx : (N_query,) or int, optional
            Integer supernova index or indices.
        use_defaults
            If True, replace metadata stored as None with standard defaults.

        Returns
        -------
        Dictionary of metadata arrays filtered to the selected supernovae.
        """
        if snid is None and idx is None:
            idx = np.arange(self.N_sn)
        else:
            idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx))
        metadata_subset = copy.deepcopy(self.metadata)
        defaults = default_values if use_defaults else {}
        for key, val in metadata_subset.items():
            metadata_subset[key] = np.array([
                defaults.get(key) if val[i] is None else val[i] for i in idx
            ])
        return metadata_subset

    def get_phot_subset(
        self,
        snid: str | ArrayLike | None = None,
        idx: int | ArrayLike | None = None,
    ) -> pd.DataFrame:
        """
        Get photometry for a subset of supernovae.

        Parameters
        ----------
        snid : (N_query,) or str, optional
            Supernova identifier(s).
        idx : (N_query,) or int, optional
            Integer supernova index or indices.

        Returns
        -------
        Photometry table filtered to the selected supernovae.
        """
        idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx))
        # isin will not preserve index ordering.
        phot_subset = pd.concat([self.photometry[self.photometry["snid"] == self.snid[i]] for i in idx])
        return phot_subset.reset_index(drop=True)

    #####################
    ### Data Addition ###
    #####################
    def append(
        self,
        ds: SNDataset | None = None,
        sn_dict: dict[str, Any] | None = None,
        obs_df: pd.DataFrame | None = None,
        phot_idx: ArrayLike | None = None,
    ) -> None:
        """
        Append an SNDataset instance, or metadata dict and photometry DataFrame.

        Parameters
        ----------
        ds
            Dataset instance to append.
        sn_dict
            Metadata dictionary (e.g. from :mod:`bayesn.io` read functions).
        obs_df
            Photometry table.
        phot_idx : (N_sn + 1,), optional
            Partition indices for the photometry table.
        """
        # Arg parsing
        if ds is None and sn_dict is None and obs_df is None:
            return
        if ds is not None and (sn_dict is not None and obs_df is not None):
            try:
                utils.assert_dicts_match(ds.metadata, sn_dict)
                pd.testing.assert_frame_equal(ds.photometry, obs_df)
            except AssertionError:
                raise ValueError(
                    "The provided arguments are not equivalent. You can either provide "
                    "a ds with metadata matching sn_dict and photometry matching "
                    "obs_df, or provide either ds or sn_dict and obs_df."
                )
        elif ds is not None:
            sn_dict, obs_df, phot_idx = ds.metadata, ds.photometry, ds.phot_idx

        # Arg sanitisation
        sn_dict = clean_sn_dict(sn_dict)
        if phot_idx is None and len(sn_dict["snid"]) == 1:
            phot_idx = np.array([0, len(obs_df)])
        obs_df = clean_obs_df(obs_df, sn_dict["snid"], phot_idx)
        self._validate_other_dtypes(sn_dict, obs_df)
        self._validate_other_lengths(sn_dict, obs_df)

        # Sort into duplicate/new and append
        all_snids = set(sn_dict["snid"])
        common_snids = all_snids.intersection(self.snid)
        new_snids = all_snids.difference(self.snid)
        # for snids, append_fn in zip(
        #     (common_snids, new_snids),
        #     (self._append_duplicate, self._append_new)
        # ):
        #     idx = np.where(np.in1d(all_snids, snids))[0]
        #     meta = {key: val[idx] for key, val in sn_dict.items()}
        #     phot = obs_df[obs_df["snid"].isin(snids)]
        #     append_fn(meta, phot)

        for i, snid in enumerate(sn_dict["snid"]):
            meta = copy.deepcopy(sn_dict)
            phot = obs_df[obs_df["snid"] == snid]
            for key, val in meta.items():
                meta[key] = np.array([val[i]])
            if snid in self.snid:
                self._append_duplicate(meta, phot)
            else:
                self._append_new(meta, phot)

        # Add Nones for missing other_metadata
        for attr in set(self.other_metadata.keys()).difference(sn_dict.keys()):
            self.other_metadata[attr] = np.append(
                self.other_metadata[attr],
                np.full(len(new_snids), None)
            )


    def _append_new(self, sn_dict: dict, obs_df: pd.DataFrame) -> None:
        """ This private method should be accessed through the append method for input
        sanitisation and error checking. The given sn_dict and obs_df are used to append
        data for SNe not currently included in the SNDataset (by snid).
        """
        N_sn = len(sn_dict["snid"])
        for attr in meta_names["str"] + meta_names["num"]:
            sn_dict_arr = sn_dict.get(attr, np.full(N_sn, None))
            setattr(self, attr, np.append(getattr(self, attr), sn_dict_arr))
        if self.sim or sn_dict.get("sim_gentypes") is not None:
            self.sim = True
            for attr in meta_names["sim"]:
                # If coming from sim=False, need an array of Nones for existing data
                if getattr(self, attr) is None:
                    setattr(self, attr, np.full(self.N_sn, None))
                sn_dict_arr = sn_dict.get(attr, np.full(N_sn, None))
                setattr(self, attr, np.append(getattr(self, attr), sn_dict_arr))

        other_keys = [key for key in sn_dict if key not in all_meta_names]
        for key in other_keys:
            if key in self.other_metadata:
                self.other_metadata[key] = np.append(self.other_metadata[key], sn_dict[key], axis=0)
            else:
                self.other_metadata[key] = np.append(np.full((self.N_sn, *sn_dict[key].shape[1:]), None), sn_dict[key], axis=0)
        self.N_sn += N_sn
        if len(self.photometry):
            self.photometry = pd.concat([self.photometry, obs_df], ignore_index=True)
        else:
            self.photometry = obs_df

    def _append_duplicate(self, sn_dict: dict, obs_df: pd.DataFrame) -> None:
        """ This private method should be accessed through the append method for input
        sanitisation and error checking. The given sn_dict and obs_df are used to append
        any new data for SNe that are already in the SNDataset (by snid).
        """
        # Add new other_metadata keys if not already present.
        for other_key in set(sn_dict.keys()).difference(all_meta_names):
            if other_key not in self.other_metadata:
                self.other_metadata[other_key] = np.full(self.N_sn, None)
        # compare metadata to make sure numeric values are in agreement.
        indices = self.get_idx(sn_dict["snid"])
        meta = self.get_metadata_subset(idx=indices)
        # some metadata should be allowed to vary.
        for key in ("survey", "N_obs"):
            meta.pop(key, None)
            sn_dict.pop(key, None)
        utils.assert_dicts_match(meta, sn_dict, flag_missing_data=False)
        # Replace Nones in metadata if sn_dict has actual values.
        for key, val in sn_dict.items():
            val = np.array(val)
            new_data_idx = np.where(
                (meta[key] == None) & (val != None)
            )[0]
            if not len(new_data_idx):
                continue
            if key in all_meta_names:
                new_meta = getattr(self, key)
            else:
                new_meta = self.other_metadata[key]
            new_meta[indices[new_data_idx]] = val[new_data_idx]
            if key in all_meta_names:
                setattr(self, key, new_meta)
            else:
                self.other_metadata[key] = new_meta

        merged_phot = self.photometry.merge(
            obs_df, on=list(self.photometry.columns), how="outer"
        )
        dups = merged_phot.duplicated(subset=req_phot_cols, keep=False)
        if any(dups):
            raise ValueError(
                "There are discrepancies in the data, where observations in the "
                "same flt at the same mjd have different photometry.\n"
                f"{merged_phot[dups]}"
            )
        self.photometry = merged_phot.sort_values(list(req_phot_cols)).reset_index(drop=True)

    ####################
    ### Data Removal ###
    ####################
    def remove_sn(
        self,
        snid: str | ArrayLike | None = None,
        idx: int | ArrayLike | None = None,
    ) -> None:
        """
        Remove all photometry and metadata for specified supernovae.

        Parameters
        ----------
        snid : (N_query,) or str, optional
            Supernova identifier(s) to remove.
        idx : (N_query,) or int, optional
            Integer index or indices of supernovae to remove.
        """
        idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx))
        if not len(idx):
            return
        snids = self.snid[idx]
        attrs_to_prune = meta_names["str"] + meta_names["num"]
        if self.sim:
            attrs_to_prune += meta_names["sim"]
        self.photometry = self.photometry[~self.photometry["snid"].isin(snids)].reset_index(drop=True)
        for attr in attrs_to_prune:
            setattr(self, attr, np.delete(getattr(self, attr), idx))
        for key in self.other_metadata:
            self.other_metadata[key] = np.delete(self.other_metadata[key], idx)
        self.N_sn -= len(idx)

    def keep_according_to_list(self, keep_list: list[str]) -> None:
        """
        Retain only supernovae with SNIDs in `keep_list`, dropping all others.

        Parameters
        ----------
        keep_list
            Supernova IDs to keep.
        """
        self.remove_sn(snid=set(self.snid).difference(keep_list))

    def remove_phot_by_idx(self, idx: int | ArrayLike | pd.Index, inplace: bool = True) -> None:
        """
        Remove photometry at specified table row indices.

        Parameters
        ----------
        idx : (N_rows,) or int
            Row indices in the photometry DataFrame to remove.
        inplace
            Whether to modify this dataset in place.
        """
        self.photometry = self.photometry.drop(idx).reset_index(drop=True)
        sn_indices = np.searchsorted(self.phot_idx, np.atleast_1d(idx), side="right")
        for sn_idx in np.unique(sn_indices):
            self.phot_idx[sn_idx:] -= list(sn_indices).count(sn_idx)
        # Reverse order to preserve index order before the one being removed.
        self.remove_sn(idx=np.where(self.N_obs == 0)[0])

    def drop_bands(self, drop_bands: list[str] | ArrayLike) -> None:
        """
        Remove observations observed in specified bandpasses.

        Parameters
        ----------
        drop_bands : (N_bands,) or list of str
            Bandpass names to remove.
        """
        self.remove_phot_by_idx(np.where(self.photometry["flt"].isin(drop_bands))[0])

    def drop_by_band_lims(
        self,
        band_lim_dict: dict[str, tuple[float, float]],
        wave_min: float,
        wave_max: float,
        ignore_missing: bool = False,
        rest_frame: bool = True,
        z_err: ArrayLike | str | None = None,
    ) -> None:
        """
        Drop bands whose redshifted transmission extends outside a wavelength range.

        Parameters
        ----------
        band_lim_dict
            Mapping from BayeSN bandpass name to 2-tuple observer-frame wavelength limits.
        wave_min
            Minimum rest-frame wavelength to retain.
        wave_max
            Maximum rest-frame wavelength to retain.
        ignore_missing
            Whether to skip bands in the dataset that are missing from `band_lim_dict`.
        rest_frame
            Whether the band_lim_dict limits should be converted to rest-frame first.
        z_err
            Redshift uncertainties that can be included to make sure sampling z_helio
            does not push the redshifted transmission outside the wavelength range.
            If None error will be assumed to be irrelevant (i.e. z will not be sampled).
            If 'default', the SNDataset.z_helio_err array will be used as z_err.
            If a number or 1D array with length N_sn, that will be treated as a normal
            standard deviation and photometry will be dropped if they come within 3
            times the uncertainty of the limits. If a 2D array with shape (N_sn, ...),
            the second axis will be assumed to be quantiles evenly spanning 0 to 1.
            Each SN's minimum and maximum values will be checked.
        """
        bands = list(band_lim_dict.keys())
        missing = set(self.unique_bands).difference(bands)
        if len(missing) and not ignore_missing:
            raise ValueError(
                "The data contain a set of bandpasses not found in band_lim_dict: "
                f"{missing}. These bands cannot be checked against redshift. If you "
                "want to proceed without checking these bands, you can set the "
                "'ignore_missing' argument to True and run this method again."
            )
        if z_err is None:
            z_err = np.zeros(self.N_sn)
        elif isinstance(z_err, str) and z_err == "default":
            z_err = self.z_helio_err
        elif isinstance(z_err, Number):
            z_err = np.full(self.N_sn, z_err)
        if z_err.ndim == 1:
            z_lo = self.z_helio - 3*z_err
            z_hi = self.z_helio + 3*z_err
        elif z_err.ndim == 2:
            z_lo, z_hi = z_err[:, 0], z_err[:, -1]
        else:
            raise ValueError(
                "Unrecognised z_err value. Should be None, 'default', an array of "
                "errors with shape (N_sn,), or an array of quantiles with shape "
                "(N_sn, N_quantiles)."
            )
        if not rest_frame:
            z_lo, z_hi = np.zeros((2, self.N_sn))
        # shape (N_bands, N_sn)
        blue_lim_arr = np.array([val[0]/(1+z_hi) for val in band_lim_dict.values()])
        red_lim_arr = np.array([val[1]/(1+z_lo) for val in band_lim_dict.values()])
        band_indices, sn_indices = np.where((blue_lim_arr < wave_min) | (red_lim_arr > wave_max))
        rm_phot_idx = []
        for sn_idx in set(sn_indices):
            sn_bands = np.array(bands)[band_indices[np.where(sn_indices == sn_idx)[0]]]
            sn_phot = self.get_phot_subset(idx=sn_idx)
            rm_phot_idx += list(sn_phot[sn_phot["flt"].isin(sn_bands)].index + self.phot_idx[sn_idx])
        self.remove_phot_by_idx(rm_phot_idx)

    def cut_by_meta_numeric(
        self,
        name: str,
        logic: str,
        val: Number,
        inplace: bool = True,
        use_defaults: bool = False
    ) -> tuple[dict[str, np.ndarray], pd.DataFrame] | None:
        """
        Filter supernovae based on numeric metadata cuts.

        Parameters
        ----------
        name
            Metadata attribute name to filter by (or 'N_obs', 'snrmaxN').
        logic
            Comparison operator ('=', '<', '<=', '>', '>=').
        val
            Threshold value for comparison.
        inplace
            Whether to remove the cut SNe in place. If False, returns (meta_dict, phot_df).
        use_defaults
            Whether to substitute missing values with standard defaults when evaluating.

        Returns
        -------
        (metadata_subset, phot_subset) if `inplace=False`, otherwise None.
        """
        old_name = name
        name = get_standard_name(old_name)
        defaults = default_values if use_defaults else {}
        if hasattr(self, name) and len(getattr(self, name)) == self.N_sn:
            meta_arr = [defaults[name] if x is None else x for x in getattr(self, name)]
        elif name.startswith("snrmax"):
            idx = int(name.strip("snrmax"))
            meta_arr = [self.calculate_snrmaxes(snid, N=idx)[idx-1] for snid in self.snid]
        else:
            raise ValueError(
                f"{old_name} not recognised. Valid options are {meta_names['num']}, "
                f"{meta_names['sim']}, N_obs, or SNRMAXN for int N."
            )
        drop = utils.where_logic(arr=meta_arr, val=val, logic=logic)[0]
        if not inplace:
            keep = list(set(np.arange(self.N_sn)).difference(drop))
            meta = self.get_metadata_subset(idx=keep, use_defaults=use_defaults)
            phot = self.get_phot_subset(idx=keep)
            return meta, phot
        self.remove_sn(idx=drop)

    def cut_by_phot_numeric(
        self, name: str, logic: str, val: Number, inplace: bool = True
    ) -> tuple[dict[str, np.ndarray], pd.DataFrame] | None:
        """
        Filter observations based on numeric photometry cuts.

        Parameters
        ----------
        name
            Photometry column name to filter.
        logic
            Comparison operator ('=', '<', '<=', '>', '>=').
        val
            Threshold value for comparison.
        inplace
            Whether to remove rows in place. If False, returns (meta_dict, phot_df).

        Returns
        -------
        (metadata_subset, phot_subset) if `inplace=False`, otherwise None.
        """
        if name not in self.photometry:
            raise ValueError(
                f"{name} not recognised. "
                f"Valid options are {self.photometry.columns}."
            )
        arr = self.photometry[name]
        drop = utils.where_logic(arr=self.photometry[name], val=val, logic=logic)[0]
        if not inplace:
            phot_copy = copy.deepcopy(self.photometry).drop(drop).reset_index(drop=True)
            idx = self.get_idx(snid=phot_copy["snid"].unique())
            return self.get_metadata_subset(idx=idx), phot_copy
        self.remove_phot_by_idx(drop)


    ###################################
    ### Astronomical Getter Methods ###
    ###################################
    # These methods should not alter state, instead returning useful transformations
    # of the metadata or photometry.
    def calculate_snrmaxes(
        self,
        snid: str,
        N: int = 3,
        default_value: float = -99.0,
    ) -> list[float]:
        """
        Calculate maximum signal-to-noise ratios in top N unique bands.

        Parameters
        ----------
        snid
            Supernova identifier.
        N
            Number of top bands to calculate SNR maxima for.
        default_value
            Padding value when fewer than N unique bands exist.

        Returns
        -------
        Top N maximum SNR values in descending order.
        """
        assert isinstance(snid, str), "Non-str snids are not supported."
        phot = self.get_phot_subset(snid=snid)
        snr = phot["flux"]/phot["flux_err"]
        snrmaxes = [snr.max()]
        for i in range(1,N):
            phot = phot[phot["flt"] != phot["flt"][snr.idxmax()]]
            if not len(phot):
                return snrmaxes + [default_value for _ in range(N-i)]
            snr = phot["flux"] / phot["flux_err"]
            snrmaxes.append(snr.max())
        return snrmaxes

    def estimate_tmax(self, snid: str) -> float:
        """
        Estimate time of maximum using SNR^2-weighted average of observation epochs.

        Parameters
        ----------
        snid
            Supernova identifier.

        Returns
        -------
        Estimated time of B-band maximum in MJD.
        """
        assert isinstance(snid, str), "Non-str snids are not supported."
        phot = self.get_phot_subset(snid)
        tmax = utils.SNR_power_weighted_ave(
            phot["mjd"],
            power=2,
            flux=phot["flux"],
            flux_err=phot["flux_err"]
        )
        return float(tmax)


    def calculate_rest_phases(self, snid: str, peak_mjd: Number | None = None) -> np.ndarray:
        """
        Calculate rest-frame phases for a given supernova.

        Parameters
        ----------
        snid
            Supernova identifier.
        peak_mjd, optional
            Estimated peak MJD. If None, uses supernova's `peak_mjd` metadata or
            estimates it via :meth:`estimate_tmax`.

        Returns
        -------
        phases : (N_obs,)
            Rest-frame phases in days relative to maximum.
        """
        assert isinstance(snid, str), "Non-str snids are not supported."
        idx = self.get_idx(snid)
        if peak_mjd is None:
            peak_mjd = self.peak_mjd[idx] or self.estimate_tmax(self.snid[idx])
        epochs = self.photometry["mjd"][self.phot_idx[idx]:self.phot_idx[idx+1]].to_numpy()
        return (epochs - peak_mjd) / (1.0 + self.z_helio[idx])

    def get_band_indices(
        self,
        band_dict: dict[str, int] | None = None,
        photometry: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Get integer indices of the filter band column.

        Parameters
        ----------
        band_dict, optional
            Mapping from filter names to integer indices.
        photometry, optional
            Photometry table to map. If None, uses `self.photometry`.

        Returns
        -------
        band_indices : (N_obs,)
            1D array mapping recorded bands to integer indices.
        """
        if photometry is None:
            photometry = self.photometry
        if band_dict is None:
            band_dict = dict(zip(self.unique_bands, range(1, len(self.unique_bands)+1)))
            band_dict["NULL_BAND"] = 0
        else:
            missing = set(self.unique_bands).difference(band_dict)
            if missing:
                start = max(band_dict.values()) + 1
                missing_dict = dict(zip(missing, range(start, start+len(missing))))
                warn(UserWarning(
                    "The provided band_dict does not cover the following bands in "
                    f"unique_bands: {missing}. They will be assigned indices as "
                    f"{missing_dict}."
                ))
                band_dict.update(missing_dict)
        return photometry["flt"].apply(lambda x: band_dict[x]).to_numpy()

    ###################################
    ### Astronomical Setter Methods ###
    ###################################
    # These methods alter state
    def fill_out_redshifts(self) -> None:
        """
        Infer missing redshift metadata by transforming between rest frames.

        Propagates between heliocentric and CMB frames and accounts for peculiar velocities.
        """
        attrs = ("ra", "dec", "z_helio", "z_cmb", "z_hubble", "z_helio_err",
                "z_cmb_err", "z_hubble_err", "vpec", "vpec_err")
        for i in range(self.N_sn):
            ra, dec, zhel, zcmb, zhub, dzhel, dzcmb, dzhub, v, dv = [getattr(self, x)[i] for x in attrs]
            has_coords = ra is not None and dec is not None
            if v is not None:
                v = v/constants.C_LIGHT
            if dv is not None:
                dv = dv/constants.C_LIGHT

            # Splitting the convert_z calls to an error and value call is half as
            # efficient, but avoids overwriting data in some edge cases. For example,
            # if zhel was calculated with a different CMB frame but does not include
            # an error term, and zcmb and dzcmb are available, dzhel will be calculated
            # while zhel will not be overwritten.
            if dzcmb is None and None not in (zhel, dzhel, ra, dec):
                dzcmb = self.z_cmb_err[i] = utils.convert_z(zhel, ra, dec, "hel", dzhel)[1]
            if zcmb is None and None not in (zhel, ra, dec):
                zcmb = self.z_cmb[i] = utils.convert_z(zhel, ra, dec, "hel")

            # zhel -> zcmb takes priority over zhub -> zcmb
            # There is no error checking if the two do not produce the same zcmb.
            if dzcmb is None and None not in (zhub, dzhub, v):
                dz_pv = dzhub*(1+v)
                z_dpv = 0 if dv is None else dv*(1+zhub)
                dzcmb = self.z_cmb_err[i] = np.hypot(dz_pv, z_dpv)
            if zcmb is None and None not in (zhub, v):
                zcmb = self.z_cmb[i] = (1+zhub)*(1+v) - 1


            # Populating zcmb first allows for potential z_hel <-> zhub.
            if dzhel is None and None not in (zcmb, dzcmb, ra, dec):
                self.z_helio_err[i] = utils.convert_z(zcmb, ra, dec, "cmb", dzcmb)[1]
            if zhel is None and None not in (zcmb, ra, dec):
                self.z_helio[i] = utils.convert_z(zcmb, ra, dec, "cmb")

            if dzhub is None and None not in (zcmb, dzcmb, v):
                dz_pv = dzcmb/(1+v)
                z_dpv = 0 if dv is None else -dv*(1+zcmb)/(1+v)**2
                self.z_hubble_err[i] = np.hypot(dz_pv, z_dpv)
            if zhub is None and None not in (zcmb, v):
                self.z_hubble[i] = (1+zcmb)/(1+v) - 1

    def set_all_rest_phases(self) -> None:
        """Calculate and store rest-frame phases for all supernovae in ``self.photometry['phase']``."""
        self.photometry["phase"] = np.concatenate([self.calculate_rest_phases(snid) for snid in self.snid])

    def recalibrate_fluxcal_zpt(self) -> None:
        """
        Rescale fluxes and flux errors to match `fluxcal_zpt`.

        Assumes observed magnitudes are correct and adjusts fluxes such that
        :math:`m = {\\rm zp} - 2.5 \\log_{10}(f)`.
        """
        pos_mask = self.photometry["flux"] > 0
        if not pos_mask.any():
            return
        data_zps = (
            self.photometry.loc[pos_mask, "mag"].astype(float)
            + 2.5 * np.log10(self.photometry.loc[pos_mask, "flux"].astype(float))
        )
        if np.isclose(np.std(data_zps), 0):
            # Assume all same zp and apply scaling to all fluxes/errs.
            pos_mask = pd.Series(np.ones(len(pos_mask)), dtype=bool)
            data_zps = np.median(data_zps)
        flux_scaling = 10 ** (0.4 * (self.fluxcal_zpt - data_zps))
        self.photometry.loc[pos_mask, "flux"] *= flux_scaling
        self.photometry.loc[pos_mask, "flux_err"] *= flux_scaling

    def apply_filter_map(self, map_dict: dict[str, str]) -> None:
        """
        Map non-standard filter names to canonical BayeSN filter names.

        Parameters
        ----------
        map_dict
            Dictionary mapping data filter names to BayeSN filter names.
        """
        self.photometry["flt"] = self.photometry["flt"].map(lambda f: map_dict.get(f, f))

    def apply_error_floor(self, error_floor: float) -> None:
        """
        Enforce a minimum photometric error floor in magnitudes.

        Parameters
        ----------
        error_floor
            Error floor in magnitudes.
        """
        if error_floor <= 0:
            return
        min_err = error_floor * (np.log(10) / 2.5) * self.photometry["flux"].to_numpy()
        self.photometry["flux_err"] = np.maximum(self.photometry["flux_err"].to_numpy(), min_err)
        self.photometry["mag_err"] = np.maximum(
            self.photometry["mag_err"].to_numpy(),
            np.full_like(self.photometry["mag_err"], error_floor)
        )

    def apply_overrides(self, overrides: dict | None = None) -> None:
        """
        Apply metadata overrides to the dataset.

        Parameters
        ----------
        overrides, optional
            Dictionary mapping metadata attribute names to override values.
        """
        overrides = overrides if overrides is not None else {}
        # Ensure overrides map to attributes.
        for key in list(overrides.keys()):
            std_name = get_standard_name(key)
            if hasattr(self, std_name):
                overrides[std_name] = overrides.pop(key)
            else:
                del overrides[key]

        # Support 0D broadcasting to N_sn.
        for key, val in overrides.items():
            if isinstance(val, Number | str | None):
                overrides[key] = np.full(self.N_sn, val)

        # If a redshift/err is overridden, set other redshifts/errs to None if they can
        # be recalculated with the overridden redshift via fill_out_redshifts.
        redshift_types = {"z_helio", "z_cmb", "z_hubble"}
        err_types = {f"{z_type}_err" for z_type in redshift_types}
        overridden_redshifts = set(overrides.keys()).intersection(redshift_types)
        overridden_errs = set(overrides.keys()).intersection(err_types)
        if overrides.get("ra") is not None and overrides.get("dec") is not None:
            has_coords = (overrides["ra"] != None) & (overrides["dec"] != None)
        else:
            has_coords = (self.ra != None) & (self.dec != None)
        if overrides.get("vpec") is not None:
            has_vpec = overrides["vpec"] != None
        else:
            has_vpec = (self.vpec != None) | ("vpec" in overrides)
        for z_err in overridden_errs:
            z_type = z_err[:-4]
            if z_type in overridden_redshifts:
                continue
            other_overrides = overridden_redshifts.difference({z_type})
            if other_overrides:
                warn(UserWarning(
                    f"{z_err} was overridden, but {z_type} was not. Other redshifts "
                    "have been overridden {other_overrides}, but their errors (or lack "
                    f"thereof) will not propagate to {z_err}."
                ))
        for z_type in overridden_redshifts:
            z_err = f"{z_type}_err"
            if z_err not in overridden_errs:
                warn(UserWarning(
                    f"{z_type} was overridden, but {z_err} was not. The redshift will "
                    "be treated as having no error, even if values exist in the files."
                ))
                overrides[z_err] = np.full(self.N_sn, None)

        # Wiping attributes allows other overrides to take priority.
        if "z_helio" in overridden_redshifts:
            self.z_cmb = np.where(has_coords, None, self.z_cmb)
            self.z_cmb_err = np.where(has_coords, None, self.z_cmb_err)
            self.z_hubble = np.where(has_coords & has_vpec, None, self.z_hubble)
            self.z_hubble_err = np.where(has_coords & has_vpec, None, self.z_hubble_err)
        if "z_cmb" in overridden_redshifts:
            self.z_helio = np.where(has_coords, None, self.z_helio)
            self.z_helio_err = np.where(has_coords, None, self.z_helio_err)
            self.z_hubble = np.where(has_vpec, None, self.z_hubble)
            self.z_hubble_err = np.where(has_vpec, None, self.z_hubble_err)
        if "z_hubble" in overridden_redshifts:
            self.z_helio = np.where(has_coords & has_vpec, None, self.z_helio)
            self.z_helio_err = np.where(has_coords & has_vpec, None, self.z_helio_err)
            self.z_cmb = np.where(has_vpec, None, self.z_cmb)
            self.z_cmb_err = np.where(has_vpec, None, self.z_cmb_err)

        for key, val in overrides.items():
            setattr(self, key, val)

    #######################
    ### Factory methods ###
    #######################
    # These class methods should return an instance of SNDataset
    @classmethod
    def from_ascii_files(
        cls,
        fname: str | Path | StringIO | ArrayLike,
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: list[str] | str | None = None,
        file_format: str | ArrayLike = "SNANA",
        overrides: dict | None = None,
        jobid: int = 1,
        njobtot: int = 1,
        sim_prescale: int = 1,
        **kwargs,
    ) -> SNDataset:
        """
        Instantiate an SNDataset from one or more ASCII files.

        Parameters
        ----------
        fname
            The file to be read (string path, Path object, or file-like text-stream),
            or an ArrayLike of files to be read.
        fluxcal_zpt
            The common zeropoint used for converting between magnitudes and fluxes.
        peakmjd_key
            Custom key(s) to check for peak MJD.
        file_format
            The kind of files to be read in ('SNANA' or 'snpy').
        overrides
            Metadata overrides.
        jobid
            The jobid and njobtot arguments can be used to parallelise the workflow of
            this method. Each method call will create a SNDataset with ~1/njobtot of
            all files if fname is an ArrayLike. The subset ingested is based on
            (SN_idx - jobid) % njobtot == 0, such that njobtot consecutive jobids will
            ensure every SN index is processed once and only once.
        njobtot
            See jobid above.
        sim_prescale
            Prescale factor for simulation files.
        **kwargs
            Additional keyword arguments directed to the read function (e.g. ``comment``, ``tablename``).

        Returns
        -------
        Loaded supernova dataset.
        """
        if overrides is None:
            overrides = {}
        # Support non-array values.
        if isinstance(fname, str | Path):
            fname = [fname,]
        sim = False
        if (isinstance(file_format, str) and file_format.lower() == "snana") or file_format[0].lower() == "snana":
            meta = io.read_snana_ascii_meta(fname[0])
            sim = "SIM_REDSHIFT_HELIO" in meta
            if sim:
                njobtot *= sim_prescale
        use_in_run = np.where((np.arange(len(fname)) + 1 - jobid) % njobtot == 0)
        fname = np.array(fname)[use_in_run]
        if isinstance(file_format, str):
            file_format = np.full_like(fname, file_format)
        else:
            file_format = np.array(file_format)[use_in_run]

        # Instantiate SNDataset then loop over files and append them, not the most
        # efficient so may need to look for other solutions.
        ds = SNDataset(sim=sim)
        for f, fmt in zip(fname, file_format):
            if fmt.lower() == "snana":
                read_fn = io.read_snana_ascii
            elif fmt.lower() in ("snpy", "snoopy"):
                read_fn = io.read_snpy
            else:
                raise ValueError(
                    f"{fmt} in file_format {file_format} not recognised. Use snana, "
                    "snpy, or snoopy, all case-insensitive."
                )
            sn_dict, obs_df = read_fn(fname=f, fluxcal_zpt=fluxcal_zpt, **kwargs)
            sn_dict = clean_peakmjd_key(sn_dict, peakmjd_key)
            ds.append(sn_dict=sn_dict, obs_df=obs_df)
        ds.photometry.reset_index(drop=True, inplace=True)
        ds.apply_overrides(overrides=overrides)

        # Cleaning
        ds.recalibrate_fluxcal_zpt()
        ds.fill_out_redshifts()
        ds.set_all_rest_phases()
        return ds

    @classmethod
    def from_table_file(
        cls,
        fname: str | Path | StringIO,
        data_root: str | Path = Path(),
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: list[str] | str = "SEARCH_PEAKMJD",
        file_format: str | ArrayLike = "SNANA",
        comment: str = "#",
        jobid: int = 1,
        njobtot: int = 1,
        sim_prescale: int = 1,
        **kwargs,
    ) -> SNDataset:
        """
        Instantiate an SNDataset from a space-separated index table.

        The only required column is "files", which may contain multiple file-paths
        separated by commas. Additional columns for any metadata attribute can be
        included to override the metadata read in from the file.

        Parameters
        ----------
        fname
            Path or stream of index table containing a ``'files'`` column.
        data_root
            Base directory for relative paths in the index file.
        fluxcal_zpt
            Standard photometric zeropoint.
        peakmjd_key
            Key used for peak MJD.
        file_format
            File format ('SNANA' or 'snpy').
        comment
            Comment character in index file.
        jobid
            Job index for distributed loading.
        njobtot
            Total number of jobs.
        sim_prescale
            Prescale factor for simulations.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_ascii_files`.

        Returns
        -------
        Combined supernova dataset.
        """
        sn_list = pd.read_csv(fname, comment=comment, sep=r"\s+")
        if "files" not in sn_list:
            raise ValueError(
                f"The file {fname} does not have a header row with a column named "
                "'files'. This column is required."
            )
        if (isinstance(file_format, str) and file_format.lower() == "snana") or file_format[0].lower() == "snana":
            first_path = Path(data_root) / sn_list["files"][0].split(",")[0]
            meta = io.read_snana_ascii_meta(first_path)
            if "SIM_REDSHIFT_HELIO" in meta:
                njobtot *= sim_prescale
        use_in_run = np.where((np.arange(sn_list.shape[0]) + 1 - jobid) % njobtot == 0)
        sn_list = sn_list.iloc[use_in_run].reset_index(drop=True)
        if isinstance(file_format, str):
            file_format = [file_format,]*sn_list.shape[0]
        elif len(file_format) != sn_list.shape[0]:
            raise ValueError(
                f"file_format was provided as a {type(file_format)} with length "
                f"{len(file_format)}, which does not match the number of rows in "
                f"the file {fname} ({sn_list.shape[0]}) for job {jobid} of "
                f"{njobtot}. You can broadcast a string to all rows if the "
                "file_format argument is a string."
            )
        ds = SNDataset()

        overrides = {col: sn_list[col] for col in sn_list.columns}
        print("Reading light curves...")
        for i, row in tqdm(sn_list.iterrows()):
            # All metadata from files should match, even if they will be overridden.
            row_ds = SNDataset.from_ascii_files(
                [Path(data_root, f) for f in row.files.split(",")],
                fluxcal_zpt=fluxcal_zpt,
                peakmjd_key=peakmjd_key,
                file_format=file_format[i],
                overrides=dict([(key, val[i]) for key, val in overrides.items()]),
                # Already handled jobid/njobtot/sim_prescale
            )
            ds.append(ds=row_ds)
        return ds

    @classmethod
    def from_snana_fits(
        cls,
        fname: str | Path,
        keep_list: list[str] | None = None,
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: list[str] | str = "SEARCH_PEAKMJD",
        jobid: int = 1,
        njobtot: int = 1,
        sim_prescale: int = 1,
        **kwargs,
    ) -> SNDataset:
        """
        Instantiate an SNDataset from SNANA FITS files (HEAD and PHOT).

        Parameters
        ----------
        fname
            Path to the SNANA FITS file (HEAD or PHOT).
        keep_list
            List of SNIDs to keep.
        fluxcal_zpt
            Photometric zeropoint.
        peakmjd_key
            Key for peak MJD.
        jobid
            Job index for batch loading.
        njobtot
            Total number of jobs.
        sim_prescale
            Prescale factor for simulations.
        **kwargs
            Additional arguments forwarded to :func:`bayesn.io.read_snana_fits`.

        Returns
        -------
        Loaded supernova dataset.
        """
        if keep_list is None:
            keep_list = []
        sn_dict, obs_df = io.read_snana_fits(
            fname,
            keep_list=keep_list,
            fluxcal_zpt=fluxcal_zpt,
            jobid=jobid,
            njobtot=njobtot,
            sim_prescale=sim_prescale,
        )
        sn_dict = clean_peakmjd_key(sn_dict, peakmjd_key)
        sim = "SIM_REDSHIFT_HELIO" in sn_dict
        ds = SNDataset(sim=sim)
        ds.append(sn_dict=sn_dict, obs_df=obs_df)
        # Cleaning
        ds.recalibrate_fluxcal_zpt()
        ds.fill_out_redshifts()
        ds.set_all_rest_phases()
        return ds

    @classmethod
    def from_snana_list(
        cls,
        fname: str | Path,
        data_root: str | Path = Path(),
        keep_list: list[str] | None = None,
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: list[str] | str = "SEARCH_PEAKMJD",
        jobid: int = 1,
        njobtot: int = 1,
        sim_prescale: int = 1,
        **kwargs,
    ) -> SNDataset:
        """
        Instantiate an SNDataset from an SNANA LIST file.

        This file contains lines of file names in the same directory.
        Accepted formats are SNANA ASCII files and SNANA FITS files (pointing to HEAD).

        Parameters
        ----------
        fname
            Path to the LIST file containing supernova file names.
        data_root
            Base directory containing the files listed in `fname`.
        keep_list
            List of SNIDs to keep.
        fluxcal_zpt
            Photometric zeropoint.
        peakmjd_key
            Key for peak MJD.
        jobid
            Job index for batch loading.
        njobtot
            Total number of jobs.
        sim_prescale
            Prescale factor for simulations.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Loaded supernova dataset.
        """
        if keep_list is None:
            keep_list = []
        sn_list = np.atleast_1d(np.loadtxt(fname, dtype="str"))
        file_format = np.array([s.rstrip(".gz").split(".")[1].lower() for s in sn_list])
        fits_files = np.where(file_format == "fits")
        ascii_files = np.where(file_format != "fits")

        ds = SNDataset()
        for sn_file in tqdm(sn_list[fits_files]):
            row_ds = SNDataset.from_snana_fits(
                fname=Path(data_root, sn_file),
                keep_list=keep_list,
                fluxcal_zpt=fluxcal_zpt,
                peakmjd_key=peakmjd_key,
                jobid=jobid,
                njobtot=njobtot,
                sim_prescale=sim_prescale,
            )
            ds.append(ds=row_ds)
        for sn_file in sn_list[ascii_files]:
            row_ds = SNDataset.from_ascii_files(
                Path(data_root, sn_file),
                fluxcal_zpt=fluxcal_zpt,
                peakmjd_key=peakmjd_key,
                file_format="SNANA",
                overrides={},
                jobid=jobid,
                njobtot=njobtot,
                sim_prescale=sim_prescale,
            )
            ds.append(row_ds)
        ds.recalibrate_fluxcal_zpt()
        ds.set_all_rest_phases()
        ds.fill_out_redshifts()
        return ds

    #####################
    ### Data Products ###
    #####################
    # These methods should not alter the state of SNDataset
    def make_fitres_table(
        self,
        version_photometry: bool = False,
        survey_overwrite: str | None = None,
        cut_dict: dict | None = None,
        keep_dict: dict | None = None,
        survey_dict: dict | None = None,
    ) -> tuple[QTable, pd.DataFrame]:
        """
        Create a fitres table compatible with SNANA and PIPPIN.

        Parameters
        ----------
        version_photometry
            There are some fitres table columns that are only included if working with
            the version_photometry arg. This boolean flag adds those extra keys if True.
        survey_overwrite
            If None, use the list of strings in the survey attribute, else use the
            survey_overwrite string for each SN in the dataset.
        cut_dict
            Similarly formatted dictionary where the values indicate what to cut.
            See :meth:`cut_fitres_table` for more details.
        keep_dict
            Dictionary with keys matching fitres_table columns and values indicating
            what to keep, cutting all else. See :meth:`cut_fitres_table` for details.
        survey_dict
            Mapping from survey names to integer IDs.

        Returns
        -------
        fitres_table
            Table of supernova parameters filtered by `keep_dict` and `cut_dict`.
        all_table
            A similar table where nothing has been cut, with an additional column ("DROP")
            indicating what cut caused the row to be dropped.
        """
        cut_dict = cut_dict if cut_dict is not None else {}
        keep_dict = keep_dict if keep_dict is not None else {}
        survey_dict = survey_dict if survey_dict is not None else {}
        meta = self.get_metadata_subset(use_defaults=True)
        varlist = ["SN:"] * self.N_sn
        if survey_overwrite is None:
            survey = meta["survey"]
        else:
            survey = [survey_overwrite,] * self.N_sn
        snrmaxes = np.array([self.calculate_snrmaxes(snid, N=3, default_value=-99) for snid in self.snid])
        arr_dict = {
            "VARNAMES:": varlist,
            "CID": meta["snid"],
            "IDSURVEY": [survey_dict.get(s, 0) for s in survey],
            "TYPE": meta["sn_type"],
            "FIELD": meta["field"],
            "zHEL": meta["z_helio"],
            "zHELERR": meta["z_helio_err"],
            "zHD": meta["z_hubble"],
            "zHDERR": meta["z_hubble_err"],
            "VPEC": meta["vpec"],
            "VPECERR": meta["vpec_err"],
            "MWEBV": meta["mwebv"],
            "HOST_LOGMASS": meta["host_logmass"],
            "HOST_LOGMASS_ERR": meta["host_logmass_err"],
            "SNRMAX1": snrmaxes[:,0],
            "SNRMAX2": snrmaxes[:,1],
            "SNRMAX3": snrmaxes[:,2],
        }
        if version_photometry:
            arr_dict.update({
                "SEARCH_PEAKMJD": self.peak_mjd,
                "NEPOCH": self.N_obs,
                "TRESTMIN": self.phase_min,
                "TRESTMAX": self.phase_max,
            })
        if self.sim:
            special_keys = {
                "SIM_REDSHIFT_CMB": "SIM_ZCMB",
                "SIM_DLMU": "SIM_DLMAG",
            }
            snana_names = [get_SNANA_name(key) for key in meta_names["sim"]]
            fitres_snana_names = [special_keys.get(key, key) for key in snana_names]
            arr_dict.update(dict(zip(fitres_snana_names, self.meta_sim)))
        fitres_table = QTable(names=list(arr_dict.keys()), data=list(arr_dict.values()))
        fitres_table["TYPE"] = fitres_table["TYPE"].astype(int)
        fitres_table, all_table = self.cut_fitres_table(fitres_table, keep_dict, cut_dict)
        return fitres_table, all_table

    def cut_fitres_table(
        self,
        fitres_table: QTable,
        keep_dict: dict | None = None,
        cut_dict: dict | None = None,
    ) -> tuple[QTable, pd.DataFrame]:
        """
        Apply cuts to fitres_table based on keep_dict and cut_dict.

        Parameters
        ----------
        fitres_table
            An astropy.table.QTable as produced by :meth:`make_fitres_table`.
        keep_dict
            Dictionary with keys matching fitres_table columns (case-insensitive) and
            2-tuple values which define an open interval, values outside of which
            (exclusive) will be cut. These cuts only affect the fitres_table, not the
            SNDataset.
        cut_dict
            Similarly formatted dictionary where the 2-tuple values define an open
            interval where values inside the interval will be cut.

            Example:
                Using a keep_dict of ``{"mwEBV": (0.00, 0.3)}`` and a cut_dict of
                ``{"HOST_logMASS": (12, np.inf), "SNRMAX3": (-100, 1)}`` will EXCLUDE SNe with
                host-galaxy stellar masses greater than 12, SNRMAX3 values greater than
                -100 and less than 1, and MW E(B-V) values <= 0 or >= 0.3.

        Returns
        -------
        cut_table
            The fitres_table without rows that are cut by keep_dict or cut_dict.
        all_table
            The original fitres_table with an additional column ("DROP") indicating
            what cut caused the row to be dropped.
        """
        keep_dict = keep_dict if keep_dict is not None else {}
        cut_dict = cut_dict if cut_dict is not None else {}
        all_table = fitres_table.copy().to_pandas()
        all_table["DROP"] = ""
        param_convert_dict = {
            "REDSHIFT": "zHD",
            "SNRMAX": "SNRMAX1",
            "ZHD": "zHD",
            "ZHD_ERR": "zHDERR",
            "ZHEL": "zHEL",
            "ZHEL_ERR": "zHELERR",
        }
        for d, cut in zip((keep_dict, cut_dict), (False, True)):
            if not len(d):
                continue
            for param_cut, (low, high) in d.items():
                param = param_cut.upper() # Older versions sliced by [7:], not sure why
                param = param_convert_dict.get(param, param)
                if param not in fitres_table.columns:
                    raise ValueError(f"{param} not in the fitres_table.")
                if cut:
                    keep = (fitres_table[param] < low) | (fitres_table[param] > high)
                else:
                    keep = (fitres_table[param] > low) & (fitres_table[param] < high)
                # Record what cut caused a row to be dropped.
                all_table.loc[
                    all_table["CID"].isin(fitres_table[~keep]["CID"]), "DROP"
                ] = param_cut
                fitres_table = fitres_table[keep]
        return fitres_table, all_table

    def make_lcplot_data(self, N: int | None = None) -> pd.DataFrame:
        """
        Make light-weight DataFrame with CID, MJD, flux, flux_err, and flt columns.

        Useful for light curve plotting code.

        Parameters
        ----------
        N
            Number of SNe to use for lcplot. Use N_sn if None.

        Returns
        -------
        Formatted light curve DataFrame.
        """
        N = N or self.N_sn
        orig_columns = ["snid", "mjd", "flux", "flux_err", "flt"]
        renamed_columns = ["CID", "MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
        lcplot_data = self.photometry[:self.phot_idx[N]][orig_columns].copy()
        lcplot_data.columns = renamed_columns
        lcplot_data["CID"] = lcplot_data["CID"].astype("O")
        return lcplot_data

    def make_bayesn_data(
        self,
        data_type: str,
        band_dict: dict[str, int] | None = None,
        N_obs_max: int | None = None,
        cosmo: Any = FlatLambdaCDM(H0=70, Om0=0.3),
        negative_flux_mag_val: Number = -99,
    ) -> ObsData:
        """
        Make the :class:`ObsData` container expected by :class:`~bayesn.bayesn_model.SEDmodel`.

        Parameters
        ----------
        data_type
            The valid options are "mag" or "flux", indicating which photometry columns
            should be included.
        band_dict
            Dictionary mapping band name keys to integer index values. See
            :meth:`get_band_indices` for details.
        N_obs_max
            Since the SEDmodel class cannot support ragged arrays, the observations of
            all SNe must be padded to N_obs_max terms and the padded values will be
            masked. If N_obs_max is unspecified, it will take the maximum value of the
            N_obs property so that all observations are in the obs_data. SNe with more
            than N_obs_max observations will not be in the returned data products.
        cosmo
            An instance of an astropy.cosmology, or at least something with a method
            named "distmod" that calculates the distance modulus for a given redshift.
        negative_flux_mag_val
            If data_type == "mag", magnitudes of negative_flux_mag_val will be
            interpreted as values that came from negative fluxes, which should be
            masked.

        Returns
        -------
        ObsData container containing padded observation arrays and metadata.
        """
        if data_type.lower() not in ("mag", "flux"):
            raise ValueError(f"data_type should be 'mag' or 'flux', not {data_type}.")
        if not hasattr(cosmo, "distmod"):
            raise ValueError(
                f"The given cosmo {cosmo} does not have a 'distmod' method. This is "
                "required for calculating a rough estimate of distance moduli given "
                "peculiar velocity corrected redshifts (z_hubble)."
            )

        meta = self.get_metadata_subset(use_defaults=True)
        N_obs_max = N_obs_max or max(self.N_obs)
        if N_obs_max < max(self.N_obs):
            meta, phot = self.cut_by_meta_numeric(
                "N_obs", ">", N_obs_max, inplace=False
            )
            counts = phot["snid"].value_counts()
            phot_idx = np.append(0, np.cumsum(counts))
        else:
            meta = self.get_metadata_subset(use_defaults=True)
            phot = self.photometry
            phot_idx = self.phot_idx

        N_sn = len(meta["snid"])  # may be less than self.N_sn if SNe were removed.
        N_obs = np.diff(phot_idx)
        mjd, flux, band_indices, mask = [np.zeros((N_obs_max, N_sn)) for _ in range(4)]
        # Errors of 0 are unphysical, so padded values are replaced with ~0.4
        flux_err = np.full((N_obs_max, N_sn), 1/np.sqrt(2*np.pi))
        for sn_idx in range(N_sn):
            i = slice(phot_idx[sn_idx], phot_idx[sn_idx+1])
            mjd[:N_obs[sn_idx], sn_idx] = phot["phase"][i]
            flux[:N_obs[sn_idx], sn_idx] = phot[data_type][i]
            flux_err[:N_obs[sn_idx], sn_idx] = phot[f"{data_type}_err"][i]
            band_indices[:N_obs[sn_idx], sn_idx] = self.get_band_indices(band_dict, phot[i])
            mask[:N_obs[sn_idx], sn_idx] = np.ones(N_obs[sn_idx])
        if data_type == "mag":  # need to do more masking
            neg_mask = flux == negative_flux_mag_val
            mjd[neg_mask] = 0
            flux[neg_mask] = 0
            flux_err[neg_mask] = 1/np.sqrt(2*np.pi)
            band_indices[neg_mask] = 0
            mask[neg_mask] = 0
        # The redshift to use for calculating muhat is z_hubble, but it may not always
        # be available. Issue a UserWarning if so.
        replace_w_cmb = (meta["z_hubble"] == None) & (meta["z_cmb"] != None)
        if any(replace_w_cmb):
            warn(UserWarning(
                "Some SNe are missing z_hubble values. The muhat (distance prior) "
                "calculation will instead use their z_cmb values. The affected SNe are "
                f"\n{meta['snid'][replace_w_cmb]}."
            ))
        z_hubble = np.where(replace_w_cmb, meta["z_cmb"], meta["z_hubble"])
        replace_w_helio = (z_hubble == None) & (meta["z_helio"] != None)
        if any(replace_w_helio):
            warn(UserWarning(
                "Some SNe are missing z_hubble, z_cmb, and ra/dec values. The muhat "
                "(distance prior) calculation will instead use their z_helio values. The "
                f"affected SNe are\n{meta['snid'][replace_w_helio]}."
            ))
        z_hubble = np.where(replace_w_helio, meta["z_helio"], z_hubble)
        return ObsData(
            host_logmass=meta["host_logmass"],
            z_hel=meta["z_helio"],
            z_hel_err=meta["z_helio_err"],
            muhat=np.array(cosmo.distmod(z_hubble.astype(float))),
            MWEBV=meta["mwebv"],
            mjd=mjd,
            flux=flux,
            flux_err=flux_err,
            band_indices=band_indices,
            mask=mask
        )
