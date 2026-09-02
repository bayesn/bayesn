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
import copy
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from io import StringIO
from itertools import chain
from numbers import Number
from pathlib import Path
from typing import Any

from astropy.cosmology import FlatLambdaCDM
from astropy.table import QTable
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
from tqdm import tqdm

from bayesn import io
from bayesn import utils, constants

########################
### Global constants ###
########################
meta_names: dict[str, list[str, ...]] = {
    "str": ["snid", "field", "idsurvey", "cutflag_snana"],
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
all_meta_names: list[str, ...] = (
    meta_names["str"] + meta_names["num"] + meta_names["sim"]
)
# These defaults should not be used to instantiate SNDataset attributes, missing data
# should be represented with python Nones, both for string lists and np.arrays. The
# default_values are for making data products compatible with SNANA.
default_values: dict[str, str | Number] = {
    "field": "VOID",
    "idsurvey": "NULL",
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
######################
### Global methods ###
######################
def get_standard_name(name: str) -> str:
    # SNANA names are largely the same but capitalised though there are a few
    # exceptions which are handled explicitly.
    name = name.lower()
    if name in all_meta_names:
        return name
    misc = {
        # metadata names
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
        "survey":              "idsurvey",
        "sim_dlmu":            "sim_dlmags",
        "sim_template_index":  "sim_template_ids",
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
        "mag_err":          "MAGERR",
    }
    if name in misc:
        return misc[name]
    if name.startswith("Z_"):
        return name.replace("Z", "REDSHIFT")
    if name.startswith("SIM_"):
        return name.rstrip("S")
    warn(UserWarning(f"Not sure what SNANA key {name} refers to, returning input."))
    return name

def clean_sn_dict(sn_dict: dict[str, str | Number | ArrayLike]) -> dict:
    """ Standardise the keys in an sn_dict to the keys expected in this class and
    ensure values are lists or np.arrays as appropriate.

    Parameters
    ----------
    sn_dict:
        A metadata dictionary such as those produced by bayesn.io.read_* methods.
    """
    # Standardising keys
    keys = list(sn_dict.keys())  # avoid changing keys during loop
    for key in keys:
        sn_dict[get_standard_name(key)] = sn_dict.pop(key)
    # Sanitising 0D values.
    for key in sn_dict:
        if key in meta_names["str"] and (isinstance(sn_dict[key], str) or not hasattr(sn_dict[key], "__iter__")):
            sn_dict[key] = np.atleast_1d(sn_dict[key]).astype(str)
        elif key not in meta_names["str"]:
            sn_dict[key] = np.atleast_1d(sn_dict[key]).astype(float)
    # Adding standard keys as Nones if needed.
    keys_to_check = meta_names["str"] + meta_names["num"]
    if any([key.startswith("sim_") for key in sn_dict]):
        keys_to_check = all_meta_names
    for key in set(keys_to_check).difference(sn_dict.keys()):
        sn_dict[key] = np.full(len(sn_dict["snid"]), None)
    return sn_dict

def clean_obs_df(
    obs_df: pd.DataFrame,
    snids: str | ArrayLike = None,
    phot_idx: None | ArrayLike = None,
) -> pd.DataFrame:
    """ Standardise the column names in obs_df, adding a 'snid' column if needed and
    phot_idx is provided.
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

    # Sorting
    obs_df = obs_df.sort_values(list(req_phot_cols)).reset_index(drop=True)
    return obs_df[[*req_phot_cols, *data_cols, *other_cols]]

@dataclass(eq=False)
class SNDataset:
    #################################
    ### Attributes and Properties ###
    #################################
    # constants
    N_sn: int = 0
    fluxcal_zpt: float = 27.5
    sim: bool = False

    # string metadata arrays
    snid:          np.ndarray = np.array([], dtype=str)
    # For SNANA compatibility
    field:         np.ndarray = np.array([], dtype=str)
    idsurvey:      np.ndarray = np.array([], dtype=str)
    cutflag_snana: np.ndarray = np.array([], dtype=str)

    # numeric metadata arrays
    ra:                np.ndarray = np.array([], dtype=float)
    dec:               np.ndarray = np.array([], dtype=float)
    peak_mjd:          np.ndarray = np.array([], dtype=float)
    sn_type:           np.ndarray = np.array([], dtype=float)
    z_helio:           np.ndarray = np.array([], dtype=float)
    z_cmb:             np.ndarray = np.array([], dtype=float)
    z_hubble:          np.ndarray = np.array([], dtype=float)
    z_helio_err:       np.ndarray = np.array([], dtype=float)
    z_cmb_err:         np.ndarray = np.array([], dtype=float)
    z_hubble_err:      np.ndarray = np.array([], dtype=float)
    mwebv:             np.ndarray = np.array([], dtype=float)
    mwebv_err:         np.ndarray = np.array([], dtype=float)
    host_logmass:      np.ndarray = np.array([], dtype=float)
    host_logmass_err:  np.ndarray = np.array([], dtype=float)
    vpec:              np.ndarray = np.array([], dtype=float)
    vpec_err:          np.ndarray = np.array([], dtype=float)
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
    def meta_str(self) -> list[np.ndarray, ...]:
        return [getattr(self, attr) for attr in meta_names["str"]]

    @property
    def meta_num(self) -> list[np.ndarray, ...]:
        return [getattr(self, attr) for attr in meta_names["num"]]

    @property
    def meta_sim(self) -> list[np.ndarray, ...]:
        return [getattr(self, attr) for attr in meta_names["sim"]]

    @property
    def metadata(self) -> dict[str, str | Number | list | np.ndarray]:
        meta_dict = dict(zip(
            meta_names["str"] + meta_names["num"], self.meta_str + self.meta_num
        ))
        if self.sim:
            meta_dict.update(dict(zip(meta_names["sim"], self.meta_sim)))
        meta_dict.update(self.other_metadata)
        return meta_dict

    @property
    def unique_bands(self) -> np.ndarray[str, ...]:
        return self.photometry["flt"].unique()

    @property
    def N_obs(self) -> np.ndarray[int, ...]:
        if not len(self.photometry):
            return np.array([])
        counts = self.photometry["snid"].value_counts()
        return np.array([counts.get(s, 0) for s in self.snid], dtype=int)

    @property
    def phot_idx(self):
        if not len(self.photometry):
            return np.array([0])
        return np.append(0, np.cumsum(self.N_obs))

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
            assert len(sn_dict[attr]) == N_sn
        if obs_df is not None:
            assert len(obs_df["snid"].unique()) == N_sn

    ######################
    ### Getter methods ###
    ######################
    def get_idx(self, snid: str | ArrayLike) -> int | list:
        """ Given a string-like snid argument, return the integer index of its SN.
        ArrayLike arguments return a list of integer indices for all SNe in snid.

        Parameters
        ----------
        snid:
            The snid (or snids) of SN(e) whose indices are being sought.

        Returns
        -------
        idx:
            The index (or indices) of the SN(e).
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
        snid: None | str | ArrayLike = None,
        idx: None | int | ArrayLike = None,
    ) -> int | ArrayLike:
        """ Several methods will accept either snid or idx arguments to indicate the
        relevant SN(e). This private method sanitises the inputs and returns indices
        so methods only have to work a single kind of argument.

        Parameters
        ----------
        snid:
            The snid (or snids) passed to a method.
        idx:
            The index (or indices) passed to a method.

        Returns
        -------
        idx:
            The index (or indices) passed to a method.
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
        snid: None | str | ArrayLike = None,
        idx: None | int | ArrayLike = None,
        use_defaults: bool = False,
    ) -> dict[str, list | ArrayLike]:
        """ Get a subset of the metadata. If snid and idx are both None, return all
        metadata, which may be preferred over self.metadata due to use_defaults.

        Parameters
        ----------
        snid:
            The snid (or snids) of SN(e) whose metadata are being requested.
        idx:
            The index (or indices) of SN(e) whose metadata are being requested.
        use_defaults:
            If True, replace metadata stored as None with the default values in
            default_values.

        Returns
        -------
        metadata_subset:
            Dictionary of metadata keys where the values only include the metadata for
            SN(e) specified by snid or idx.
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
        snid: None | str | ArrayLike = None,
        idx: None | int | ArrayLike = None,
    ) -> pd.DataFrame:
        """ Get a subset of the photometry DataFrame.

        Parameters
        ----------
        snid:
            The snid (or snids) of SN(e) whose photometry is being requested.
        idx:
            The index (or indices) of SN(e) whose photometry is being requested.

        Returns
        -------
        phot_subset:
            DataFrame photometry including only data for SN(e) specified by snid or idx.
        """
        idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx))
        # isin will not preserve index ordering.
        phot_subset = pd.concat([self.photometry[self.photometry["snid"] == self.snid[i]] for i in idx])
        return phot_subset.reset_index(drop=True)

    def get_phot_idx_subset(
        self,
        snid: None | str | ArrayLike = None,
        idx: None | int | ArrayLike = None
    ):
        """ The get_phot_subset method does not provide phot_idx data for mapping SNe
        to photometry. This method will provide an updated phot_idx array if the
        arguments are identical.

        Parameters
        ----------
        snid:
            The snid (or snids) of SN(e) whose phot indices are being requested.
        idx:
            The index (or indices) of SN(e) whose phot indices are being requested.

        Returns
        -------
        phot_subset:
            Array of indices appropriate for the photometry subset restricted to SN(e)
            specified by snid or idx.
        """
        idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx)).astype(int)
        return np.append(0, np.cumsum(self.N_obs[idx]))

    #####################
    ### Data Addition ###
    #####################
    def append(
        self,
        ds: None = None,
        sn_dict: None | dict = None,
        obs_df: pd.DataFrame = None,
        phot_idx: None | ArrayLike = None
    ) -> None:
        """ Append an instance of the SNDataset class to the instance that called the
        method, or append an sn_dict/obs_df such as the read_*.py methods in io.py
        return. May need to clean SNANA_names.
        Avoid creating duplicate metadata.
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
        # all_snids = set(sn_dict["snid"])
        # common_snids = all_snids.intersection(self.snid)
        # new_snids = all_snids.difference(self.snid)
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
        none_arr = np.full(len(sn_dict["snid"]), None)
        for attr in set(self.other_metadata.keys()).difference(sn_dict.keys()):
            self.other_metadata[attr] = np.append(
                self.other_metadata[attr],
                np.full(len(sn_dict["snid"]), None)
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
                self.other_metadata[key] = np.append(self.other_metadata[key], sn_dict[key])
            else:
                self.other_metadata[key] = np.append(np.full(self.N_sn, None), sn_dict[key])
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
        # Comparing metadata to make sure numeric values are in agreement.
        indices = self.get_idx(sn_dict["snid"])
        meta = self.get_metadata_subset(idx=indices)
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
        snid: None | str | ArrayLike = None,
        idx: None | int | ArrayLike = None
    ) -> None:
        """ Remove all photometry and metadata from the SN(e) specified by either snid
        or idx. Adjusts phot_idx to preserve mappings for all other SNe.

        Parameters
        ----------
        snid:
            The snid (or snids) of the SN(e) to be removed.
        idx:
            The index (or indices) of the SN(e) to be removed.
        """
        idx = np.atleast_1d(self._parse_snid_idx_args(snid, idx))
        snids = self.snid[idx]
        attrs_to_prune = meta_names["str"] + meta_names["num"]
        if self.sim:
            attrs_to_prune += meta_names["sim"]
        self.photometry = self.photometry[~self.photometry["snid"].isin(snids)].reset_index(drop=True)
        for attr in attrs_to_prune:
            setattr(self, attr, np.delete(getattr(self, attr), idx))
        counts = self.photometry["snid"].value_counts()
        N_obs = np.array([counts.get(s, 0) for s in self.snid], dtype=int)
        # self.phot_idx = np.append(0, np.cumsum(N_obs))
        self.N_sn -= len(idx)

    def keep_according_to_list(self, keep_list: list[str, ...]) -> None:
        """ Given a list of SNID values to keep, drop all others.

        Parameters
        ----------
        keep_list:
            SNID values that should be kept.
        """
        self.remove_sn(snid=set(self.snid).difference(keep_list))

    def remove_phot_by_idx(self, idx: int | ArrayLike | pd.Index, inplace=True):
        """ Remove the photometry at index (or indices if ArrayLike) idx.
        Adjust phot_idx accordingly and remove SNe with no photometry afterwards.

        Parameters
        ----------
        idx:
            An integer or 1D ArrayLike of integers that give the indices of the rows
            in the photometry DataFrame to be removed.
        inplace:
            If True, actually remove the indicated data, otherwise return a new
            DataFrame and metadata.
        """
        self.photometry = self.photometry.drop(idx).reset_index(drop=True)
        sn_indices = np.searchsorted(self.phot_idx, np.atleast_1d(idx), side="right")
        for sn_idx in np.unique(sn_indices):
            self.phot_idx[sn_idx:] -= list(sn_indices).count(sn_idx)
        # Reverse order to preserve index order before the one being removed.
        self.remove_sn(idx=np.where(self.N_obs == 0)[0])

    def drop_bands(self, drop_bands: list[str, ...]) -> None:
        """ Removes photometric data observed in a band in drop_bands.

        Parameters
        ----------
        drop_bands:
            ArrayLike of bands to be removed from the dataset.
        """
        self.remove_phot_by_idx(np.where(self.photometry["flt"].isin(drop_bands))[0])

    def drop_by_band_lims(
        self,
        band_lim_dict: dict[str, tuple[Number, Number]],
        wave_min: Number,
        wave_max: Number,
        ignore_missing: bool = False
    ) -> None:
        """ Drop data from bands whose redshifted transmissions contain significant
        (>1%) throughput outside a given wavelength range.

        Parameters
        band_lim_dict:
            Keys corresponding to BayeSN bandpass names, values are 2-tuples indicating
            the lower and upper observer-frame wavelength limits between which the
            transmission is defined. The limits are defined by the transmission function
            reaching 1% of its maximum throughput.
        wave_min:
            The bluest rest-frame wavelength to be retained.
        wave_max:
            The reddest rest-frame wavelength to be retained.
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
        # shape (N_bands, N_sn)
        blue_lim_arr = np.array([val[0]/(1+self.z_helio) for val in band_lim_dict.values()])
        red_lim_arr = np.array([val[1]/(1+self.z_helio) for val in band_lim_dict.values()])
        band_indices, sn_indices = np.where((blue_lim_arr < wave_min) | (red_lim_arr > wave_max))
        rm_phot_idx = []
        for sn_idx in set(sn_indices):
            sn_bands = np.array(bands)[band_indices[np.where(sn_indices == sn_idx)[0]]]
            sn_phot = self.get_phot_subset(self.snid[sn_idx])
            rm_phot_idx += list(sn_phot[sn_phot["flt"].isin(sn_bands)].index + self.phot_idx[sn_idx])
        self.remove_phot_by_idx(rm_phot_idx)

    def cut_by_meta_numeric(
        self,
        name: str,
        logic: str,
        val: Number,
        inplace: bool = True,
        use_defaults: bool = False
    ) -> None | tuple[dict[str, str | Number | ArrayLike], pd.DataFrame, np.ndarray]:
        """ Remove data from SNe selected based on numeric metadata.

        Parameters
        ----------
        name:
            The name of the metadata attribute whose values will be filtered.
        logic:
            The logical operator of the comparison. Should be in =, <, <=, >, or >=.
            See the where_logic docstring in utils.py for more options.
        val:
            The value against which the selected metadata will be compared.
        inplace:
            If True, remove the cut SNe from the dataset, otherwise return a tuple
            containing a cut metadata dict, photometry DataFrame, and phot_idx array.
        use_defaults:
            If True, replace metadata stored as None with the default values in
            default_values when determining cuts.
        """
        if name not in meta_names["num"] and name != "N_obs":
            raise ValueError(
                f"{name} not recognised. "
                f"Valid options are {meta_names['num']+meta_names['sim']}."
            )
        defaults = default_values if use_defaults else {}
        meta_arr = [defaults[name] if x is None else x for x in getattr(self, name)]
        drop = utils.where_logic(arr=meta_arr, val=val, logic=logic)[0]
        if not inplace:
            keep = list(set(np.arange(self.N_sn)).difference(drop))
            meta = self.get_metadata_subset(idx=keep, use_defaults=use_defaults)
            phot = self.get_phot_subset(idx=keep)
            phot_idx = self.get_phot_idx_subset(idx=keep)
            return meta, phot, phot_idx
        self.remove_sn(idx=drop)

    def cut_by_phot_numeric(
        self, name: str, logic: str, val: Number, inplace: bool = True
    ) -> None | tuple[dict[str, str | Number | ArrayLike], pd.DataFrame, np.ndarray]:
        """ Remove data from SNe selected based on photometry data.

        Parameters
        ----------
        name:
            The name of the attribute whose values will be filtered.
        logic:
            The logical operator of the comparison. Should be in =, <, <=, >, or >=.
            See the where_logic docstring in utils.py for more options.
        val:
            The value against which the selected metadata will be compared.
        inplace:
            TODO: add support for inplace being False.
        """
        if name not in self.photometry:
            raise ValueError(
                f"{name} not recognised. "
                f"Valid options are {self.photometry.columns}."
            )
        arr = self.photometry[name]
        drop = utils.where_logic(arr=self.photometry[name], val=val, logic=logic)[0]
        if not inplace:
            raise NotImplementedError(
                "inplace filtering is a little harder since the subset methods are "
                "build on SN indices rather than photometry."
            )
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
        default_value: int = -99.
    ) -> list[float, ...]:
        """ Calculates SNR maxima in top N unique bands for LC cuts.

        Parameters
        ----------
        snid:
            snid of SN for which to calculate snrmaxes.
        N:
            Number of unique bandpasses to calculate snrmaxes for. The returned list
            will have length N regardless of the number of unique bandpasses.
        default_value:
            Value to pad return list with when there are fewer than N unique bands.

        Returns
        -------
        snrmaxes:
            list of N floats providing the SNR maxima in descending order.
        """
        phot = self.get_phot_subset(snid)
        snr = phot["flux"]/phot["flux_err"]
        snrmaxes = [snr.max()]
        for i in range(1,N):
            phot = phot[phot["flt"] != phot["flt"][snr.idxmax()]]
            if not len(phot):
                return snrmaxes + [default_value for _ in range(N-i)]
            snr = phot["flux"] / phot["flux_err"]
            snrmaxes.append(snr.max())
        return snrmaxes

    def calculate_rest_phases(self, snid: str, peak_mjd: Number | None = None) -> None:
        """ Given MJDs, a heliocentric redshift, and fiducial times (peak B-band), get
        rest-frame phases (MJD - t_max)/(1+z_helio) and set it in self.photometry["phase"]

        Parameters
        ----------
        peak_mjd:
            A guess for the value of tmax in modified Julian Date. Used to define
            rest-frame phase. If None, this will be inferred from the photometry using
            a SNR^2-weighted average of available epochs. This is not a reliable metric
            if the peak is not covered.
        """
        idx = self.get_idx(snid)
        pkmjd = peak_mjd or self.peak_mjd[idx]
        if pkmjd is None:
            self.peak_mjd[idx] = pkmjd = self.estimate_tmax(self.snid[idx])
        epochs = self.photometry["mjd"][self.phot_idx[idx]:self.phot_idx[idx+1]].to_numpy()
        return (epochs - pkmjd) / (1.0 + self.z_helio[idx])

    def estimate_tmax(self, snid: str) -> Number:
        """ Use a SNR^2-weighted average of available epochs to estimate the time of
        B-band maximum for SN snid.

        If a SN's peak_mjd is None, it will need to be estimated to calculate
        rest-frame phases. Fortunately, the guess does not need to be very accurate if
        BayeSN is sampling for time of maximum, but to constrain the searched parameter
        space, the guess needs to be within 10 days of the truth.

        Parameters
        ----------
        snid:
            snid of SN for which to estimate tmax.
        """
        phot = self.get_phot_subset(snid)
        tmax = utils.SNR_power_weighted_ave(
            phot["mjd"],
            power=2,
            flux=phot["flux"],
            flux_err=phot["flux_err"]
        )
        return tmax


    def get_band_indices(
        self,
        band_dict: dict[str, int],
        photometry: None | pd.DataFrame = None
    ) -> np.ndarray:
        """ Get integer indices of the flt column in the photometry attribute.

        Parameters
        ----------
        band_dict:
            Dictionary with keys matching the names of the filters in the unique_bands
            property and values corresponding to the index. If band_dict is None, use
            the arbitrary ordering of unique_bands to assign 1-based indices.
        photometry:
            If None, use the attached photometry attribute. Allows for getting mapped
            band indices of a subset of the entire DataFrame by passing the subset as an
            argument.

        Returns
        -------
        band_indices:
            A 1-D array mapping the recorded bands to the specified indices.
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
        """ Given redshift metadata, try to infer any missing values by converting
        between heliocentric and CMB rest-frames and treating peculiar velocities.

        For error propagation, we ignore uncertainty in the CMB dipole and correlation
        between peculiar velocities and redshifts.
        We assume if an uncertainty is availabe, the redshift is also available.
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
                dz_pv = dzhub/(1+v)
                z_dpv = 0 if dv is None else -dv*(1+zhub)/(1+v)**2
                dzcmb = self.z_cmb_err[i] = np.sqrt(dz_pv**2 + z_dpv**2)
            if zcmb is None and None not in (zhub, v):
                zcmb = self.z_cmb[i] = (1+zhub)*(1+v) - 1


            # Populating zcmb first allows for potential z_hel <-> zhub.
            if dzhel is None and None not in (zcmb, dzcmb, ra, dec):
                self.z_helio_err[i] = utils.convert_z(zcmb, ra, dec, "cmb", dzcmb)[1]
            if zhel is None and None not in (zcmb, ra, dec):
                self.z_helio[i] = utils.convert_z(zcmb, ra, dec, "cmb")

            if dzhub is None and None not in (zcmb, dzcmb, v):
                dz_pv = dzcmb*(1+v)
                z_dpv = 0 if dv is None else dv*(1+zcmb)
                self.z_hubble_err[i] = np.sqrt(dz_pv**2 + z_dpv**2)
            if zhub is None and None not in (zcmb, v):
                self.z_hubble[i] = (1+zcmb)*(1+v) - 1

    def set_all_rest_phases(self) -> None:
        self.photometry["phase"] = np.concatenate([self.calculate_rest_phases(snid) for snid in self.snid])

    def recalibrate_fluxcal_zpt(self) -> None:
        """ Adjust the zeropoint in mag = zp - 2.5*log10(flux)
        Assume the magnitudes are correct.
        This is an irreversible operation as the data zeropoints are not retained.
        """
        data_zps = np.array(self.photometry["mag"].astype(float) + 2.5*np.log10(self.photometry["flux"].astype(float)))
        flux_scaling = 10**(0.4*(self.fluxcal_zpt - data_zps))
        self.photometry["flux"] *= flux_scaling
        self.photometry["flux_err"] *= flux_scaling

    def apply_filter_map(self, map_dict: dict[str, str]) -> None:
        """
        Parameters
        ----------
        map_dict:
            Dictionary mapping non-standard filter names to their corresponding names in
            bayesn/bayesn-filters/filters.yaml. This does not need to include filters whose
            names in the data file are already aligned with their names in BayeSN.
        """
        self.photometry["flt"] = self.photometry["flt"].map(lambda f: map_dict.get(f, f))

    def apply_error_floor(self, error_floor: float) -> None:
        """
        Parameters
        ----------
        error_floor:
            Error floor in magnitudes. flux_err values less than this value will be
            replaced with the error floor converted to flux space via
            error_floor * (np.log(10)/2.5) * self.photometry["flux"].
        """
        if error_floor <= 0:
            return
        min_err = error_floor * (np.log(10) / 2.5) * self.photometry["flux"].to_numpy()
        self.photometry["flux_err"] = np.maximum(self.photometry["flux_err"].to_numpy(), min_err)
        self.photometry["mag_err"] = np.maximum(
            self.photometry["mag_err"].to_numpy(),
            np.full_like(self.photometry["mag_err"], error_floor)
        )

    #######################
    ### Factory methods ###
    #######################
    # These class methods should return an instance of SNDataset
    @classmethod
    def from_ascii_files(
        cls,
        fname: str | Path | StringIO | ArrayLike,
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: str = "SEARCH_PEAKMJD",
        file_format: str | ArrayLike = "SNANA",
        overrides: dict = {},
        jobid: int = 1,
        njobtot: int = 1,
        **kwargs,
    ):
        """ Instantiate a SNDataset from a file or list of files in the ascii format.
        Parameters
        ----------
        fname:
            The file to be read (string path, Path object, or file-like text-stream),
            or an ArrayLike of files to be read.
        fluxcal_zpt:
            The common zeropoint used for converting between magnitudes and fluxes.
        file_format:
            The kind of files to be read in.
        jobid:
            The jobid and njobtot arguments can be used to parallelize the workflow of
            this method. Each method call will create a SNDataset with ~1/njobtot of
            all file if fname is an ArrayLike. The subset ingested is based on
            (SN_idx - jobid) % njobtot == 0, such that njobtot consecutive jobids will
            ensure every SN index is processed once and only once.
        njobtot:
            See jobid above.
        kwargs:
            kwargs are directed to the read function, with the supported kwargs below
            read_snana_ascii:
                tablename:
                    Lines for observations in SNANA files begin with a string for the
                    table name. The lines do not contain metadata so will be ignored.
                    If the tablename is inaccurate, the returned metadata will either
                    raise a ValueError when it tries to interpret more than 2 or 3 data
                    columns as a header term, or will include a key-value pair (and
                    maybe one for error) in the metadata where the key is the actual
                    table name.
                comment: str
                    Lines starting with the comment string will not be read in for
                    metadata.
            read_snpy:
                comment:
                    Lines starting with the comment string will not be read in, and
                    text after the comment string will be ignored.

        """
        # Support string | Path values for single files/file formats.
        if isinstance(fname, str | Path):
            fname = [fname,]
        use_in_run = np.where((np.arange(len(fname)) + 1 - jobid) % njobtot == 0)
        fname = np.array(fname)[use_in_run]
        if isinstance(file_format, str):
            file_format = np.full_like(fname, file_format)
        else:
            file_format = np.array(file_format)[use_in_run]

        # Instantiate SNDataset then loop over files and append them, not the most
        # efficient so may need to look for other solutions.
        ds = SNDataset()
        for f, fmt in zip(fname, file_format):
            if fmt.lower() == "snana":
                read_fn = io.read_snana_ascii
            elif fmt.lower() in ("snpy", "snoopy"):
                read_fn = io.read_snpy
            sn_dict, obs_df = read_fn(fname=f, fluxcal_zpt=fluxcal_zpt)
            sn_dict["SEARCH_PEAKMJD"] = sn_dict.pop(peakmjd_key)
            ds.append(sn_dict=sn_dict, obs_df=obs_df)
        ds.photometry.reset_index(drop=True, inplace=True)

        for key, val in overrides.items():
            if isinstance(val, Number | str | None) and ds.N_sn == 1:
                val = np.array([val])
            elif isinstance(val, Number | str | None):
                val = np.full(ds.N_sn, val)
            setattr(ds, key, val)

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
        table_format: str = "SNANA",
        file_format: str | ArrayLike = "SNANA",
        comment="#",
        jobid: int = 1,
        njobtot: int = 1,
        **kwargs,
    ):
        """ Instantiate an SNDataset from a space-separated file. The only required
        column is "files", which may contain multiple file-paths separated by commas.
        Additional columns for any metadata attribute can be included to override the
        metadata read in from the file. The SNID value within each file in a single row
        must be identical so that the from_ascii_files method produces a dataset with
        only one SN.

        The accepted formats are ascii files in SNANA or snpy format.
        """
        sn_list = pd.read_csv(fname, comment=comment, sep=r"\s+")
        if "files" not in sn_list:
            raise ValueError(
                f"The file {fname} does not have a header row with a column named "
                "'files'. This column is required."
            )
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

        # metadata overrides
        all_overrides = {}
        for col in sn_list.columns:
            std_name = get_standard_name(col)
            if hasattr(ds, std_name):
                all_overrides[std_name] = sn_list[col]
        # If a redshift/err is overridden, set other redshifts/errs to None so they are
        # recalculated with the overridden redshift when fill_out_redshifts is called
        # in the from_ascii_files method.
        redshift_types = {"z_helio", "z_cmb", "z_hubble"}
        overridden_redshifts = set(all_overrides.keys()).intersection(redshift_types)
        if overridden_redshifts:
            for other_key in redshift_types.difference(overridden_redshifts):
                all_overrides[other_key] = np.full(sn_list.shape[0], None)

        err_types = {f"{k}_err" for k in redshift_types}
        overridden_errs = set(all_overrides.keys()).intersection(err_types)
        if overridden_errs:
            for other_err in err_types.difference(overridden_errs):
                all_overrides[other_err] = np.full(sn_list.shape[0], None)

        print("Reading light curves...")
        for i, row in tqdm(sn_list.iterrows()):
            # All metadata from files should match, even if they will be overridden.
            row_ds = SNDataset.from_ascii_files(
                [Path(data_root, f) for f in row.files.split(",")],
                fluxcal_zpt=fluxcal_zpt,
                fmt=file_format[i],
                overrides=dict([(key, val[i]) for key, val in all_overrides.items()]),
            )
            ds.append(ds=row_ds)
        return ds

    @classmethod
    def from_snana_fits(
        cls,
        fname: str | Path,
        keep_list: list[str, ...] = [],
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: str = "SEARCH_PEAKMJD",
        jobid: int = 1,
        njobtot: int = 1,
        **kwargs,
    ):
        ds = SNDataset()
        sn_dict, obs_df = io.read_snana_fits(
            fname,
            jobid=jobid,
            njobtot=njobtot
        )
        sn_dict["peak_mjd"] = sn_dict.pop(peakmjd_key)
        if "SIM_GENTYPE" in sn_dict:
            ds.sim = True
            for attr in meta_names["sim"]:
                setattr(ds, attr, np.array([]))
        ds.append(sn_dict, obs_df)
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
        keep_list: list[str, ...] = [],
        fluxcal_zpt: Number = 27.5,
        peakmjd_key: str = "SEARCH_PEAKMJD",
        jobid: int = 1,
        njobtot: int = 1,
        **kwargs,
    ):
        """ Instantiate a SNDataset from a path to a LIST file as produced by SNANA.
        This file contains lines of file names in the same directory.

        The accepted formats are SNANA ascii files and SNANA fits files. Paths to fits
        files should point to the HEAD files, not the PHOT files.
        """
        sn_list = np.atleast_1d(np.loadtxt(fname, dtype="str"))
        file_format = np.array([s.rstrip(".gz").split(".")[1].lower() for s in sn_list])
        fits_files = np.where(file_format == "fits")
        ascii_files = np.where(file_format != "fits")

        use_in_run = np.where((np.arange(len(ascii_files[0])) + 1 - jobid) % njobtot == 0)
        ds = SNDataset()
        for sn_file in tqdm(sn_list[fits_files]):
            file_ds = SNDataset.from_snana_fits(
                fname=Path(data_root, sn_file),
                keep_list=kwargs.get("keep_list", []),
                fluxcal_zpt=fluxcal_zpt,
                peakmjd_key=peakmjd_key,
                jobid=jobid,
                njobtot=njobtot,
            )
            ds.append(ds=file_ds)
        for sn_file in sn_list[ascii_files][use_in_run]:
            row_ds = SNDataset.from_ascii_files(
                Path(data_root, sn_file),
                fluxcal_zpt=fluxcal_zpt,
                peakmjd_key=peakmjd_key,
                file_format="SNANA",
                overrides={},
            )
            ds.append_ds(row_ds)
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
        idsurvey_overwrite: None | str = None,
        cut_dict: dict = {},
        keep_dict: dict = {},
    ) -> QTable:
        """ Create a fitres table which has something to do with SNANA and PIPPIN...
        Parameters
        ----------
        version_photometry:
            There are some fitres table columns that are only included if working with
            the version_photometry arg. This boolean flag adds those extra keys if True.
        idsurvey_overwrite:
            If None, use the list of strings in the idsurvey attribute, else use the
            idsurvey_overwrite string for each SN in the dataset.
        keep_dict:
            Dictionary with keys matching fitres_table columns and values indicating
            what to keep, cutting all else.
            See the cut_fitres_table docstring for more details.
        cut_dict:
            Similarly formatted dictionary where the values indicate what to cut.
            See the cut_fitres_table docstring for more details.

        Returns
        -------
        fitres_table:
            Something important for SNANA or PIPPIN, not sure exactly what it does.
            SNe are sculpted by keep_dict and cut_dict.
            The QTable allows for metadata to be attached later on.
        all_table:
            A similar table where nothing has been cut. This table has an additional
            additional column ("DROP") indicating what cut caused the row to be dropped.
        """
        meta = self.get_metadata_subset(use_defaults=True)
        varlist = ["SN:"] * self.N_sn
        if idsurvey_overwrite is None:
            idsurvey = [idsurvey_overwrite,] * self.N_sn
        else:
            idsurvey = meta["idsurvey"]
        t_ranges = np.zeros((self.N_sn, 2))
        snrmaxes = np.zeros((self.N_sn, 3))
        for i in range(self.N_sn):
            snid = self.snid[i]
            phot = self.get_phot_subset(snid)
            snrmaxes[i] = self.calculate_snrmaxes(snid, N=3, default_value=-99)
            if version_photometry:  # can be a bit lazy if not using t_ranges
                t_ranges[i] = phot["phase"].min(), phot["phase"].max()
        arr_dict = {
            "VARNAMES:": varlist,
            "CID": meta["snid"],
            "IDSURVEY": idsurvey,
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
                "TRESTMIN": t_ranges[:,0],
                "TRESTMAX": t_ranges[:,1],
            })
        if self.sim:
            arr_dict.update(dict(zip(
                [get_SNANA_name[key] for key in meta_names["sim"]],
                self.meta_sim
            )))
        fitres_table = QTable(names=list(arr_dict.keys()), data=list(arr_dict.values()))
        fitres_table["TYPE"] = fitres_table["TYPE"].astype(int)
        fitres_table, all_table = self.cut_fitres_table(fitres_table, keep_dict, cut_dict)
        return fitres_table, all_table

    def cut_fitres_table(
        self,
        fitres_table: QTable,
        keep_dict: dict,
        cut_dict: dict,
    ) -> tuple[QTable, QTable]:
        """ Apply cuts to fitres_table based on cut_dict.

        Parameters
        ----------
        fitres_table:
            An astropy.table.QTable as produced by the get_fitres_table method.
        keep_dict:
            Dictionary with keys matching fitres_table columns (case-insensitive) and
            2-tuple values which define an open interval, values outside of which
            (exclusive) will be cut. These cuts only affect the fitres_table, not the
            SNDataset.
        cut_dict:
            Similarly formatted dictionary where the 2-tuple values define an open
            interval where values inside the interval will be cut.

            Example:
                Using a keep_dict of {"mwEBV": (0.00, 0.3)} and a cut_dict of
                {"HOST_logMASS": (12, np.inf), "SNRMAX3": (-100, 1)} will EXCLUDE SNe with
                host-galaxy stellar masses greater than 12, SNRMAX3 values greater than
                -100 (-99 would not cut default values of -99) and less than 1,
                and MW E(B-V) values <= 0 or >= 0.3.

        Returns
        -------
        cut_table:
            The fitres_table without rows that are cut by keep_dict or cut_dict.
        all_table:
            The original fitres_table with an additional column ("DROP") indicating
            what cut caused the row to be dropped.
        """
        all_table = fitres_table.copy().to_pandas()
        all_table["DROP"] = ""
        param_convert_dict = {"REDSHIFT": "zHD", "SNRMAX": "SNRMAX1"}
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
            drop = (1 - keep).sum()
            # Record what cut caused a row to be dropped.
            all_table.loc[
                all_table["CID"].isin(fitres_table[~keep]["CID"]), "DROP"
            ] = param_cut
            fitres_table = fitres_table[keep]
        return fitres_table, all_table

    def make_lcplot_data(self, N: None | int = None) -> pd.DataFrame:
        """ Make light-weight DataFrame with CID, MJD, flux, flux_err, and flt columns.
        Useful for plotting code.

        Parameters
        ----------
        N:
            Number of SNe to use for lcplot. Use N_sn if None.
        """
        N = N or self.N_sn
        orig_columns = ["mjd", "flux", "flux_err", "flt"]
        renamed_columns = ["MJD", "FLUXCAL", "FLUXCALERR", "FLT"]
        lcplot_data = self.photometry[:self.phot_idx[N]][orig_columns].copy()
        lcplot_data.columns = renamed_columns
        cid_arr = np.concatenate(
            [np.full(self.N_obs[i], self.snid[i]) for i in range(N)]
        )
        lcplot_data.insert(loc=0, column="CID", value=cid_arr)
        return lcplot_data
    def make_bayesn_data(
        self,
        data_type: str,
        band_dict: None | dict[str, int] = None,
        N_obs_max: None | int = None,
        cosmo: Any = FlatLambdaCDM(H0=70, Om0=0.3),
        negative_flux_mag_val: Number = -99,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Make the arrays expected by bayesn_model.py's SEDmodel class.
        This method will return numpy arrays, so for faster sampling they should be
        converted to jax arrays and loaded to the device in bayesn_model.py.

        Parameters
        ----------
        data_type:
            The valid options are "mag" or "flux", indicating which photometry columns
            should be included.
        band_dict:
            Dictionary with mapping band name keys to integer index values. See the
            docstring of SNDataset.get_band_indices for more details.
        N_obs_max:
            Since the SEDmodel class cannot support ragged arrays, the observations of
            all SNe must be padded to N_obs_max terms and the padded values will be
            masked. If N_obs_max is unspecified, it will take the maximum value of the
            N_obs property that all observations are in the obs_data. SNe with more
            than N_obs_max observations will not be in the returned data products.
        cosmo:
            An instance of an astropy.cosmology, or at least something with a method
            named "distmod" that calculates the distance modulus for a given redshift,
            returning something that can be interpreted as an np.array.
        negative_flux_mag_val:
            If data_type == "mag", magnitudes of negative_flux_mag_val will be
            interpreted as values that came from negative fluxes, which should be
            masked.

        Returns
        -------
        sn_data:
            A shape (5, N_sn) array with the first axis including:
                host-galaxy mass: Number
                redshift: Positive Number
                redshift_error: Positive Number
                muhat: Positive Number
                MWEBV: Positive Number
        obs_data:
            A shape (5, N_obs_max, N_sn) array with the first axis including:
                phase: Number
                flux or mag: Number
                flux_err or mag_err: positive Number
                band_indices: int
                mask: bool
        """
        if data_type.lower() not in ("mag", "flux"):
            raise ValueError(f"datatype should be 'mag' or 'flux', not {data_type}.")
        if not hasattr(cosmo, "distmod"):
            raise ValueError(
                f"The given cosmo {cosmo} does not have a 'distmod' method. This is "
                "required for calculating a rough estimate of distance moduli given "
                "peculiar velocity corrected redshifts (z_hubble)."
            )

        meta = self.get_metadata_subset(use_defaults=True)
        N_obs_max = N_obs_max or max(self.N_obs)
        if N_obs_max < max(self.N_obs):
            meta, phot, phot_idx = self.cut_by_meta_numeric(
                "N_obs", ">", N_obs_max, inplace=False
            )
        else:
            meta = self.get_metadata_subset(use_defaults=True)
            phot = self.photometry
            phot_idx = self.phot_idx

        sn_data = np.array([
            meta["host_logmass"], meta["z_helio"], meta["z_helio_err"],
            np.array(cosmo.distmod(meta["z_hubble"])), meta["mwebv"]
        ])

        N_sn = len(meta["snid"])  # may be less than self.N_sn if SNe were removed.
        N_obs = np.diff(phot_idx)
        obs_data = np.zeros((5, N_obs_max, N_sn))  # pads with 0s
        # Errors of 0 are unphysical, so padded values are replaced with ~0.4
        obs_data[2] = 1/np.sqrt(2*np.pi)
        for sn_idx in range(N_sn):
            i = slice(phot_idx[sn_idx], phot_idx[sn_idx+1])
            obs_data[:, :N_obs[sn_idx], sn_idx] = np.array([
                phot["phase"][i], phot[data_type][i], phot[f"{data_type}_err"][i],
                self.get_band_indices(band_dict, phot[i]), np.ones(N_obs[sn_idx])
            ])
        if data_type == "mag":  # need to do more masking
            mask = obs_data[1] == negative_flux_mag_val
            obs_data[:2, mask] = 0  # phase, mag
            obs_data[3:, mask] = 0  # band_indices, mask
        return sn_data, obs_data
