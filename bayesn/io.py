"""
Low-level file-handling I/O
Methods in this module should either:
    read_*: convert data in various formats to data in the standardised format of
            sn_dict and obs_df expected by datasets.py's SNDataset factory methods
    write_*: write data in those formats to various formats.
This module is not for performing astronomical data transformations.
"""

from collections import OrderedDict as odict
import copy
from io import StringIO
from itertools import pairwise
from numbers import Number
from pathlib import Path
import gzip
import re
import time
from typing import Any
from warnings import warn

from astropy.io import fits
import astropy.table as at
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
import sncosmo

from bayesn import constants, utils

sn_dict_keys = (
    # SNANA keys
    "SNID", "RA", "DECL", "MWEBV", "HOSTGAL_LOGMASS", "REDSHIFT_HELIO", "REDSHIFT_CMB",
    "REDSHIFT_FINAL", "VPEC", "VPEC_ERR", "SEARCH_PEAKMJD", "ZP_FLUXCAL",
    # Added keys
    "MWEBV_ERR", "HOSTGAL_LOGMASS_ERR", "REDSHIFT_HELIO_ERR", "REDSHIFT_CMB_ERR",
    "REDSHIFT_FINAL_ERR",
)
aliases = {
    "DECL": "DEC",
    "SEARCH_PEAKMJD": "PEAKMJD",
    "ZP_FLUXCAL": "FLUXCAL_ZPT",
}
obs_df_columns = ["MJD", "FLT", "flux", "flux_err", "mag", "mag_err"]  # list for df indexing

###############
### Reading ###
###############
def read_snana_spectra(file_path: str | Path | StringIO):
    """
    Read spectroscopy from the SNANA lightcurve file
    Forked and lightly modified from David Jones' SALTShaker repo
    https://github.com/djones1040/SALTShaker/blob/main/saltshaker/util/snana.py

    Parameters
    ----------
    file_path : str
        path to the light curve file.

    Returns
    -------
    spectrum : dict
        dictionary with keys for each SPECTRUM_ID and values of 2-tuples.
        The first element in the 2-tuple is a float for the observation MJD.
        The second elment is a 4xN np.ndarray spanning:
            wavelength_min, wavelength_max, flux, flux_error.

    Notes
    -----
    Expected format is something like
    ```
    ...
    NSPECTRA: 21


    NVAR_SPEC: 5
    VARNAMES_SPEC: LAMMIN LAMMAX  FLAM  FLAMERR SPECFLAG

    SPECTRUM_ID: 1
    SPECTRUM_MJD:  48639.50
    SPECTRUM_NLAM:     1520
    SPEC:   3540.52   3542.84 6.48624e-14 2.81711e-15 1
    SPEC:   3542.84   3545.14 6.48624e-14 2.81711e-15 1
    ...
    SPEC:   7054.62   7056.93 3.98786e-15 2.76859e-15 1
    SPECTRUM_END:

    SPECTRUM_ID: 2
    SPECTRUM_MJD:  48635.50
    SPECTRUM_NLAM:     1520
    SPEC:   3520.64   3522.97 5.53949e-14 1.38963e-15 1
    SPEC:   3527.62   3529.95 5.35848e-14 1.38963e-15 1
    ...
    ```
    """

    if isinstance(file_path, str | Path) and str(file_path).endswith('.gz'):
        f = gzip.open(file_path, 'rt')
    elif isinstance(file_path, str | Path):
        f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    spec_lines = np.array([line.split()[1:] for line in lines if line.startswith('SPEC:')]).astype(float)

    spectra = {}
    startSpec = False
    spec_line_idx = 0
    for line in lines:
        if line.startswith('VARNAMES_SPEC'):
            specvarnames = line.split()[1:]
        elif startSpec and line.startswith('SPEC:'):
            spec_line_idx += 1
        elif line.startswith('SPECTRUM'):
            if line.startswith('SPECTRUM_ID'):
                startSpec = True
                specid = int(line.split()[1])
                spectra[specid] = {}
                idx_start = spec_line_idx
            elif not line.startswith('SPECTRUM_END'):
                # SPECTRUM_MJD or SPECTRUM_NLAM
                spectra[specid][line.split()[0].replace("SPECTRUM_", "").strip(':')] = float(line.split()[1])
            elif startSpec:
                # SPECTRUM_END
                startSpec = False
                idx_end = spec_line_idx
                for var_idx, column in enumerate(spec_lines.T):
                    spectra[specid][specvarnames[var_idx]] = column[idx_start:idx_end]
    return spectra

def read_snana_ascii_meta(
    fname: str | Path | StringIO,
    tablename: str = "OBS",
    comment: str = "#",
    stat_and_sys: bool = False
) -> odict[str, Any]:
    """ sncosmo.read_snana_ascii does not support uncertainties in the header keywords.
    This method will read lines in the format "KEY: number1 number2" and add key-value
    pairs "KEY: number1" and "KEYERR: number2" to the returned metadata. The numbers
    are detected through regex, with support for integers, floats, engineering notation
    (e or E). Text following the comment arg substring is ignored. If three numbers are
    detected and the stat_and_sys arg is True, number2 and number3 will be treated as
    statistical and systematic error terms and added in quadrature to make a single
    KEYERR value.


    Parameters
    ----------
    fname:
        The path of the file to be read (string or Path object), or a file-like text
        stream.
    tablename:
        Lines for observations in SNANA files begin with a string for the table name.
        The lines do not contain metadata so will be ignored. If the tablename is
        inaccurate, the returned metadata will either raise a ValueError when it tries
        to interpret more than 2 or 3 data columns as a header term, or will include a
        key-value pair (and maybe one for error) in the metadata where the key is the
        actual table name.
    comment:
        Lines starting with the comment string will not be read in, and text after the
        comment string will be ignored.
    stat_and_sys:
        If the header contains lines with three numeric substrings and this boolean arg
        is True, the second and third numbers will be added in quadrature to get an
        error. If False, a ValueError will be raised as it is no longer clear what
        the numbers mean.

    Returns
    -------
    meta: odict
        Keys include all header terms, defined as strings beginning a line and followed
        by a ":" and a non-empty string ("END:" will be ignored).
    """
    fh = open(fname, "r") if isinstance(fname, str | Path) else fname
    lines = fh.readlines()
    fh.close()
    lines = [
        line.split(comment)[0].rstrip("\n") for line in lines
        if len(line.split()) > 1
        and len(line.split(":")) >= 2
        and line.split()[0] != tablename+":"
        and not line.startswith(comment)
    ]
    lines = [" ".join(l.split()) for l in lines]

    keys, vals = [[] for _ in range(2)]
    raw_keys = [l.split()[0].rstrip(":") for l in lines]
    raw_vals = [":".join(l.split(":")[1:]).strip() for l in lines]
    # raw_vals is string which may or may not contain numeric substrings.
    # If any numeric substrings are found, then assume non-numeric text is comments,
    # units, or +/- signs which can be safely dropped.
    # strings with no float-like substrings are string values and are retained in full.

    # regex explanation: [-+]? for optional sign symbols (?:...) for non-capturing group
    # to prevent findall from spitting out things matching the ...
    # \d* for 0 or more digits, \d+ for 1 or more, \.? for 0 or 1 "." symbols.
    # | is a logical or, allowing for matches to numbers starting with ".".
    # (?:[eE][-+]?\d+)? for non-captured, optional engineering notation
    number_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

    # skip numeric searching to allow for mixed names like SN1234abc, which would
    # otherwise have its numeric component read as the value, 1234.
    non_numeric_fields = ("SNID", "IAUC", "SURVEY")
    for i in range(len(raw_vals)):
        if raw_keys[i] in non_numeric_fields:
            keys.append(raw_keys[i])
            vals.append(raw_vals[i])
            continue
        matches = number_pattern.findall(raw_vals[i])
        if len(matches):
            float_vals = [float(l) for l in matches]
            # Assume the first numeric substring is the value.
            keys.append(raw_keys[i])
            vals.append(float_vals[0])
            if len(float_vals) == 2:
                # Assume the second (if it exists) is some kind of uncertainty
                keys.append(raw_keys[i]+"_ERR")  # consistent with SNANA keys
                vals.append(float_vals[1])
            elif len(float_vals) == 3 and stat_and_sys:
                # Quadrature sum of stat+sys error if specified.
                keys.append(raw_keys[i]+"_ERR")  # consistent with SNANA keys
                vals.append(np.sqrt(float_vals[1]**2 + float_vals[2]**2))
            elif len(float_vals) >= 3:
                raise ValueError(
                    f"Found {len(float_vals)} numeric strings while reading {fname} "
                    f"The raw line reads '{lines[i]}'. Text after {comment} is ignored."
                    "Single numeric strings are interpreted as values, two are "
                    "interpreted as a value and uncertainty. Three values are may be "
                    "interpreted as a statistical and systematic error if the "
                    "stat_and_sys argument is True, in which case their quadrature sum "
                    f"will be assigned to {raw_keys[i]}_ERR. Other values are not "
                    "supported."
                )
        else:
            keys.append(raw_keys[i])
            vals.append(raw_vals[i])
    return odict(zip(keys, vals))

def read_snana_ascii(
    fname: str | Path | StringIO,
    fluxcal_zpt: None | Number = None,
    tablename: str = "OBS",
    comment: str = "#",
    **kwargs
) -> tuple[odict[str, str | Number], pd.DataFrame]:
    """ Given a path to an ascii file in the SNANA format, read it, extract relevant
    data, and return the data in two standard formats. SN-level data are returned as a
    dictionary, while observation-level data are returned in a DataFrame.

    Parameters
    ----------
    fname:
        The path of the file to be read (string or Path object), or a file-like text
        stream. The SNANA format has a number of metadata keys, indicated with a line
        "KEY: VALUE". The observation-level data is stored in tables where each row in
        the table starts with the table name and a ":" (e.g. "OBS: "), then columns as
        indicated by VARLIST. The file must have the columns MJD, FLT, FLUXCAL, and
        FLUXCALERR. MAG and MAGERR columns are ignored.
    fluxcal_zpt:
        The common zeropoint used for converting between magnitudes and fluxes. Like
        peak_mjd, if this arg is None, it will be inferred from sn_dict, or if that is
        also None, set to a default of 27.5.
    tablename:
        Lines for observations in SNANA files begin with a string for the table name.
        The lines do not contain metadata so will be ignored. If the tablename is
        inaccurate, the returned metadata will either raise a ValueError when it tries
        to interpret more than 2 or 3 data columns as a header term, or will include a
        key-value pair (and maybe one for error) in the metadata where the key is the
        actual table name.
    comment:
        Lines starting with the comment string will not be read in for metadata.
    Returns
    -------
    sn_dict:
        Contains data specific to the SN, but not to each observation.
        See SNANA_keys variable for sn_dict keys consistent with SNANA
            RA: right ascension [deg]
            DECL: declination [deg]
            MWEBV: Host-galaxy E(B-V) [mag]
            HOSTGAL_LOGMASS: log_{10}(host-galaxy stellar mass/1 Msun)
            REDSHIFT_HELIO: Heliocentric redshift
            REDSHIFT_CMB: CMB rest-frame redshift
            REDSHIFT_FINAL: Redshift to use on a Hubble Diagram, generally z_CMB
                with peculiar velocities subracted out.
            VPEC: Peculiar velocity [km / s]
            VPEC_ERR: Uncertainty in peculiar velocity [km / s]
            SEARCH_PEAKMJD: A guess of the B-band time of maximum light, which is
                used to define rest-frame phase [MJD]
        Added Keys
            MWEBVERR:           Uncertainty on MWEBV [mag]
            HOSTGAL_LOGMASSERR: Uncertainty on HOSTGAL_LOGMASS
            REDSHIFT_HELIOERR:  Uncertainty on REDSHIFT_HELIO
            REDSHIFT_CMBERR:    Uncertainty on REDSHIFT_CMB
            REDSHIFT_FINALERR:  Uncertainty on REDSHIFT_FINAL
    obs_df:
        Contains a row for each observation with the following columns:
            MJD: Phase of observation (rest-frame) with respect to peak_mjd [days]
            FLT: BayeSN name of the photometric filter used in the observation.
            flux: Unitless flux with fiducial zeropoint of either 27.5 or another key
                  specified in the file.
            flux_err: Uncertainty on the flux.
            mag: Apparent magnitude.
            mag_err: Uncertainty on those magnitudes.
    """
    # --- get sn_dict from file ---
    file_meta = read_snana_ascii_meta(fname=copy.deepcopy(fname), tablename=tablename, comment=comment)
    sn_dict = odict()
    for key in set(file_meta).union(set(sn_dict_keys)):
        sn_dict[key] = file_meta.get(key)
    if isinstance(fluxcal_zpt, Number):
        sn_dict["ZP_FLUXCAL"] = fluxcal_zpt
    else:
        fluxcal_zpt = sn_dict["ZP_FLUXCAL"] = file_meta.get("ZP_FLUXCAL", 27.5)
    for key in ("VARLIST", "FILTERS"):
        sn_dict.pop(key, None)

    # --- get data from file ---
    if isinstance(fname, Path):
        fname = str(fname)  # sncosmo does not like PosixPaths
    _, obs_df = sncosmo.read_snana_ascii(fname, default_tablename=tablename)
    obs_df = obs_df["OBS"].to_pandas()

    if "BAND" in obs_df.columns:  # This column can have different names, forcing FLT
        obs_df = obs_df.rename(columns={"BAND": "FLT"})
    obs_df = obs_df.rename(columns={"FLUXCAL": "flux", "FLUXCALERR": "flux_err"})
    obs_df = obs_df.dropna(subset=["flux", "flux_err"])
    obs_df["mag"], obs_df["mag_err"] = utils.flux_to_mag(
            obs_df["flux"].values.astype(float),
            obs_df["flux_err"].values.astype(float),
            fluxcal_zpt,
            nan_val=-99  # for backwards compatibility/consistency.
            # mags and mag_errs for negative flux should probably be nans at some point.
    )
    obs_df = obs_df[obs_df_columns]
    obs_df["snid"] = sn_dict.get("SNID")
    return sn_dict, obs_df

def read_snana_fits(
    fname: str | Path,
    keep_list: list[str, ...] = [],
    fluxcal_zpt: Number = 27.5,
    jobid: int = 1,
    njobtot: int = 1,
) -> tuple[dict, pd.DataFrame]:
    """ Given a path to an SNANA format fits (or fits.gz) HEAD file, read it and the
    corresponding PHOT file, extract relevant data, and return the data in two standard
    formats. SN-level data are returned as a dictionary, while observation-level data
    are returned in a DataFrame.

    Unlike the read ascii methods, SNANA-format fits files usually contain data for
    multiple SNe, so this method's intended use case is not to iterate over many files
    pertaining to one SN, but to parse large fits files efficiently. To that end, it
    be advantageous to split the data processing over multiple jobs using the jobid and
    njobtot arguments (see below).

    The sn_dict and obs_df arguments in the read ascii methods are not accepted as
    arguments here, since one should not have to update the data structures with
    multiple method calls.

    Parameters
    ----------
    fname:
        The path of the HEAD file to be read (string or Path object). The SNANA fits
        files come in pairs, a HEAD file with one row of data per SN, and a PHOT file
        with one row of data per observation. The PTROBS_MIN and PTROBS_MAX values in
        the head file provide pointers to the start and end indices of that SN's data
        in the PHOT file.
    map_dict:
        Dictionary mapping non-standard filter names to their corresponding names in
        bayesn/bayesn-filters/filters.yaml. This does not need to include filters whose
        names in the data file are already aligned with their names in BayeSN.
    peak_mjd_key:
        The HEAD file should contain guesses for the value of tmax in modified Julian
        Date for each SN. This argument provides the name of the column with that data.
    fluxcal_zpt:
        The common zeropoint used for converting between magnitudes and fluxes. Like
        peak_mjd, if this arg is None, it will be inferred from sn_dict, or if that is
        also None, set to a default of 27.5.
    error_floor:
        Error floor in magnitudes. FLUXCALERR values less than this value will be
        replaced with the error floor converted to flux space via
        error_floor * (np.log(10)/2.5) * file_data["flux"].
    drop_bands:
        List of bands to be ignored during fitting. Defaults to empty list
        From the input yaml, any bandpasses that should be excluded from the data.
    jobid:
        The jobid and njobtot arguments can be used to parallelize the workflow of this
        method. Each method call will ingest ~1/njobtot of the entire dataset. The
        subset ingested is based on (SN_idx - jobid) % njobtot == 0, such that njobtot
        consecutive jobids will ensure every SN index is processed once and only once.
    njobtot:
        See jobid above.

    Returns
    -------
    sn_dict:
        Contains data specific to the SN, but not to each observation.
        See SNANA_keys variable for sn_dict keys consistent with SNANA
            RA: right ascension [deg]
            DECL: declination [deg]
            MWEBV: Host-galaxy E(B-V) [mag]
            HOSTGAL_LOGMASS: log_{10}(host-galaxy stellar mass/1 Msun)
            REDSHIFT_HELIO: Heliocentric redshift
            REDSHIFT_CMB: CMB rest-frame redshift
            REDSHIFT_FINAL: Redshift to use on a Hubble Diagram, generally z_CMB
                with peculiar velocities subracted out.
            VPEC: Peculiar velocity [km / s]
            VPEC_ERR: Uncertainty in peculiar velocity [km / s]
            SEARCH_PEAKMJD: A guess of the B-band time of maximum light, which is
                used to define rest-frame phase [MJD]
        Added Keys
            MWEBVERR:           Uncertainty on MWEBV [mag]
            HOSTGAL_LOGMASSERR: Uncertainty on HOSTGAL_LOGMASS
            REDSHIFT_HELIOERR:  Uncertainty on REDSHIFT_HELIO
            REDSHIFT_CMBERR:    Uncertainty on REDSHIFT_CMB
            REDSHIFT_FINALERR:  Uncertainty on REDSHIFT_FINAL
    obs_df:
        Contains a row for each observation with the following columns:
            t: Phase of observation (rest-frame) with respect to peak_mjd [days]
            flux: Unitless flux with fiducial zeropoint of either 27.5 or another key
                  specified in the file.
            flux_err: Uncertainty on the flux.
            FLT: BayeSN name of the photometric filter used in the observation.
    """
    head_file = Path(fname)
    if not head_file.exists():
        head_file = Path(str(fname) + ".gz")
    head_data = np.array(fits.open(head_file)[1].data).view(np.ndarray)
    phot_file = str(head_file).replace("HEAD", "PHOT")
    head_data = head_data.byteswap().newbyteorder()
    phot_data = fits.getdata(phot_file, 1, view=np.ndarray, memmap=True)
    n_sne_in_file = head_data.shape[0]
    use_in_run = (np.arange(n_sne_in_file) + 1 - jobid) % njobtot == 0
    idx = np.where(use_in_run)[0]
    head_names = head_data.dtype.names

    # All per-SN arrays from head_data, with defaults for optional fields.
    snid_decoded = (np.char.decode(head_data["SNID"], "utf-8")
        if head_data["SNID"].dtype.kind == "S"
        else head_data["SNID"].astype(str))
    snid_decoded = np.char.strip(snid_decoded)
    sn_dict = {}
    for key in head_names:
        sn_dict[key] = head_data[key]
    sn_dict["SNID"] = snid_decoded
    for key in ("VARLIST", "FILTERS"):
        sn_dict.pop(key, None)

    # Per-SN job/keep_list mask: SNe this job will actually process.
    job_per_sn = np.zeros(n_sne_in_file, dtype=bool)
    job_per_sn[idx] = True
    if len(keep_list):
        job_per_sn &= np.array([s in keep_list for s in snid_decoded])

    # Per-row keep mask, built from PTROBS bounds of kept SNe only.
    # Boolean-indexing the memmap'd phot_data lets the OS page in
    # only those rows for partitioned jobs.
    pointer_min = head_data["PTROBS_MIN"] - 1
    pointer_max = head_data["PTROBS_MAX"]
    row_keep = np.zeros(len(phot_data), dtype=bool)
    sn_idx = np.full(len(phot_data), -1, dtype=np.int64)
    for k in np.where(job_per_sn)[0]:
        row_keep[pointer_min[k]:pointer_max[k]] = True
        sn_idx[pointer_min[k]:pointer_max[k]] = k
    sn_idx = sn_idx[row_keep]

    phot_data = phot_data[row_keep][["MJD", "BAND", "FLUXCAL", "FLUXCALERR"]]
    phot_data = phot_data.byteswap().newbyteorder()
    phot_df = pd.DataFrame(phot_data, columns=phot_data.dtype.names)
    phot_df["FLT"] = phot_df.pop("BAND").str.decode("utf-8").str.strip()
    phot_df.rename(columns={"FLUXCAL": "flux", "FLUXCALERR": "flux_err"}, inplace=True)
    keep = ~np.isnan(phot_df["flux"].values) & ~np.isnan(phot_df["flux_err"].values)
    phot_df = phot_df.iloc[keep].reset_index(drop=True)
    sn_idx = sn_idx[keep]
    sn_dict["phot_idx"] = np.append(0, np.cumsum(sn_dict["NOBS"]))

    phot_df['mag'] = fluxcal_zpt - 2.5 * np.log10(phot_df['flux'].values)
    phot_df['mag_err'] = (2.5 / np.log(10)) * phot_df['flux_err'].values / phot_df['flux'].values
    phot_df["snid"] = snid_decoded[sn_idx]
    return sn_dict, phot_df

def read_snpy(
    fname: str | Path | StringIO,
    fluxcal_zpt: Number = 27.5,
    comment: str = "#",
) -> tuple[odict[str, str | Number], pd.DataFrame]:
    """ Given an ascii file in the SNooPy format, read it, extract relevant data, and
    return the data in two standard formats. SN-level data are returned as a dictionary
    while observation-level data are returned in a DataFrame.

    Parameters
    ----------
    fname:
        The path of the file to be read (string or Path object), or a file-like text
        stream. The file should be in the SNooPy format, meaning the first
        (non-comment) line includes the SN name, heliocentric redshift, ra, and dec.
        The file must then contain filter blocks, where the block for "filter X" begins
        with that line, then any number of rows of listing observer-frame epoch, observed
        magnitude, and the uncertainty of that magnitude, separated with whitespace.
    fluxcal_zpt:
        The common zeropoint used for converting between magnitudes and fluxes. Like
        peak_mjd, if this arg is None, it will be inferred from sn_dict, or if that is
        also None, set to a default of 27.5.
    comment:
        Lines starting with the comment string will not be read in, and text after the
        comment string will be ignored.

    Returns
    -------
    sn_dict:
        Contains data specific to the SN, but not to each observation.
        See SNANA_keys variable for sn_dict keys consistent with SNANA
            RA: right ascension [deg]
            DECL: declination [deg]
            MWEBV: Host-galaxy E(B-V) [mag]
            HOSTGAL_LOGMASS: log_{10}(host-galaxy stellar mass/1 Msun)
            REDSHIFT_HELIO: Heliocentric redshift
            REDSHIFT_CMB: CMB rest-frame redshift
            REDSHIFT_FINAL: Redshift to use on a Hubble Diagram, generally z_CMB
                with peculiar velocities subracted out.
            VPEC: Peculiar velocity [km / s]
            VPEC_ERR: Uncertainty in peculiar velocity [km / s]
            SEARCH_PEAKMJD: A guess of the B-band time of maximum light, which is
                used to define rest-frame phase [MJD]
        Added Keys
            MWEBVERR:           Uncertainty on MWEBV [mag]
            HOSTGAL_LOGMASSERR: Uncertainty on HOSTGAL_LOGMASS
            REDSHIFT_HELIOERR:  Uncertainty on REDSHIFT_HELIO
            REDSHIFT_CMBERR:    Uncertainty on REDSHIFT_CMB
            REDSHIFT_FINALERR:  Uncertainty on REDSHIFT_FINAL
    obs_df:
        Contains a row for each observation with the following columns:
            MJD: Phase of observation (rest-frame) with respect to peak_mjd [days]
            FLT: BayeSN name of the photometric filter used in the observation.
            flux: Unitless flux with fiducial zeropoint of either 27.5 or another key
                  specified in the file.
            flux_err: Uncertainty on the flux.
            mag: Apparent magnitudes
            mag_err: Uncertainty on those magnitudes
    """
    fh = open(fname, "r") if isinstance(fname, str) else fname
    lines = fh.readlines()
    fh.close()
    lines = [
        line.rstrip("\n").strip(" ").split(comment)[0] for line in lines
        if len(line) and not line.startswith(comment)
    ]

    # --- get sn_dict from file or compare given sn_dict with file values. ---
    SNID, z_hel, ra, dec = lines[0].split()
    z_hel, ra, dec = map(lambda x: float(x), (z_hel, ra, dec))
    z_cmb = utils.convert_z(z_hel, ra, dec, z_in_type="hel")
    MWEBV = utils.get_MWEBV(ra, dec)
    sn_dict = odict(zip(sn_dict_keys, [None for _ in sn_dict_keys]))
    sn_dict.update({
        "SNID": SNID,
        "RA": ra,
        "DECL": dec,
        "REDSHIFT_HELIO": z_hel,
        "REDSHIFT_CMB": z_cmb,
        "MWEBV": MWEBV,
        "ZP_FLUXCAL": fluxcal_zpt,
    })

    # --- get data from file ---
    filt_indices = [lines.index(l) for l in lines if l.startswith("filter")]
    all_filters = [lines[idx].split()[1] for idx in filt_indices]
    filt_indices.append(len(lines))  # stopping point for final block
    data_index_edges = [(edges[0]+1, edges[1]) for edges in pairwise(filt_indices)]
    N_obs = sum(edges[1] - edges[0] for edges in data_index_edges)
    file_data = np.empty((N_obs, len(obs_df_columns)), dtype=object)

    running_index = 0
    for filt, (start, end) in zip(all_filters, data_index_edges):
        # Don't need to vectorise since only N_bands filter names, not N_obs.
        filt_data = np.array([l.split() for l in lines[start:end]]).astype(float)
        # start/end track file line indices, not the indices of file_data.
        file_data_rows = slice(running_index, running_index + end-start)
        file_data[file_data_rows, :3] = filt_data  # t, mag, mag_err
        flux, flux_err = utils.mag_to_flux(
            mag=filt_data[:,1], mag_err=filt_data[:,2], zp=fluxcal_zpt
        )
        for i, val in zip(range(3,7), (flux, flux_err, filt)):
            file_data[file_data_rows, i] = val
        running_index += end - start
    t_obs, mag, mag_err, flux, flux_err, FLT = file_data.T

    obs_df = pd.DataFrame(
        np.array([t_obs, FLT, flux, flux_err, mag, mag_err]).T,
        columns=obs_df_columns,
    )
    return sn_dict, obs_df
###############
### Writing ###
###############
def _write_snana_lcfile(
    output_dir: str | Path,
    snname: str,
    sn_dict: odict,
    obs_df: pd.DataFrame,
    author: None | str = "anonymous",
    survey: None | str = None,
    paper: None | str = None,
    filename: None | str = None
) -> str:
    """ Private method that only supports writing from sn_dict and obs_df structures
    """
    output_dir = Path(output_dir)
    if not output_dir.exists(): output_dir.mkdir()

    fluxcal_zpt = sn_dict.get("ZPT_FLUXCAL")
    # Column which designates observations
    obs_df["VARLIST:"] = "OBS:"
    # Round fluxes and flux errors
    obs_df["FLUXCAL"] = np.round(obs_df["flux"], 4)
    obs_df["FLUXCALERR"] = np.round(obs_df["flux_err"], 4)
    obs_df["MAG"] = np.round(obs_df["mag"], 4)
    obs_df["MAGERR"] = np.round(obs_df["mag_err"], 4)

    # Reorder columns
    obs_df = obs_df[["VARLIST:", "MJD", "FLT", "FLUXCAL", "FLUXCALERR", "MAG", "MAGERR"]]

    # Divider for the header
    divider = "-" * 59

    # Preamble
    datestamp = time.strftime("%Y.%m.%d", time.localtime())
    timestamp = time.strftime("%H.%M hrs (%Z)", time.localtime())
    preamble_str = (
        f"# {sn_dict['SNID']}\n"
         "# SNANA-like file generated from user-provided data\n"
        f"# Zeropoint of the converted SNANA file: {fluxcal_zpt} mag\n"
        f"# {divider}\n"
        f"# Data table created by: {author}\n"
        f"# On date: {datestamp} (yyyy.mm.dd); {timestamp}.\n"
        f"# Script used: BayeSNmodel.io.write_snana_lcfile.py\n" +
        f"# {divider}\n"
    )
    # Metadata
    metadata_str_list = []
    # Removing metadata used for the table.
    for key in ("NOBS", "NVAR", "VARLIST"):
        sn_dict.pop(key, None)
    for key in sn_dict:
        if key.endswith("ERR"):
            continue  # skipping to conform to val +- err format
        err_key = f"{key}_ERR"
        if key in sn_dict and err_key in sn_dict:
            metadata_str_list.append(f"{key}: {sn_dict[key]} +- {sn_dict[err_key]}")
        elif key in sn_dict:
            metadata_str_list.append(f"{key}: {sn_dict[key]}")
    metadata_str_list.append(f"FILTERS: {','.join(obs_df['FLT'].unique())}")
    metadata_str = "\n".join(metadata_str_list) + f"\n# {divider}\n"

    # Data
    data_str = f"NOBS: {len(obs_df)}\nNVAR: 6\n{obs_df.to_string(index=False)}\nEND:"

    if filename is None:
        filename = snname + (survey is not None) * "_{}".format(survey) + (paper is not None) * "_{}".format(
            paper) + ".snana.dat"
    fpath = Path(output_dir, filename)
    with open(Path(output_dir, filename), "w") as f:
        f.write(preamble_str)
        f.write(metadata_str)
        f.write(data_str)
    return filename

def write_snana_lcfile(
    output_dir: str | Path,
    snname: str,
    sn_dict: None | odict[str, str|Number] = None,
    obs_df: None | pd.DataFrame = None,
    mjd: None | ArrayLike = None,
    flt: None | ArrayLike = None,
    flux: None | ArrayLike = None,
    flux_err: None | ArrayLike = None,
    mag: None | ArrayLike = None,
    mag_err: None | ArrayLike = None,
    fluxcal_zpt: Number = 27.5,
    author: None | str ="anonymous",
    survey: None | str = None,
    paper: None | str = None,
    filename: None | str = None,
    **kwargs
):
    """
    Write user data to an SNANA-like ascii light curve file

    Parameters
    ----------
    output_dir:
        Path to a directory where the file will be written. A default filename
        will be used, but you can specify your own with the `filename` argument.
        Default name format is `snname[_survey][_paper].snana.dat`, with the
        survey and/or paper being appended to the name if provided.
    snname:
        Name of the supernova
    sn_dict:
        Dictionary containing metadata for the supernova
    obs_df:
        Pandas DataFrame containing data for each observation of the supernova.
    mjd:
        Modified Julian Dates of observations
    flt:
        Filter idenitifiers of observations
    flux:
        Flux of observations
    flux_err:
        Flux errors of observations
    mag:
        Magnitudes of observations
    mag_err:
        Magnitude errors of observations
    author : str, optional
        Who is creating this file? Will be printed into the header's
        preamble, if desired
    survey : str, optional
        Optional argumanet specifying the survey the data came from. Will be
        written into the header and filename if provided.
    paper : str, optional
        Optional argument specifying the paper the data came from. Will be
        written into the filename if provided.
    filename : str, optional
        Custom filename to save as within `output_dir`. If not provided,
        a default format will be used. Do not provide an extension, as
        this will be added automatically.
    kwargs:
        These kwargs are for providing metadata. Only the following keys are supported
        ra, dec, tmax, z_helio, z_helio_err, z_cmb, z_cmb_err, ebv_mw,
        ebv_mew_err, vpec, vpec_err, host_logmass, host_logmass_err.

    Returns
    -------
    path : str
        Full path to the generated light curve file.

    Notes
    -----
    This will write a user's data to the SNANA-like file format readable by
    out I/O routines. It will write the provided metadata into the file
    header, so this will be read in and used correctly by BayeSN. All vital
    metadata are required as inputs to this function.
    """
    if sn_dict is not None and obs_df is not None:
        return _write_snana_lcfile(
            output_dir=output_dir,
            snname=snname,
            sn_dict=sn_dict,
            obs_df=obs_df,
            author=author,
            survey=survey,
            paper=paper,
            filename=filename
        )

    # Required data: mjd, flt, either flux/mag and either flux_err/mag_err
    if flux is not None and flux_err is not None:
        mag, mag_err = utils.flux_to_mag(flux, flux_err, zp=fluxcal_zpt, nan_val=-99)
    elif mag is not None and mag_err is not None:
        flux, flux_err = utils.mag_to_flux(mag, mag_err, zp=fluxcal_zpt)
    obs_df = pd.DataFrame({
        "MJD": mjd,
        "FLT": flt,
        # Following fields will be renamed in _write_snana_lcfile
        "flux": flux,
        "flux_err": flux_err,
        "mag": mag,
        "mag_err": mag_err
    })
    if None in obs_df.values:
        raise ValueError(
            "Missing required data: mjd, flt, and either flux/flux_err or mag/mag_err"
        )

    kwarg_map = {
        "ra": "RA",
        "dec": "DECL",
        "tmax": "SEARCH_PEAKMJD",
        "z_helio": "REDSHIFT_HELIO",
        "z_helio_err": "REDSHIFT_HELIO_ERR",
        "z_cmb": "REDSHIFT_CMB",
        "z_cmb_err": "REDSHIFT_CMB_ERR",
        "ebv_mw": "MWEBV",
        "ebv_mw_err": "MWEBV_ERR",
        "vpec": "VPEC",
        "vpec_err": "VPEC_ERR",
        "host_logmass": "HOSTGAL_LOGMASS",
        "host_logmass_err": "HOSTGAL_LOGMASS_ERR",
    }
    sn_dict = {"SNID": snname, "ZP_FLUXCAL": fluxcal_zpt}
    for key, val in kwargs.items():
        if key not in kwarg_map:
            raise KeyError(
                "The supported kwargs are limited to the following:\n"
                "ra, dec, tmax, z_helio, z_helio_err, z_cmb, z_cmb_err, ebv_mw, "
                "ebv_mew_err, vpec, vpec_err, host_logmass, host_logmass_err."
            )
        sn_dict[kwarg_map[key]] = float(val)

    return _write_snana_lcfile(
        output_dir=output_dir,
        snname=snname,
        sn_dict=sn_dict,
        obs_df=obs_df,
        author=author,
        survey=survey,
        paper=paper,
        filename=filename
    )
