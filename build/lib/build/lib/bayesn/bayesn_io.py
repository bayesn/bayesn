"""
BayeSN I/O Utilities. Currently defines function for writing output SNANA-format light curves for simulated light
curves
"""

import os
import numpy as np
import astropy.table as at
import sncosmo
import time


def write_snana_lcfile(output_dir, snname, mjd, flt, mag, magerr, tmax, z_helio, z_cmb, z_cmb_err, ebv_mw, ra=None,
                       dec=None, author="anonymous", survey=None, paper=None, filename=None):
    """
    Write user data to an SNANA-like light curve file

    Parameters
    ----------
    output_dir : str
        Path to a directory where the file will be written. A default filename
        will be used, but you can specify your own with the `filename` argument.
        Default name format is `snname[_survey][_paper].snana.dat`, with the
        survey and/or paper being appended to the name if provided.
    snname : str
        Name of the supernova
    mjd : list or :py:class:`numpy.array`
        Modified Julian Dates of observations
    flt : list or :py:class:`numpy.array` of str
        Filter idenitifiers of observations
    mag : list or :py:class:`numpy.array`
        Magnitudes of observations
    magerr : list or :py:class:`numpy.array`
        Magnitude errors of observations
    tmax : float
        Estimated time of maximum
    z_helio : float
        Heliocentric redshift
    z_cmb : float
        CMB-frame redshift
    z_cmb_err : float
        Error on CMB-frame redshift (excluding peculiar velocity uncertainty contribution)
    ebv_mw : float
        E(B-V) reddening due to the Milky Way
    ra : float, optional
        Right Ascension, to be writen to the header if desired
    dec :  float, optional
        Declination, to be written into the header if desired
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
    if not (len(mjd) == len(flt) == len(mag) == len(magerr)):
        raise ValueError("Provided columns are not the same length!")

    if not os.path.exists(output_dir):
        raise ValueError("Requested output directory does not exist!")

    tab = at.Table([mjd, flt, mag, magerr], names=["MJD", "FLT", "MAG", "MAGERR"])
    # Compute fluxcal and fluxcalerr
    tab["FLUXCAL"] = 10 ** ((27.5 - tab["MAG"]) / 2.5)
    tab["FLUXCALERR"] = tab["FLUXCAL"] * tab["MAGERR"] * np.log(10) / 2.5
    # Column which designates observations
    tab["VARLIST:"] = ["OBS:"] * len(tab)
    # Round fluxes and flux errors
    tab["FLUXCAL"] = np.round(tab["FLUXCAL"], 4)
    tab["FLUXCALERR"] = np.round(tab["FLUXCALERR"], 4)
    # Reorder columns
    tab = tab["VARLIST:", "MJD", "FLT", "FLUXCAL", "FLUXCALERR", "MAG", "MAGERR"]

    # Divider for the header
    divider = "-" * 59

    # Write a preamble to the metadata dictionary
    datestamp = time.strftime("%Y.%m.%d", time.localtime())
    timestamp = time.strftime("%H.%M hrs (%Z)", time.localtime())
    preamble = ("\n# SNANA-like file generated from user-provided data\n" +
                "# Zeropoint of the converted SNANA file: 27.5 mag\n" +
                "# {}\n".format(divider) +
                "# Data table created by: {}\n".format(author) +
                "# On date: {} (yyyy.mm.dd); {}.\n".format(datestamp, timestamp) +
                "# Script used: BayeSNmodel.io.write_snana_lcfile.py\n" +
                "# {}".format(divider))
    tab.meta = {"# {}".format(snname): preamble}

    # Add metadata
    tab.meta["SNID:"] = snname
    if survey is not None:
        tab.meta["SOURCE:"] = survey
    if ra is not None:
        tab.meta["RA:"] = ra
    if dec is not None:
        tab.meta["DEC:"] = dec
    filters = ",".join(at.unique(tab, keys="FLT")["FLT"])
    tab.meta.update(
        {"MWEBV:": ebv_mw, "REDSHIFT_HELIO:": z_helio, "REDSHIFT_CMB:": z_cmb, "REDSHIFT_CMB_ERR:": z_cmb_err,
         "PEAKMJD:": tmax, "FILTERS:": filters, "#": divider, "NOBS:": len(tab), "NVAR:": 6})

    # Write to file
    if filename is None:
        filename = snname + (survey is not None) * "_{}".format(survey) + (paper is not None) * "_{}".format(
            paper) + ".snana.dat"
    sncosmo.write_lc(tab, os.path.join(output_dir, filename), fmt="salt2", metachar="")

    # Write terminating line
    with open(os.path.join(output_dir, filename), "a") as f:
        f.write("END:")

    # Return filename
    return filename
