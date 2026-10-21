"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model.
"""

from __future__ import annotations

import argparse
import os
import copy
from collections import OrderedDict as odict
from collections.abc import Callable
import functools
from functools import partial
from numbers import Number
from pathlib import Path
import re
import subprocess
import sys
from warnings import warn

import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.table import Table, QTable
import astropy.units as u
import arviz
import h5py
import numpy as np
import numpyro
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_median,
    init_to_sample,
    init_to_value,
    Predictive,
    SVI,
    Trace_ELBO,
)
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer.util import log_density, _unconstrain_reparam
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoMultivariateNormal,
    AutoDiagonalNormal,
    AutoLaplaceApproximation,
)
from numpyro.handlers import substitute, trace
import pandas as pd
import pickle
import jax
from jax import device_put, jit, Array
from jax.lax import cond
import jax.numpy as jnp
from jax.random import PRNGKey, split
from jax.scipy.stats import norm
from jax.scipy.special import ndtri, ndtr
from jax.typing import ArrayLike
from jaxlib.xla_extension import ArrayImpl
from astropy.cosmology import FlatLambdaCDM
import matplotlib as mpl
from matplotlib import rc
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.stats import truncnorm, chi2
import sncosmo
import timeit
from ruamel.yaml import YAML
import time
from tqdm import tqdm
from typing import Any, NamedTuple

from .lm_optim import run_lm_laplace_gn, compute_gn_scale_tril
from .spline_utils import invKD, spline_coeffs, spline_coeffs_step
from .extinction_relations import DustExtRel
from .io import write_snana_lcfile, read_snana_spectra, read_snana_ascii
from bayesn.datasets import SNDataset, ObsData
from bayesn.utils import _predict, _prior_pot, find_data_dir_in_SNANA
import bayesn.zltn_utils as zltn
from bayesn import constants

yaml = YAML(typ="safe")
yaml.default_flow_style = False

jax.config.update("jax_enable_x64", True)  # Enables 64 computation

np.seterr(divide="ignore", invalid="ignore")  # Disable divide by zero warnings

# jax.config.update("jax_platform_name", "cpu")  # Forces CPU

BASE_DIR: Path = Path(__file__).parent.absolute()
with open(BASE_DIR / "defaults.yaml", "r") as file:
    default_kwargs = yaml.load(file)
default_kwargs["AV_dist"] = dist.Exponential


class DustHyperParams(NamedTuple):
    """
    Container for population-level dust parameters.

    Attributes
    ----------
    sigma0
        Achromatic intrinsic scatter scale.
    tauA
        Scale parameter for the exponential distribution of host-galaxy dust extinction.
    mu_R
        Mean of the host-galaxy RV distribution, if inferred.
    sigma_R
        Standard deviation of the host-galaxy RV distribution, if inferred.
    uniform_RV_min
        Lower bound of uniform RV distribution, if applicable.
    uniform_RV_max
        Upper bound of uniform RV distribution, if applicable.
    mu_z_grad
        Redshift evolution gradient for mu_R.
    tau_z_grad
        Redshift evolution gradient for tauA.
    RV
        Shared host-galaxy RV value or status string.
    """
    sigma0: float
    tauA: float
    mu_R: float | None
    sigma_R: float | None
    uniform_RV_min: float | None
    uniform_RV_max: float | None
    mu_z_grad: float
    tau_z_grad: float
    RV: Number | str


class DustPop(NamedTuple):
    """
    Container for high-mass and low-mass population dust parameters.

    Attributes
    ----------
    HM
        Dust hyperparameters for the high-mass host galaxy population.
    LM
        Dust hyperparameters for the low-mass host galaxy population, if modeled separately.
    HM_flag : (N_sn,)
        Boolean array indicating whether each SN belongs to the high-mass host population.
    sigma0 : (N_sn,) or scalar
        Achromatic intrinsic scatter per SN or shared across the sample.
    """
    HM: DustHyperParams
    LM: DustHyperParams | None = None
    HM_flag: ArrayLike | None = None
    sigma0: ArrayLike | float | None = None


class SEDmodel(object):
    """
    BayeSN-SED Model.

    Class which imports a BayeSN model, and allows one to fit or simulate
    Type Ia supernovae based on this model.

    Methods
    -------
    get_flux_batch
        Get integrated fluxes across a large number of SNe, phases, and bands.
    get_flux_from_chains
        Get model photometry for posterior samples from model fitting chains.
    get_mag_batch
        Get magnitudes across a large number of SNe, phases, and bands.
    get_spectra
        Get spectra for a large number of SNe and phases. Wraps a more performant,
        lower-level method that has more restrictive arguments.
    initial_guess
        Defined method used to initialise chains for model training.
    parse_args
        Parse the args from the input yaml file along with any command line arguments
        to define the job being run.
    postprocess
        Postprocess the output of the MCMC run if required and save the chains and
        summaries.
    process_dataset
        Process a set of data for use by the BayeSN model. Calls lower-level methods
        depending on the input yaml for use with version photometry or a data table.
    run
        Run an inference job using the BayeSN model.
    sample_AV
        Sample AV from the population distribution based on a pre-trained tauA.
    sample_del_M
        Sample delta_M from the population distribution based on a pre-trained sigma0.
    sample_epsilon
        Sample epsilon from the population distribution based on a pre-trained L_Sigma.
    sample_theta
        Sample theta from the standard normal distribution.
    simulate_light_curve
        Simulate a light curve or set of light curves from the BayeSN SED model.
    simulate_spectrum
        Simulate a spectrum or set of spectra from the BayeSN SED model.
    _model
        Defines the BayeSN parameters to infer conditioned on input data.
        Calls helper functions that specify the components in a modular fashion.

    Attributes
    ----------
    __root_dir__ : Path
        Absolute path to the bayesn/bayesn directory.
    example_lc : Path
        Absolute path to a SNANA-format ascii file light curve.
    sim : bool
        Indicates whether data is simulated.
    cosmo : astropy.cosmology.FlatLambdaCDM
        Defines the fiducial cosmology assumed by the model when training.
    sigma_pec : Array (), default jnp.array(150/constants.C_LIGHT)
        Peculiar velocity to be used in calculating redshift uncertainties.
    apply_dovekie_mag_shifts : bool, default True
        Whether to add Dovekie magnitude updates.
    shift_df : None | pd.DataFrame, default None
        DataFrame with columns 'BAND', 'MAG_SHIFT', 'LAM_SHIFT' overriding shifts.
    apply_mag_shifts : bool, default False
        Whether to add 'MAG_SHIFT' values from ``shift_df``.
    apply_lam_shifts : bool, default False
        Whether to increase transmission wavelengths by 'LAM_SHIFT' values.
    ZPT : Number, default 27.5
        The Pogson zero point used for converting between fluxes and magnitudes.
    RV_MW : Array (), default jnp.array(3.1)
        RV value for calculating Milky Way extinction.
    trunc_val : Number, default 1.2
        Lower limit for RV based on pure Rayleigh Scattering.
    spectrum_bins : int, default 300
        Number of wavelength bins used for modelling spectra.
    band_oversampling : int, default 51
        Affects high-resolution wavelength grid spacing.
    max_redshift : Number, default 4
        Maximum redshift for mapping observer-frame bandpasses to rest-frame.
    hsiao_l : Array (N_hsiao_l,)
        Wavelengths (Angstroms) where the Hsiao SED time-series is defined.
    hsiao_t : Array (N_hsiao_t,)
        Rest-frame phases (days) where the Hsiao SED time-series is defined.
    min_hsiao_wave : Number
        The bluest wavelength in ``hsiao_l``.
    max_hsiao_wave : Number
        The reddest wavelength in ``hsiao_l``.
    hsiao_offset : Number
        The difference between t=0 and earliest phase in ``hsiao_t``.
    J_l_T_hsiao : Array (spectrum_bins, N_hsiao_l + 1)
        Matrix for interpolating Hsiao spline to ``model_wave``.
    KD_t_hsiao : Array (N_hsiao_t, N_hsiao_t + 1)
        Matrix for phase spline interpolation of Hsiao template.
    hsiao_flux : Array (spectrum_bins, N_hsiao_t)
        Hsiao SED fluxes interpolated to ``model_wave``.
    ext_rel : DustExtRel
        The extinction relation parametrizing A(x)/A(V) as a function of RV and wavelength.
    model_name : str
        The name of a loaded built-in model or path to a custom model.
    l_knots : Array (N_l_knots,)
        Wavelength knots defining the model's warping surfaces.
    tau_knots : Array (N_tau_knots,)
        Phase knots defining the model's warping surfaces.
    M0 : Array ()
        Reference absolute magnitude scaling the Hsiao template.
    sigma0 : Array ()
        Scale factor for achromatic intrinsic scatter.
    tauA : Array ()
        Scale factor for the exponential distribution of host-galaxy A_V.
    W0 : Array (N_l_knots, N_tau_knots)
        Mean SED warping surface.
    W1 : Array (N_l_knots, N_tau_knots)
        First functional principal component warping surface.
    L_Sigma : Array (N_knots_sig, N_knots_sig)
        Covariance matrix describing the epsilon perturbation distribution.
    RV : None | Array ()
        Shared host-galaxy RV value, if applicable.
    mu_R : None | Array ()
        Mean of the host-galaxy RV distribution, if applicable.
    sigma_R : None | Array ()
        Standard deviation of the host-galaxy RV distribution, if applicable.
    N_knots_sig_l : int
        Number of active wavelength knots for epsilon perturbations.
    N_knots_sig : int
        Total number of epsilon perturbation knots.
    min_wave : Array ()
        The bluest wavelength knot.
    max_wave : Array ()
        The reddest wavelength knot.
    model_wave : Array (spectrum_bins,)
        Rest-frame model wavelength vector.
    hires_wave : Array (N_hires,)
        High-resolution log-uniform wavelength vector.
    hires_spacing : Array ()
        Log10 spacing between neighbouring elements in ``hires_wave``.
    J_l_T : Array (spectrum_bins, N_l_knots + 1)
        Wavelength spline interpolation matrix.
    KD_t : Array (N_tau_knots, N_tau_knots + 1)
        Phase spline boundary condition solver matrix.
    J_t_map : Callable
        JIT-compiled vmap of ``spline_coeffs_step``.
    mw_ext : Array (N_hires,)
        High-resolution vector of Milky Way extinction values.
    photoz : bool
        Whether photometric redshift is sampled.
    data : ObsData
        NamedTuple populated by ``process_dataset``.
    dataset : SNDataset
        SNDataset populated by ``process_dataset``.
    band_weights : Array (N_sn, spectrum_bins, N_used_bands)
        Rest-frame transformed transmission weights for data bands.
    J_t : Array (N_sn, N_tau_knots + 1, N_max_epochs)
        Phase spline interpolation matrix for observations.
    hsiao_interp : Array (3, N_max_epochs, N_sn)
        Hsiao template phase interpolation coordinates.
    z_u_grid : None | Array (N_z_u,)
        CDF probability levels for host photo-z quantiles.
    z_icdf_grid : None | Array (N_sn, N_z_u)
        SN-specific redshift values at photo-z quantile grid points.
    fitres_table : None | QTable
        Populated by ``process_dataset``.
    lcplot_data : None | pd.DataFrame
        Populated by ``process_dataset``.
    filter_yaml : None | Path
        Path to custom filter YAML file.
    filter_dict : dict[str, dict[str, str | ArrayLike]]
        Dictionary of filter and standard transmission profiles.
    dovekie_labels : np.ndarray (N_dovekie,)
        BayeSN names of bandpasses with Dovekie cross-calibration.
    dovekie_cov : Array (N_dovekie, N_dovekie)
        Covariance matrix of zero-point shifts from Dovekie.
    band_dict : dict[str, int], N_bands keys
        Mapping from BayeSN bandpass name to loaded index.
    band_interpolate_weights : Array (N_bands, N_hires)
        Upsampled observer-frame transmission profiles on ``hires_wave``.
    band_lim_dict : dict[str, tuple[Number, Number]] N_bands keys
        Cut-on and cut-off wavelengths (1% level) for each bandpass.
    zp_dict : dict[str, Number], N_bands keys
        Zero-point mapping for loaded bandpasses.
    zps : Array (N_bands,)
        Instrumental zero-points in the order of ``band_dict``.
    calib_cov : Array (N_bands, N_bands)
        Covariance matrix of zero-point shifts across loaded bands.
    wave_sigmas : Array (N_bands,)
        Uncertainties in wavelength shifts across loaded bands.
    used_band_inds : np.ndarray (N_used_bands,)
        Indices into ``band_dict`` for currently used bands.
    used_band_dict : dict[int, int], N_used_bands keys
        Mapping from loaded band index to active dataset band index.
    used_zps : Array (N_used_bands,)
        Zero-points for active bands in dataset.
    used_calib_cov : Array (N_used_bands, N_used_bands)
        Sub-matrix of calibration covariance for active bands.
    used_calib_chcov : Array (N_used_bands, N_used_bands)
        Cholesky factor of ``used_calib_cov``.
    used_wave_sigmas : Array (N_used_bands,)
        Wavelength shift uncertainties for active bands.

    Notes
    -----
    Attribute Organization
    ~~~~~~~~~~~~~~~~~~~~~~

    General Attributes
    ^^^^^^^^^^^^^^^^^^
    * ``__root_dir__``: Absolute path to the bayesn/bayesn directory.
    * ``example_lc``: Absolute path to a SNANA-format ascii file light curve.
    * ``sim``: Indicates whether data is simulated.
    * ``cosmo``: Defines the fiducial cosmology assumed by the model when training.
      Can be anything that has a ``distmod`` method to convert cosmological redshifts
      to distance moduli.
    * ``sigma_pec``: Peculiar velocity to be used in calculating redshift uncertainties.
    * ``apply_dovekie_mag_shifts``: Whether to add the mag_update and mag_cal elements
      from ``filters.yaml`` to defined magnitudes based on the Dovekie analysis
      (Popovic et al. 2025).
    * ``shift_df``: Optional DataFrame with columns 'BAND', 'MAG_SHIFT', 'LAM_SHIFT'
      overriding defaults from ``filters.yaml``.
    * ``apply_mag_shifts``: Whether to add 'MAG_SHIFT' values from ``shift_df``.
    * ``apply_lam_shifts``: Whether to shift transmission wavelengths by 'LAM_SHIFT' values.
    * ``ZPT``: The Pogson zero point used for converting between fluxes and magnitudes.
    * ``RV_MW``: RV value for calculating Milky Way extinction.
    * ``trunc_val``: Lower limit for RV based on pure Rayleigh Scattering.
    * ``spectrum_bins``: Number of wavelength bins used for modelling spectra (default 300).
    * ``band_oversampling``: Oversampling factor for high-resolution wavelength grid (default 51).
    * ``max_redshift``: Maximum redshift for mapping observer-frame bandpasses to rest-frame.
    * ``hsiao_l``: Wavelengths (Angstroms) where the Hsiao SED time-series is defined.
    * ``hsiao_t``: Rest-frame phases (days) where the Hsiao SED time-series is defined.
    * ``min_hsiao_wave`` / ``max_hsiao_wave``: Bluest and reddest wavelengths in ``hsiao_l``.
    * ``hsiao_offset``: Difference between t=0 and earliest phase in ``hsiao_t``.
    * ``J_l_T_hsiao``: Matrix for interpolating Hsiao spline to ``model_wave``.
    * ``KD_t_hsiao``: Matrix for phase spline interpolation of Hsiao template.
    * ``hsiao_flux``: Hsiao SED fluxes interpolated to ``model_wave``.
    * ``ext_rel``: The extinction relation parametrizing A(x)/A(V) as a function of RV
      and wavelength.

    Definable Model-Specific Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * ``model_name``: The name of a loaded built-in model or path to a custom model.
    * ``l_knots``: Array of wavelength knots defining the model's warping surfaces.
    * ``tau_knots``: Array of phase knots defining the model's warping surfaces.
    * ``M0``: Reference absolute magnitude for scaling the Hsiao template.
    * ``sigma0``: Scale factor for the achromatic offset parameter.
    * ``tauA``: Scale factor for the Exponential distribution of host-galaxy A_V.
    * ``W0``: Mean SED warping surface of shape (N_l_knots, N_tau_knots).
    * ``W1``: First functional principal component warping surface of shape (N_l_knots, N_tau_knots).
    * ``L_Sigma``: Covariance matrix describing the epsilon perturbation distribution.
    * ``RV``: Shared host-galaxy RV value for models with ``shared_RV=True``.
    * ``mu_R``: Mean of host-galaxy RV distribution for models with ``shared_RV=False``.
    * ``sigma_R``: Standard deviation of host-galaxy RV distribution for models with ``shared_RV=False``.

    Calculated Model-Specific Attributes
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * ``N_knots_sig_l``: Active wavelength knots (N_l_knots - 2).
    * ``N_knots_sig``: Total epsilon perturbation knots (N_knots_sig_l * N_tau_knots).
    * ``min_wave`` / ``max_wave``: Bluest and reddest wavelength knots.
    * ``model_wave``: Rest-frame model wavelength vector of shape (spectrum_bins,).
    * ``hires_wave``: High-resolution log-uniform wavelength vector of length N_hires.
    * ``hires_spacing``: Log10 spacing between neighbouring elements in ``hires_wave``.
    * ``J_l_T``: Spline interpolation matrix of shape (spectrum_bins, N_l_knots + 1).
    * ``KD_t``: Boundary condition matrix of shape (N_tau_knots, N_tau_knots + 1).
    * ``J_t_map``: JIT-compiled vmap of ``spline_coeffs_step``.
    * ``mw_ext``: High resolution vector of Milky Way extinction values of shape (N_hires,).

    Attributes Affecting Likelihood Construction
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * ``photoz``: Indicates whether redshift should be sampled.

    Attributes Based on Data
    ^^^^^^^^^^^^^^^^^^^^^^^^
    * ``data``: ObsData NamedTuple populated by ``process_dataset``.
    * ``dataset``: SNDataset dataclass populated by ``process_dataset``.
    * ``band_weights``: Rest-frame transformed band weights of shape (N_sn, spectrum_bins, N_used_bands).
    * ``J_t``: Phase spline interpolation matrix of shape (N_sn, N_tau_knots + 1, N_max_epochs).
    * ``hsiao_interp``: Linear interpolation coordinates for Hsiao template of shape (3, N_max_epochs, N_sn).
    * ``z_u_grid``: Quantile probability levels for host photo-z grid.
    * ``z_icdf_grid``: SN-specific redshift values at photo-z grid points.
    * ``fitres_table``: Summary QTable populated by ``process_dataset``.
    * ``lcplot_data``: Light curve observation DataFrame populated by ``process_dataset``.

    Bandpass Management
    ^^^^^^^^^^^^^^^^^^^
    Attributes are organised into three hierarchical levels:

    Level 1 (Loaded during initialisation)
        * ``filter_yaml``: Path to custom filter YAML configuration.
        * ``filter_dict``: Dictionary containing 'standards' and 'filters' transmission data.
        * ``dovekie_labels``: BayeSN names of bandpasses with Dovekie calibrations.
        * ``dovekie_cov``: Covariance matrix of zero-point shifts from Dovekie.

    Level 2 (Loaded as needed and cached)
        * ``band_dict``: Mapping from BayeSN bandpass names to loaded order.
        * ``band_interpolate_weights``: Linear interpolation of transmission functions on ``hires_wave``.
        * ``band_lim_dict``: 1% cut-on and cut-off wavelengths for each bandpass.
        * ``zp_dict``: Instrumental zero-points for loaded bandpasses.
        * ``zps``: Values of ``zp_dict`` in loaded order.
        * ``calib_cov``: Covariance matrix of zero-point shifts for loaded bandpasses.
        * ``wave_sigmas``: Uncertainties in wavelength shifts for loaded bandpasses.

    Level 3 (Restricted to active dataset)
        Populated by ``_set_used_bands`` for the active dataset:
        * ``used_band_inds``: Band indices in dataset order.
        * ``used_band_dict``: Active dataset index lookup.
        * ``used_zps``: Zero-points in dataset order.
        * ``used_calib_cov``: Sub-matrix of calibration covariance in dataset order.
        * ``used_calib_chcov``: Cholesky decomposition of ``used_calib_cov``.
        * ``used_wave_sigmas``: Wavelength shift uncertainties in dataset order.
    """
    ######################
    ### Initialisation ###
    ######################
    def __init__(
        self,
        num_devices: int = 4,
        load_model: str = "T21_model",
        filter_yaml: Path | str | None = None,
        fiducial_cosmology: dict[str, float] = {"H0": 73.24, "Om0": 0.28},
        load_ext_rel: str = "G23",
        apply_dovekie_mag_shifts: bool = True,
        shift_file: str | None = None,
        apply_mag_shifts: bool = False,
        apply_lam_shifts: bool = False,
        fluxcal_zpt: float = 27.50,
    ):
        """
        Initialises the BayeSN SED model by loading a pre-computed model, transmission
        functions, and dust extinction relation.

        Parameters
        ----------
        num_devices
            If running on a CPU, numpyro will by default see it as a single device.
            This argument will set the number of available cores for numpyro to use
            (e.g., set to 4, you can train 4 chains on 4 cores in parallel).
        load_model
            Can be either a pre-defined BayeSN model name (see list below), or
            a path to directory containing a set of .txt files from which a
            valid model can be constructed. Currently implemented default models:

            'G26_model'
                Grayling+26 BayeSN model (arXiv:2606.19429). Covers rest wavelength
                range of 2800-10800A (ugriz). Intended for cosmology; jointly fits
                filter wavelength and zero-point cross-calibration shifts alongside
                the SED, using Dovekie as an informative prior. Population RV
                distribution. Trained on 1024 SNe Ia from the Kenworthy+21
                compilation with Dovekie calibration updates (Foundation, CfA3,
                CfA4, CSP, SDSS, PS1, DES, SNLS).
            'G25_model'
                Grayling+26 phase-extended optical+NIR BayeSN model (MNRAS 548,
                stag340; arXiv:2510.11719; BayeSN-TD). Covers rest wavelength range
                of 2800-18500A (UBgVrizYJH) and phase range -10 to +85 days,
                motivated by fitting late-time observations of strongly-lensed
                SNe Ia. Population RV distribution. Trained on 278 SNe Ia combining
                Avelino+19 low-z compilation, Foundation DR1 (Foley+18, Jones+19),
                and CSP-I.
            'W22_model'
                Ward+22 No-Split BayeSN model (ApJ 956, 111; arXiv:2209.10558).
                Covers rest wavelength range of 3000-18500A (BVRIYJH). No treatment
                of host mass effects. Global RV assumed. Trained on Foundation DR1
                (Foley+18, Jones+19) and low-z Avelino+19 (ApJ, 887, 106) compilation
                of CfA, CSP and others.
            'T21_model'
                Thorp+21 No-Split BayeSN model (arXiv:2102:05678). Covers rest
                wavelength range of 3500-9500A (griz). No treatment of host mass
                effects. Global RV assumed. Trained on Foundation DR1 (Foley+18,
                Jones+19).
            'M20_model'
                Mandel+20 BayeSN model (arXiv:2008.07538). Covers rest wavelength
                range of 3000-18500A (BVRIYJH). No treatment of host mass effects.
                Global RV assumed. Trained on low-z Avelino+19 (ApJ, 887, 106)
                compilation of CfA, CSP and others.
        fiducial_cosmology
            Dictionary containing keys "H0" and "Om0" for initialising an
            ``astropy.cosmology.FlatLambdaCDM`` instance.
            Default from Riess+16 (ApJ, 826, 56).
        filter_yaml
            Path to yaml file containing details on filters and standards to use.
            If not specified, will look for a file called ``filters.yaml`` in
            directory that BayeSN is called from.
        load_ext_rel
            Name of dust extinction relation to load. Available choices:

            * 'CCM89' : Cardelli, Clayton & Mathis (1989)
            * 'C94' : Cardelli (1994)
            * 'O94' : O'Donnell (1994)
            * 'F99' : Fitzpatrick (1999)
            * 'F99_AVGLMC' : Fitzpatrick (1999) Average LMC
            * 'F99_LMC2' : Fitzpatrick (1999) LMC2 supershell
            * 'F99_SNANA' : Fitzpatrick (1999) as implemented in SNANA
            * 'C00' : Calzetti et al. (2000)
            * 'F04' : Fitzpatrick (2004)
            * 'VCG04' : Valencic, Clayton & Gordon (2004)
            * 'GCC09' : Gordon, Cartledge & Clayton (2009)
            * 'F19' : Fitzpatrick et al. (2019)
            * 'D22' : Decleir et al. (2022)
            * 'G23' : Gordon et al. (2023)
        apply_dovekie_mag_shifts
            Whether to add the ``mag_update`` and ``mag_cal`` elements from the
            ``filters.yaml`` file to defined magnitudes. These values are based on
            the Dovekie analysis (Popovic et al. 2025).
        shift_file
            Path to a CSV file with columns 'BAND', 'MAG_SHIFT', and 'LAM_SHIFT'
            overriding the defaults from ``filters.yaml``:

            * 'BAND' : bandpass names in BayeSN convention.
            * 'MAG_SHIFT' : magnitude value to add to defined magnitude (zero-point).
            * 'LAM_SHIFT' : Angstrom value to add to transmission function wavelengths.
        apply_mag_shifts
            Whether to add any 'MAG_SHIFT' values from ``shift_df`` / ``shift_file``
            to defined magnitudes. Overrides ``apply_dovekie_mag_shifts``.
        apply_lam_shifts
            Whether to increase transmission functions by any 'LAM_SHIFT' values.
        fluxcal_zpt
            The Pogson zero point used for converting between fluxes and magnitudes:
            ``mag = fluxcal_zpt - 2.5 * log10(flux)``.
        """

        # Settings for jax/numpyro
        numpyro.set_host_device_count(num_devices)
        self.start_time = time.time()
        self.end_time = None
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        print("Current devices:", jax.devices())

        self.__root_dir__ = BASE_DIR
        print(f"Currently working in {os.getcwd()}")

        self.filter_yaml = Path(filter_yaml) if filter_yaml is not None else filter_yaml
        built_in_models = [f.name for f in self.__root_dir__.glob("model_files/*_model")]

        # Model-independent terms
        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.shift_df = None
        if shift_file is not None:
            if not Path(shift_file).exists():
                raise FileNotFoundError(f'Specified shift file {shift_file} does not exist')
            self.shift_df = pd.read_csv(shift_file, comment='#')
            req_cols = {"BAND", "LAM_SHIFT", "MAG_SHIFT"}
            missing_cols = req_cols.difference(self.shift_df.columns)
            if len(missing_cols):
                raise ValueError(
                    f"shift_file {shift_file} lacks required columns: {missing_cols}"
                )
        self.apply_dovekie_mag_shifts = apply_dovekie_mag_shifts
        self.apply_mag_shifts = apply_mag_shifts
        self.apply_lam_shifts = apply_lam_shifts
        self.ZPT = fluxcal_zpt  # Common fluxcal zero point for all bands
        self.RV_MW = device_put(jnp.array(3.1))
        self.sigma_pec = device_put(jnp.array(150 / 3e5))
        self.trunc_val = 1.2  # lower limit for RV based on pure Rayleigh Scattering
        self.spectrum_bins = 300
        self.band_oversampling = 51
        self.max_redshift = 4
        self.sim = False  # Keep track of whether data is simulated
        # Define example light curve for jupyter notebook demos
        self.example_lc = self.__root_dir__ / "data/example_lcs/Foundation_DR1_2016W.txt"

        # Model-dependent terms
        self.model_name = load_model
        if Path(load_model).exists():
            print(f"Loading custom model at {load_model}")
            with open(load_model, "r") as file:
                params = yaml.load(file)
        elif load_model in built_in_models:
            print(f"Loading built-in model {load_model}")
            with open(
                self.__root_dir__ / "model_files" /  load_model / "BAYESN.YAML",
                "r",
            ) as file:
                params = yaml.load(file)
        else:
            raise FileNotFoundError(
                f"Specified model {load_model} does not exist and does not correspond to one "
                f"of the built-in model {built_in_models}"
            )
        self.l_knots = jnp.array(params["L_KNOTS"])
        self.tau_knots = jnp.array(params["TAU_KNOTS"])
        self.N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        self.N_knots_sig_l = self.l_knots.shape[0] - 2
        self.N_knots_sig = (self.N_knots_sig_l) * self.tau_knots.shape[0]
        self.W0 = jnp.array(params["W0"])
        self.W1 = jnp.array(params["W1"])
        self.L_Sigma = jnp.array(params["L_SIGMA_EPSILON"])
        self.M0 = jnp.array(params["M0"])
        self.sigma0 = jnp.array(params["SIGMA0"])
        self.tauA = jnp.array(params["TAUA"])
        for key in ("RV", "mu_R", "sigma_R"):
            setattr(self, key, None)
        if "RV" in params:
            self.RV = jnp.array(params["RV"])
        elif "MUR" in params:
            self.mu_R = jnp.array(params["MUR"])
            self.sigma_R = jnp.array(params["SIGMAR"])
        # Build the model wavelengths in log space
        self.min_wave = min(self.l_knots)
        self.max_wave = max(self.l_knots)
        model_log_wave = jnp.linspace(
            jnp.log10(self.min_wave), jnp.log10(self.max_wave), self.spectrum_bins
        )
        self.model_wave = 10 ** model_log_wave

        # Similarly build a high-resolution model based on the Hsiao template
        self._load_hsiao_template()  # sets self.{min/max}_hsiao_wave
        hsiao_log_wave = jnp.linspace(
            jnp.log10(self.min_hsiao_wave),
            jnp.log10(self.max_hsiao_wave),
            self.spectrum_bins
        )
        self.hires_spacing = (hsiao_log_wave[1] - hsiao_log_wave[0])/self.band_oversampling
        hsiao_max_log_wave = (
            jnp.log10(self.max_hsiao_wave * (1 + self.max_redshift))
            + self.hires_spacing
        )
        hires_log_wave = jnp.arange(
            jnp.log10(self.min_hsiao_wave), hsiao_max_log_wave, self.hires_spacing
        )
        self.hires_wave = 10**hires_log_wave

        # Eventually various observer-frame transmission functions will be manipulated
        # to sample the rest-frame SED. There are some computations that can be done
        # now and scaled by arbitrary redshifts later.
        KD_l = invKD(self.l_knots)
        self.J_l_T = device_put(spline_coeffs(self.model_wave, self.l_knots, KD_l))
        self.KD_t = device_put(invKD(self.tau_knots))

        self.load_ext_rel(load_ext_rel)
        self._init_band_weights()
        self._load_dovekie_cov()
        self.J_t_map = jit(
            jax.vmap(spline_coeffs_step, in_axes=(0, None, None, None)),
            static_argnames=("extrap",)
        )
        self.photoz = False # gate phase extrapolation (only needed when z floats)

        # Initialising terms that will be populated later.
        for attr in (
            "data",
            "dataset",
            "band_weights",
            "J_t",
            "hsiao_interp",
            "z_u_grid",
            "z_icdf_grid",
            "all_table",
            "fitres_table",
            "lcplot_data",
        ):
            setattr(self, attr, None)

    def load_ext_rel(self, ext_rel_name: str) -> None:
        """
        Change the extinction relation and recalculate Milky Way extinction.

        Changing the ``ext_rel`` attribute directly does not recalculate ``mw_ext``.
        This method allows changing the extinction relation cleanly.

        Parameters
        ----------
        ext_rel_name
            Name of dust extinction relation to load (e.g. 'G23', 'F19', 'CCM89').
        """
        hires = DustExtRel(ext_rel_name, x_in=self.hires_wave, verbose=False)
        self.mw_ext = hires.get_axav(self.RV_MW, verbose=False)[0]
        self.ext_rel = DustExtRel(ext_rel_name, x_in=self.model_wave)

    def _load_hsiao_template(self, file_path: None | str | Path = None) -> None:
        """
        Loads the Hsiao template from the internal HDF5 file or a custom path.
        Supports:
            h5 files with "default" group name and datasets "phase", "wave", and "flux"
            dat files with space separated columns in the order phase, wave, and flux
        Stores the template as attributes {KD_t, J_l_T}_hsiao and hsiao_{t, l, flux}.
        """
        # Loading hsiao_{phase, wave, flux}
        default_file_path = self.__root_dir__ / "data/hsiao.h5"
        file_path = default_file_path if file_path is None else Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Hsiao template at file_path {file_path} does not exist. Use "
                "file_path=None to load the file in {default_file_path}."
            )
        if str(file_path).endswith("h5"):
            with h5py.File(file_path, "r") as file:
                data = file["default"]
                # Empty tuple index retrieves all scalar data.
                hsiao_phase = data["phase"][()].astype("float64")
                hsiao_wave = data["wave"][()].astype("float64")
                hsiao_flux = data["flux"][()].astype("float64")
        else:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, comment="#")
            try:  # Check for string row, indicating header
                float(df.iloc[0,0])
            except ValueError:
                df = df.drop(df.index[0]).reset_index(drop=True).astype(float)
            df.columns = ["phase", "wave", "flux"]
            hsiao_phase = np.sort(np.unique(df["phase"]))
            hsiao_wave = np.sort(np.unique(df["wave"]))
            if (np.diff(df["phase"]) >= 0).all():
                # monotonic in phase, reshape C style
                hsiao_flux = np.array(df["flux"]).reshape(
                    len(hsiao_phase), len(hsiao_wave)
                )
            elif (np.diff(df["wave"]) >= 0).all():
                # monotonic in wave, reshape F style
                hsiao_flux = np.array(df["flux"]).reshape(
                    len(hsiao_phase), len(hsiao_wave), order="F"
                )

        # Storing relevant data in SEDmodel
        self.min_hsiao_wave = hsiao_wave.min()
        self.max_hsiao_wave = hsiao_wave.max()
        KD_l_hsiao = invKD(hsiao_wave)
        self.KD_t_hsiao = device_put(invKD(hsiao_phase))
        self.J_l_T_hsiao = device_put(
            spline_coeffs(self.model_wave, hsiao_wave, KD_l_hsiao)
        )

        self.hsiao_t = device_put(hsiao_phase)
        self.hsiao_offset = int(-hsiao_phase[0]) # phase -> template index shift
        self.hsiao_l = device_put(hsiao_wave)
        self.hsiao_flux = device_put(hsiao_flux.T)
        padded_flux = jnp.concatenate([jnp.ones((1, self.hsiao_flux.shape[1])), self.hsiao_flux])
        self.hsiao_flux = device_put(jnp.matmul(self.J_l_T_hsiao, padded_flux))

    def _init_band_weights(self) -> None:
        """
        Band weights (modified transmission functions) will be needed for using
        photometry from different instruments. The band weights will be handled at
        three levels to avoid unnecessary computation / memory usage.
        Level 1:
            The light-weight metadata from filters.yaml and any custom filters defined
            by the user, loaded into the SEDmodel.filter_dict attribute.
        Level 2:
            A cache of actual band weights calculated from transmission functions will
            be added to the following SEDmodel attributes when first needed:
                band_dict : dict[str, int]
                    Keys are BayeSN convention bandpass names and values are integers
                    corresponding to their alphabetical rank.
                zp_dict : dict[str, float]
                    Keys are BayeSN convention bandpass names and values are floats
                    corresponding to the "zero-point" in the filters.yaml file with any
                    magcal, magupdate, or magshift modifiers applied.
                band_lim_dict : tuple(float, float)
                    A 2-tuple indicating the lower and upper wavelengths limits between
                    which the transmission is defined. The limits are defined by the
                    transmission function reaching 1% of its maximum throughput.
                band_interpolate_weights :
                zps : Array shape (N_bandpasses)
                    The values of zp_dict.
                wave_sigmas : Array shape (N_bandpasses)
                    Prior Gaussian uncertainties for wavelength shifts.
                calib_cov : Array shape (N_bandpasses, N_bandpasses)
                    Prior Gaussian covariance matrix for magnitude offsets.
        Level 3:
            A subset of level 2 data needed for a particular inference. The information
            will be stored in the following SEDmodel attributes as part of a fit method
            or process_dataset invocation:
                used_band_inds
                used_band_dict
                used_zps
                used_calib_cov
                used_calib_chcov
                used_wave_sigmas

        This specific method initialises attributes at all three levels, populated with
        a NULL band, which is a fake band with a very wide wavelength range used only
        for padded data points to ensure that these padded data points never fall out
        of the wavelength coverage of the model. These padded data points will not
        contribute to the likelihood in any way and are entirely for computational
        reasons.
        Additionally, this method loads the level 1 metadata from the filters.yaml file.

        This code is partly based off ParSNiP from Boone+21.
        """
        # Store metadata at level 1
        self.filter_dict = self._load_filter_dict()

        # Instantiate level 2 attributes with the NULL band
        self.band_dict = {"NULL_BAND": 0}
        self.zp_dict = {"NULL_BAND": 10} # arbitrary number
        self.zps = jnp.array(list(self.zp_dict.values()))
        self.band_lim_dict = {"NULL_BAND": (self.hires_wave[0], self.hires_wave[-1])}
        self.band_interpolate_weights = jnp.atleast_2d(jnp.ones_like(self.hires_wave))
        self.calib_cov = jnp.diag(jnp.array([1,])**2)
        self.wave_sigmas = jnp.array([10,])

        # Instantiate level 3 attributes
        self.used_band_inds = np.array(list(self.band_dict.values()))
        self.used_band_dict = {val: val for val in self.band_dict.values()}
        self.used_zps = self.zps
        self.used_calib_cov = self.calib_cov
        self.used_calib_chcov = jnp.linalg.cholesky(self.used_calib_cov)
        self.used_wave_sigmas = self.wave_sigmas

    def _load_dovekie_cov(self) -> None:
        dovekie_cov = np.load(
                self.__root_dir__ / "bayesn-filters/DOVEKIE_COV_V9.3.npz"
            )
        dovekie_labels = np.loadtxt(
                self.__root_dir__ / "bayesn-filters/DOVEKIE_CHCOV_labels.txt",
                dtype=str
            ).T[1]
        # Skip PS1 aperture photometry values
        PS1_ap_idx = 4
        self.dovekie_cov = dovekie_cov["cov"][PS1_ap_idx:,PS1_ap_idx:]
        self.dovekie_labels = dovekie_labels[PS1_ap_idx:]

    def _load_filter_dict(self) -> dict[str, Any]:
        # Load in-built filter yaml first
        with open(self.__root_dir__ / "bayesn-filters/filters.yaml", "r") as file:
            filter_dict = yaml.load(file)

        # Prepend root locations for in-built filters
        for key, val in filter_dict["standards"].items():
            filter_dict["standards"][key]["path"] = str(
                self.__root_dir__ / "bayesn-filters" / val["path"]
            )

        for key, val in filter_dict["filters"].items():
            filter_dict["filters"][key]["path"] = str(
                self.__root_dir__ / "bayesn-filters" / val["path"]
            )

        # Add custom filters, if specified
        if self.filter_yaml is not None:
            if not self.filter_yaml.exists():
                raise FileNotFoundError(
                    f"Specified filter yaml {self.filter_yaml} does not exist"
                )
            with open(self.filter_yaml, "r") as file:
                custom_filter_dict = yaml.load(file)
            # Add custom standards if specified---------------------
            # May need environment variables e.g. $SNDATA_ROOT or ${HOME}
            env_var_pattern = re.compile(
                r"\${([A-Z0-9_]+)}" # looking for ${caps/numbers/_}
                r"|"                # or
                r"\$([A-Z0-9_]+)"   # $caps/numbers/_
            )
            if "standards" in custom_filter_dict:
                standards_root = Path(custom_filter_dict.get("standards_root", ""))
                for key, val in custom_filter_dict["standards"].items():
                    path = standards_root / val["path"]
                    for _, env_var in env_var_pattern.findall(str(path)):
                        env = os.getenv(env_var)
                        if env is None:
                            raise FileNotFoundError(
                                f"The environment variable {env_var} was not found"
                            )
                        path = Path(str(path).replace(f"${env_var}", env))
                    if not path == path.absolute():
                        # If relative path, prepend yaml location
                        path = self.filter_yaml.absolute().parent / path
                    custom_filter_dict["standards"][key]["path"] = str(path)
                    # Add custom standard and overwrite existing one of same name if present
                    filter_dict["standards"][key] = custom_filter_dict["standards"][key]
            # Add custom filters
            filters_root = Path(custom_filter_dict.get("filters_root", ""))
            for key, val in custom_filter_dict["filters"].items():
                path = filters_root / val["path"]
                for _, env_var in env_var_pattern.findall(str(path)):
                    env = os.getenv(env_var)
                    if env is None:
                        raise FileNotFoundError(
                            f"The environment variable {env_var} was not found"
                        )
                    path = Path(str(path).replace(f"${env_var}", env))
                if not path == path.absolute():
                    # If relative path, prepend yaml location
                    path = self.filter_yaml.absolute().parent / path
                custom_filter_dict["filters"][key]["path"] = str(path)
                # Add custom filter and overwrite existing one of same name if present
                filter_dict["filters"][key] = custom_filter_dict["filters"][key]

        # Load standard spectra if necessary, AB is just calculated analytically so no standard spectrum is required----
        for key, val in filter_dict["standards"].items():
            path = val["path"]
            if ".fits" in path:  # If fits file
                with fits.open(path) as hdu:
                    standard_df = pd.DataFrame.from_records(hdu[1].data)
                standard_lam, standard_f = (
                    standard_df.WAVELENGTH.values,
                    standard_df.FLUX.values,
                )
            else:
                standard_txt = np.loadtxt(path)
                standard_lam, standard_f = standard_txt[:, 0], standard_txt[:, 1]
            filter_dict["standards"][key]["lam"] = standard_lam
            filter_dict["standards"][key]["f_lam"] = standard_f
        return filter_dict

    ########################################
    ### post-init Band Weight Management ###
    ########################################
    def load_bandpass(
        self,
        name: str,
        apply_dovekie_mag_shifts: bool | None = None,
        shift_df: pd.DataFrame | None = None,
        apply_mag_shifts: bool | None = None,
        apply_lam_shifts: bool | None = None,
    ) -> dict[str, Any]:
        """
        Read the filter subdictionary stored in ``self.filter_dict['filters'][name]``,
        apply any relevant shifts (from the dictionary itself, initialization attributes,
        or an optional ``shift_df``), and return the processed dictionary.

        All shift arguments may be None, in which case they default to the attributes
        assigned at initialisation.

        Parameters
        ----------
        name
            Bandpass name in BayeSN convention.
        apply_dovekie_mag_shifts
            Whether to add the Dovekie magnitude shifts.
        shift_df
            Optional DataFrame with columns 'BAND', 'MAG_SHIFT', and 'LAM_SHIFT'
            overriding defaults from ``filters.yaml``.
        apply_mag_shifts
            Whether to add any 'MAG_SHIFT' values from ``shift_df`` to defined magnitudes.
        apply_lam_shifts
            Whether to shift transmission wavelengths by 'LAM_SHIFT' values.

        Returns
        -------
        dict
            Filter dictionary with keys including 'defined_mag', 'lam', 'trans',
            and associated calibration metadata.
        """
        if name not in self.filter_dict["filters"]:
            raise ValueError(
                f"Unrecognised bandpass name: {name}. Valid options can be found by"
                "calling SEDmodel.list_bandpasses()."
            )
        # Prioritise arguments over stored attributes.
        if apply_dovekie_mag_shifts is None:
            apply_dovekie_mag_shifts = self.apply_dovekie_mag_shifts
        if shift_df is None:
            shift_df = self.shift_df
        if apply_lam_shifts is None:
            apply_mag_shifts = self.apply_mag_shifts
        if apply_lam_shifts is None:
            apply_lam_shifts = self.apply_lam_shifts

        ret_dict = {key: val for key, val in self.filter_dict["filters"][name].items()}
        ret_dict["defined_mag"] = ret_dict.pop("magzero", 0)
        lam_shift = ret_dict.pop("lam_shift", 0) * int(apply_lam_shifts)
        lam, trans = np.loadtxt(ret_dict["path"]).T
        idx = np.argsort(lam)
        lam = lam[idx]
        trans = trans[idx]
        # Convert wavelength units if required, model is defined in Angstroms
        units = ret_dict.get("lam_unit", "AA").lower()
        if "nm" in units or "nanomet" in units:
            lam *= 10
        elif "micron" in units or "um" in units:
            lam *= 1e4
        mag_shift = 0
        if shift_df is not None and name in shift_df.BAND.values:
            shift = shift_df[shift_df.BAND == name]
            lam_shift = shift["LAM_SHIFT"].values[0] * int(apply_lam_shifts)
            mag_shift = shift["MAG_SHIFT"].values[0] * int(apply_mag_shifts)
        lam += lam_shift
        ret_dict["lam"] = lam
        ret_dict["trans"] = trans
        ret_dict["defined_mag"] += mag_shift + (
                ret_dict.pop("magupdate", 0) + ret_dict.pop("magcal", 0)
            ) * int(apply_dovekie_mag_shifts)
        return ret_dict

    def _load_band_weights(
        self,
        bands_to_load: list[str],
        apply_dovekie_mag_shifts: None | bool = None,
        shift_df: None | pd.DataFrame = None,
        apply_mag_shifts: None | bool = None,
        apply_lam_shifts: None | bool = None
    ) -> None:
        """
        Sets up the interpolation for the band weights used for photometry as well as
        calculating the zero points for each band. This code is partly based off
        ParSNiP from Boone+21.

        All arguments except bands_to_load may be None, in which case they will be
        taken from the attributes of the same names assigned at initialisation.

        Parameters
        ----------
        apply_dovekie_mag_shifts: bool, default True
            Whether to add the mag_update and mag_cal elements from the
            BASE_DIR/bayesn/bayesn-filters/filters.yaml file to defined magnitudes.
            These values are based on the Dovekie analysis (Popovic et al. 2025).
        shift_df: None | pd.DataFrame, default None
            If not None, then a pd.DataFrame with columns/dtypes
                BAND: strings
                    bandpass names in BayeSN convention
                MAG_SHIFT: Numbers
                    Magnitude value to add to defined magnitude (zero-point).
                LAM_SHIFT: Numbers
                    Angstrom value to add to the wavelengths of the transmission fn.
            These values override the mag_shift and lam_shift values taken from
            bayesn/bayesn-filters/filters.yaml.
        apply_mag_shifts: bool, default False
            Whether to add any "MAG_SHIFT" values from shift_df to defined magnitudes.
            This overrides the effects of apply_dovekie_mag_shifts.
        apply_lam_shifts: bool, default False
            Whether to increase transmission functions by any "lam_shift" values in
            BASE_DIR/bayesn/bayesn-filters/filters.yaml, which can be overridden by
            "LAM_SHIFT" values from shift_df.
        """
        def ab_standard_flam(l):  # Can just use analytic function for AB spectrum
            f = (const.c.to("AA/s").value / 1e23) * (l**-2) * 10 ** (-48.6 / 2.5) * 1e23
            return f

        dlambda = jnp.diff(self.hires_wave)
        dlambda = jnp.r_[dlambda, dlambda[-1]]
        # Load filters------------------------------
        # If not working with lam_shifts, the transmission functions can be
        # pre-processed a bit. Thus, band_weights and band_weights_shift are both
        # calculated and stored in band_interpolate_weights
        band_ind = len(self.band_dict)
        new_bands = set(np.unique(bands_to_load)).difference(set(self.band_dict))
        new_bands_zeros = np.zeros(len(new_bands))
        new_bands_zeros_2d = np.zeros((len(new_bands), len(self.hires_wave)))
        band_interpolate_weights = np.append(self.band_interpolate_weights, new_bands_zeros_2d, axis=0)
        zps = np.append(self.zps, new_bands_zeros)
        # Temporarily calling zp_errs independent, then will add in covariances from
        # Dovekie before storing in SEDmodel.calib_cov
        zp_errs = np.append(np.diag(np.sqrt(self.calib_cov)), new_bands_zeros)
        wave_sigmas = np.append(self.wave_sigmas, new_bands_zeros)
        for i, band in enumerate(new_bands):
            one_band_dict = self.load_bandpass(
                band,
                apply_dovekie_mag_shifts=apply_dovekie_mag_shifts,
                shift_df=shift_df,
                apply_mag_shifts=apply_mag_shifts,
                apply_lam_shifts=apply_lam_shifts
            )
            i += band_ind
            lam = one_band_dict["lam"]
            trans = one_band_dict["trans"]
            magsys = one_band_dict["magsys"]
            defined_mag = one_band_dict["defined_mag"]
            zp_errs[i] = one_band_dict.get("magzero_err", 0.01)  # Rough guess, not principled. Fix this.
            wave_sigmas[i] = one_band_dict.get("wave_sigma", 10)  # Rough guess, not principled. Fix this.
            band_low_lim = lam[np.where(trans > 0.01 * trans.max())[0][0]]
            band_up_lim = lam[np.where(trans > 0.01 * trans.max())[0][-1]]

            # Interpolate the bands to match the sampling of the high-resolution model.
            # This will allow for fast and decently accurate linear interpolation when
            # calculating observer-frame transmissions.
            band_conv_transmission = jnp.interp(
                self.hires_wave, lam, trans, left=0, right=0
            )
            # band_conv_transmission = scipy.interpolate.interp1d(R[:, 0], R[:, 1], kind="cubic",
            #                                                     fill_value=0, bounds_error=False)(band_wave)

            num = self.hires_wave * band_conv_transmission * dlambda
            denom = jnp.sum(num)
            # bandpasses that are 0 over all of hires_wave will cause nans.
            band_interpolate_weights[i] = jnp.nan_to_num(num / denom)

            # Get zero points
            if magsys == "ab":
                zp = ab_standard_flam(lam)
            else:
                standard = self.filter_dict["standards"][magsys]
                zp = interp1d(standard["lam"], standard["f_lam"], kind="cubic")(lam)

            int1 = simpson(lam * zp * trans, x=lam)
            int2 = simpson(lam * trans, x=lam)
            self.band_dict[band] = i
            self.band_lim_dict[band] = [band_low_lim, band_up_lim]
            zp = 2.5 * np.log10(int1 / int2) + defined_mag
            zps[i] = zp
            self.zp_dict[band] = zp

        self.band_interpolate_weights = jnp.array(band_interpolate_weights)
        self.zps = jnp.array(zps)
        self.wave_sigmas = jnp.array(wave_sigmas)
        calib_cov = np.diag(zp_errs**2)
        dov_inds = []
        for dov_idx, bp in enumerate(self.dovekie_labels):
            if bp not in self.band_dict:
                continue
            dov_inds.append((dov_idx, self.band_dict[bp]))
        for dov_ind_1, ind_1 in dov_inds:
            for dov_ind_2, ind_2 in dov_inds:
                calib_cov[ind_1, ind_2] = self.dovekie_cov[dov_ind_1, dov_ind_2]
        self.calib_cov = calib_cov

    def _set_used_bands(self, bands: list[str] | None = None) -> None:
        """
        Sets the attributes used in various methods to include only bandpasses in the
        data and the NULL_BAND which will never produce NaNs when band_indices masked
        with 0s map to it.

        The affected attributes and their roles are
        used_band_inds: np.ndarray
            Indices of bandpasses as they appear in bayesn/bayesn-filters/filters.yaml.
        used_band_dict: dict
            Key-value pairs mapping used_band_inds (keys) to their integer order (value).
        used_zps: jax.Array
            Zero-points to convert between fluxes and magnitudes.
            The order corresponds to the used_band_dict values.
        used_calib_cov: np.ndarray
            A subset of the whole covariance matrix (SEDmodel.calib_cov) that only
            includes elements describing the covariance between bandpasses in the data.
        used_calib_chcov: jax.Array
            Cholesky decomposition of used_calib_cov
        used_wave_sigmas: jax.Array
            Uncertainties for wavelength shifts (Angstroms) for bandpasses in the data.

        The methods that use these attributes are:
        _calculate_band_weights,
        get_flux_batch (and therefore get_mag_batch),
        _model,
        _process_dataset_version_photometry,
        _process_dataset_data_table,
        _add_data_from_file,
        simulate_lightcurve,
        sample_lambda_shift,
        sample_mag_shift,

        The SEDmodel._load_band_weights call uses the default arguments of None for the
        shift booleans and shift_df, leading to the use of the attributes loaded during
        initialisation.
        """
        if bands is None:  # Assumes all bands in the level 2 cache are to be used.
            bands = list(self.band_dict.keys())
        else:
            _, idx = np.unique(bands, return_index=True)
            bands = np.array(bands)[np.sort(idx)]
            self._load_band_weights(bands)
        if "NULL_BAND" not in bands:
            bands = ["NULL_BAND",] + list(bands)
        self.used_band_inds = np.array([self.band_dict[b] for b in bands])
        self.used_band_dict = dict(zip(self.used_band_inds, range(len(bands))))
        self.used_zps = self.zps[self.used_band_inds]
        self.used_calib_cov = self.calib_cov[jnp.ix_(self.used_band_inds, self.used_band_inds)]
        self.used_calib_chcov = jnp.linalg.cholesky(self.used_calib_cov)
        self.used_wave_sigmas = self.wave_sigmas[self.used_band_inds]
        used_dov_inds = set([self.band_dict[band] for band in self.dovekie_labels if band in self.band_dict]).intersection(set(self.used_band_inds))
        non_null_inds = set(ind for ind in self.used_band_inds if ind != 0)
        if len(used_dov_inds) and len(used_dov_inds) != len(non_null_inds):
            warn(UserWarning("""
                WARNING: The used bands include some that were analysed in Dovekie
                and some that were not. The mag_shift covariance matrix is defined
                for those in Dovekie, but may not be appropriate for mixed samples.
                The mag_shift covariance matrix for bandpasses not in Dovekie
                are purely diagonal and based on magzero_err terms taken from
                bayesn/bayesn-filters/filters.yaml if available, or 0.01 mag as an
                arbitrary guesstimate. You may want to avoid the vary_mag_shift flag.
                """)
            )


    def _calculate_band_weights(self, redshifts: ArrayLike, ebv: ArrayLike, lam_shifts: Number | ArrayLike = 0) -> Array:
        """
        Calculates the observer-frame band weights for each of N_sn SNe.
        The wavelength vector SEDmodel.hires_wave is spaced log-uniformly, meaning
        incrementing the index is equivalent to mulitplying the wavelength by 10**dx
        where dx is the log-spacing (SEDmodel.hires_spacing).
        Given a lambda_min at index 0 the index of an observer-frame wavelength lambda
        is log_{10}(lambda/lambda_min)/dx, which is probably not an integer.

        Given any vector defined at the wavelengths of SEDmodel.hires_wave, the vector
        can be linearly interpolated with the indices bordering the float index.
        The vectors of interest are:
            SEDmodel.band_interpolate_weights: transmission functions for loaded bands
            SEDmodel.mw_ext: A(x)/A_V for RV=SEDmodel.RV_MW

        Parameters
        ----------
        redshifts: ArrayLike shape (N_sn,)
            Array of redshifts for each SN
        ebv: ArrayLike shape (N_sn,)
            Array of Milky Way E(B-V) values for each SN
        lam_shifts: ArrayLike shape (N_bandpasses,)
            Value of the wavelength lambda shift to apply to each bandpass.
            Positive values correspond to a redshift.

        Returns
        -------
        weights: ArrayLike shape (N_sn, N_wl, N_bandpasses)
            Array containing observer-frame band weights
        """
        # supporting lam_shifts = const -> lam_shifts[i] = const for all N_bands
        N_bands = self.used_band_inds.shape[0]
        lam_shifts = jnp.empty(N_bands).at[:].set(jnp.atleast_1d(lam_shifts))

        # Calculating float indices of SEDmodel.hires_wave for obs_frame_wave
        # Integer indices and remainder allow for interpolation
        obs_frame_wave = self.model_wave[None,:,None] * (1+redshifts)[:,None,None]
        # Positive lam_shifts correspond to redshifts. Since the weights are
        # interpolated, obs_frame_wave must be decreased by lam_shift.
        obs_frame_wave -= lam_shifts[None, None, :]
        locs = jnp.log10(obs_frame_wave/self.min_hsiao_wave)/self.hires_spacing
        int_locs = locs.astype(jnp.int32)
        remainders = locs - int_locs

        # Linear piecewise interpolation.
        # Not strictly correct, but if the resolution of SEDmodel.hires_wave is high
        # enough, the log-uniform curve will look locally linear between neighbouring
        # indices.
        interp_weights = self.band_interpolate_weights[self.used_band_inds]
        band_idx = jnp.arange(N_bands)[None,None,:]
        start = interp_weights[band_idx, int_locs]
        end = interp_weights[band_idx, int_locs + 1]
        weights = remainders * end + (1-remainders)*start

        # band_interpolate_weights is normalised to sum to 1, but interpolating at a
        # different wavelength grid will require another round of normalisation.
        norm = jnp.sum(weights, axis=1)
        safe_norm = jnp.where(norm > 0, norm, 1.)
        weights = jnp.where((norm > 0)[:, None, :], weights/safe_norm[:, None, :], 0.)

        # MW extinction
        mw_avax = remainders * self.mw_ext[int_locs + 1] + (1-remainders) * self.mw_ext[int_locs]
        mw_weights = jnp.power(10, -0.4*mw_avax*(self.RV_MW*ebv)[:,None,None])
        weights *= mw_weights

        # Photons emitted at rate r in the rest-frame will be received at rate r/(1+z)
        # in the observer-frame, modelled as a reduction in transmission.
        return weights / (1+redshifts)[:,None,None]

    #####################
    ### Configuration ###
    #####################
    def parse_args(self, args: dict, verbose: bool = True) -> dict:
        """ Given an incomplete args dict, return a complete args dict by interpreting
        the mode, detecting RV and shared_RV, and populating missing args with default
        values, and validating the total configuration.

        Parameters
        ----------
        args
            Arguments from input yaml file before command line overrides,
            defines model wavelength range and data set to load.

        Returns
        -------
        args updated with defaults and validated

        Notes
        -----
        THIS IS NOT IDEMPOTENT.
        If jobsplit is None, snana will become False and jobsplit will become [1, 1]
        If run again, snana will become True.
        """
        args.pop("CONFIG", None)
        args.pop("config", None)
        args = self._parse_mode(args, verbose=verbose)
        detected_rv_type, RV = self._get_rv_type(args, verbose=verbose)
        args["shared_RV"] = detected_rv_type
        args["RV"] = RV
        if verbose:
            self._print_default_assignments(args)

        for param, default in default_kwargs.items():
            args[param] = args.pop(param, default)

        # Handle default_kwargs that are "default" or based on model values.
        for key, val in args.items():
            if val != "default":
                continue
            if key in ("l_knots", "tau_knots"):
                args[key] = getattr(self, key).tolist()
            elif key == "outputdir":
                args[key] = "."
            elif key == "RV":
                if self.RV is not None:
                    args["RV"] = float(self.RV)
                elif self.mu_R is not None:
                    args["RV"] == "normal"
            elif key == "mu_R":
                args[key] = float(self.mu_R or self.RV)
            elif key == "sigma_R":
                args[key] = float(self.sigma_R or 0.5)
            elif hasattr(self, key):
                args[key] = getattr(self, key)
            elif key in default_kwargs:
                args[key] = default_kwargs[key]
            else:
                raise NotImplementedError(f"There is no default value for key {key}.")

        # The VI fitting method uses a modified exponential for AV.
        if args.get("fit_method") == "vi":
            args["AV_dist"] = zltn.My_Exponential

        args["outputdir"] = Path(args["outputdir"]).absolute()
        args["photoz"] = args.get("photoz", False)
        args["num_zltn_iter"] = args.get("num_zltn_iter", 4000 if args["photoz"] else 1500)
        if isinstance(args["keep_list"], str | Path) and Path(args["keep_list"]).exists():
            keep_list = pd.read_csv(args["keep_list"], comment="#", sep=r"\s+")
            if "CID" in keep_list.columns:
                keep_list = keep_list["CID"].values
            elif "SNID" in keep_list.columns:
                keep_list = keep_list["SNID"].values
            else:
                keep_list = pd.read_csv(args["keep_list"], header=None)[0].values
            args["keep_list"] = keep_list.astype(str)
        elif isinstance(args["keep_list"], str):
            args["keep_list"] = [args["keep_list"]]
        elif hasattr(args["keep_list"], "__iter__"):
            args["keep_list"] = list(args["keep_list"])
        else:
            args["keep_list"] = None

        for key in {"mode", "fit_method", "laplace_method", "lm_solver"}:
            args[key] = args[key].lower()
        pdp = args.get("private_data_path", [])
        args["private_data_path"] = [pdp] if isinstance(pdp, str) else pdp
        if args["jobsplit"] is not None:
            args["snana"] = True
        else:
            args["jobsplit"] = [1, 1]
            args["snana"] = False
        args["jobid"], args["njobtot"] = args["jobsplit"]

        if not (args["mode"].startswith("fit") and args["snana"]):
            try:
                if not args["outputdir"].exists():
                    args["outputdir"].mkdir()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Requested output directory does not exist and could not be created"
                )
        self._check_args_valid(args, verbose=verbose)

        if args["train_new_model"]:
            self.l_knots = device_put(np.array(args["l_knots"], dtype=float))
            KD_l = invKD(self.l_knots)
            self.J_l_T = device_put(
                spline_coeffs(self.model_wave, self.l_knots, KD_l)
            )
            self.tau_knots = device_put(np.array(args["tau_knots"], dtype=float))
            self.KD_t = device_put(invKD(self.tau_knots))
            self.N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
            self.N_knots_sig_l = self.l_knots.shape[0] - 2
            self.N_knots_sig = (self.N_knots_sig_l) * self.tau_knots.shape[0]
        return args

    def _parse_mode(self, args: dict, verbose: bool = True) -> dict:
        """
        This method interprets the mode config parameter.
        Mode was a macro for several configurations of kwargs.
        If something is in the yaml but conflicts with the mode, raise an error.
        Otherwise, populate args with the model's appropriate kwargs as if they
        were in the yaml.

        The supported modes are
            "fitting"
            "training_popRv"
            "training_globalRv"
            "training_uniformRv"
            "dust"
            "dust_redshift"
            "dust_split_mag"
            "dust_split_sed"

        Parameters
        ----------
        args
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        verbose:

        Returns
        -------
        args: dict
            Original args updated with mode-appropriate values.
        """
        mode = args.get("mode", "custom").lower()

        expected_values = {
            "infer_dust_properties": [False, True, True],
            "train_new_model": [False, True, False],
            "fix_tmax": [False, True, True],
            "vary_redshift": [False, False, False],
            "muhat_err": [5, None, None],
            "data_type": ["flux", "mag", "flux"],
            "mu_R_min": [1.2, 1, 1.2],
            "mu_R_max": [6, 5, 6],
        }
        for mode_idx, substr in enumerate(("fit", "train", "dust")):
            if not mode.startswith(substr):
                continue
            for key, val in expected_values.items():
                # hard-coding dust_redshift since it contains the dust substring.
                if substr == "dust" and "redshift" in mode and key == "vary_redshift":
                    val[mode_idx] = True
                if key not in args:
                    args[key] = val[mode_idx]
                elif args[key] != val[mode_idx]:
                    raise ValueError(
                        f"mode is {mode} but {key} is {args[key]}. "
                        f"This is not consistent with typical {mode} behavior. "
                        "Consider changing the mode or setting it to 'custom'. "
                    )

        for split_variant in ("split_mag", "split_sed"):
            if split_variant in mode:
                args["split_variant"] = split_variant

        if mode.startswith("dust"):
            if args.get("shared_RV", False):
                raise ValueError(
                    f"mode is {mode} but shared_RV is {args['shared_RV']}. "
                    f"This is not consistent with typical {mode} behavior. "
                    "Consider changing the mode or setting it to 'custom'."
                )
            args["shared_RV"] = False

        # RV information was previously provided through the mode parameter.
        # Check for backwards compatibility.
        # e.g. fit_poprv should give shared_RV = False
        #      train_global_rv should give shared_RV = True
        supported_RV_dict = {True: "global", False: "pop"}
        if "shared_RV" in args and supported_RV_dict[not args["shared_RV"]] in mode:
            raise ValueError(
                f"shared_RV was provided as {args['shared_RV']}. However, the mode "
                f"{mode} seems to indicate that a different shared_RV should be used. "
                "Please remove the inconsistency."
            )
        for shared_RV, mode_name in supported_RV_dict.items():
            if mode_name not in mode:
                continue
            args["shared_RV"] = shared_RV
            if verbose:
                warn(UserWarning(
                    f"shared_RV was inferred as {args['shared_RV']} from the mode "
                    f"{args['mode']}. Indicating shared_RV with the mode is supported "
                    "for backwards compatibility, but in the future please consider "
                    "including the shared_RV parameter in the input yaml file."
                ))
        if not any([val in mode for val in supported_RV_dict.values()]):
            if verbose and "rv" in mode:
                warn(UserWarning(
                    f"The indicated mode {mode} contains the string 'rv', but not the "
                    "strings 'global' or 'pop', which correspond to shared_RV True and "
                    "False respectively. Indicating shared_RV and RV with the mode arg "
                    "will be deprecated at some point, but for now you may wish to fix "
                    "the mode argument to include 'global' or 'pop' (case insensitive)."
                ))
            return args
        return args

    def _get_rv_type(self, args: dict, verbose: bool = True) -> tuple[bool, Number | str]:
        """ The shared_RV attribute indicats whether all SNe share a single RV, and the
        RV attribute can either specify that value, or be a string indicating what
        distribution the RV(s) should be sampled from. These kwargs can be passed in
        the input yaml or inferred from non-default arguments using this method.
        This method attempts to infer shared_RV and RV, raising an error if there are
        ambiguous indicators.

        Parameters
        ----------
        args
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
            The args passed to this method should include any modifications from the
            "_parse_mode" method, but should not include default values from the
            "default_kwargs" dict because the provided arguments may indicate what RV
            is intended.
        verbose:

        Returns
        -------
        detected_shared_RV:
            True for a global RV shared by all SNe, False for SN-specific RVs. In both
            cases, the values depend on detected_RV.
        detected_RV:
            Either "normal", "uniform", or a number for a Normal distribution with a
            lower limit of 1.2, a uniform distribution, or a fixed number (Dirac Delta
            distribution).
        """
        detected_shared_RV = args.get("shared_RV")
        detected_RV = args.get("RV", "default")
        model_shared_RV = self.mu_R is None
        if detected_shared_RV is None:
            detected_shared_RV = True if detected_RV != "default" else model_shared_RV
        elif detected_shared_RV and (
            (detected_RV == "default" and model_shared_RV)
            or isinstance(detected_RV, Number)
        ):
            detected_shared_RV = True
            if verbose:
                msg = f"detected_shared_RV is False, but detected_RV is {detected_RV}, "
                if detected_RV == "default":
                    msg += f"which will be treated as {self.RV} and "
                else:
                    msg += "which will be "
                msg += "used for all SNe as though shared_RV=True."
                warn(UserWarning(msg))

        if detected_RV == "default":
            msg = f"RV is 'default', which will be treated as "
            if model_shared_RV:
                detected_RV = float(self.RV)
                msg += f"{self.RV} to match the loaded model's 'RV' attribute. "
            else:
                detected_RV = "normal"
                msg += f"'normal' since the loaded model has a 'mu_R' attribute. "
            msg += "If this is not the desired behavior, provide more specific args."
            if verbose:
                warn(UserWarning(msg))
        elif isinstance(detected_RV, str) and detected_RV not in ("uniform", "normal"):
            raise ValueError(
                f"RV was specified as {detected_RV}, which is not a supported option. "
                "RV should be a number or one of the strings 'default', 'uniform', or "
                "'normal'."
            )

        return detected_shared_RV, detected_RV

    def _print_default_assignments(self, args: dict) -> None:
        """Print out when relevant parameters are being assigned their default values.
        The top-level dicts' keys are kwargs that, depending on their values, will
        depend on the values of other kwargs. If the top-level kwarg is specified as
        one of the keys in the sub-dict, the corresponding values in that sub-dict
        list the kwargs that will be required.
        """
        relevant_kwargs_dict = {
            # As an example: if the RV kwarg is specified as "normal", the relevant
            # kwargs are "mu_R" and "sigma_R".
            "RV": {
                "normal": ("mu_R", "sigma_R",),
                "uniform": ("uniform_RV_min", "uniform_RV_max")
            },
            "split_variant": {
                "split_mag": ("M_split",),
                "split_sed": ("M_split",),
            },
            "fit_method": {
                "vi": (
                    "laplace_method", "zltn_lr", "zltn_lr_final",
                    "zltn_particles", "num_zltn_iter", "batch_size"
                )
            },
            "laplace_method": {
                "lm": ("lm_maxiter", "lm_lam_init", "lm_use_linesearch", "lm_solver", "stage2_tmax_prior_std"),
            },
        }
        if args.get("infer_dust_properties"):
            relevant_kwargs_dict.update({
                "RV": {
                    "normal": ("mu_R_min", "mu_R_max", "sigma_sigma_R"),
                },
                "vary_redshift": {
                    True: ("tau_z_min", "tau_z_max"),
                },
            })

        for kwarg_name, subdict in relevant_kwargs_dict.items():
            if args.get(kwarg_name) not in subdict:
                continue
            sub_kwargs = subdict[args[kwarg_name]]
            missing_kwargs = set(sub_kwargs).difference(args)
            missing_defaults = [f"{kw}={default_kwargs[kw]}" for kw in missing_kwargs]
            if not len(missing_kwargs):
                continue
            s = "s" if len(missing_kwargs) > 1 else ""
            warn(UserWarning(
                f"{kwarg_name}={args[kwarg_name]}, and so the kwargs {missing_kwargs} "
                f"are relevant but not provided in the input args. The default{s} "
                f"{', '.join(missing_defaults)} will be assumed."
            ))
    def _check_args_valid(self, args: dict, verbose=True) -> None:
        """
        Validates the input args by looking for parameters that will cause problems
        downstream, e.g. unsupported data_type/fit_method.

        Parameters
        ----------
        args
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        """
        def check_scalar(val_name: str) -> None:
            try:
                float(args[val_name])
            except (ValueError, TypeError):
                raise TypeError(
                    f"{val_name} must be a float-like value. Instead got {args[val_name]}."
                )

        # Warnings for weird configurations that do not break things.
        if not args["shared_RV"] and isinstance(args["RV"], Number) and verbose:
            warn(UserWarning(
                f"shared_RV is False but RV is {args['RV']}. The run will proceed but "
                "you should know that all SNe will share that value for RV."
            ))

        # Errors for configuraitons that will break things.
        if args["RV"] not in ("normal", "uniform") and not isinstance(args["RV"], Number):
            raise ValueError(
                f"RV is {args['RV']}, which is not a supported option. Please set RV "
                "to 'normal', 'uniform', or a number."
            )
        if args["RV"] == "normal" and args["infer_dust_properties"]:
            for name in ("mu_R_min", "mu_R_max", "sigma_sigma_R"):
                check_scalar(name)
            if float(args["mu_R_min"]) >= float(args["mu_R_max"]):
                raise ValueError("mu_R_min must be less than mu_R_max.")
            if float(args["sigma_sigma_R"]) <= 0:
                raise ValueError("sigma_sigma_R must be positive.")
            if verbose and args["shared_RV"]:
                warn(UserWarning(
                    "RV is 'normal' and shared_RV and infer_dust_properties are both "
                    "True. mu_R and sigma_R will be poorly constrained with one shared "
                    "draw of RV."
                ))
        elif args["RV"] == "normal" and not args["infer_dust_properties"]:
            for name in ("mu_R", "sigma_R"):
                check_scalar(name)
            if float(args["sigma_R"]) <= 0:
                raise ValueError("sigma_R must be positive.")
        elif args["RV"] == "uniform":
            for name in ("uniform_RV_min", "uniform_RV_max"):
                check_scalar(name)
            if float(args["uniform_RV_min"]) >= float(args["uniform_RV_max"]):
                raise ValueError("uniform_RV_min must be less than uniform_RV_max. ")
            if args["infer_dust_properties"] and not args["shared_RV"]:
                raise NotImplementedError(
                    "Inferring the hyper-parameters of uniform RV distributions is "
                    "not supported. Please change one of the following arguments: "
                    "'RV', 'shared_RV', 'infer_dust_properties'."
                )

        choice_dict = {
            "data_type": {"flux", "mag"},
            "fit_method": {"vi", "mcmc"},
            "laplace_method": {"svi", "lm"},
            "lm_solver": {"gn", "hvp_cg"},
            "split_variant": {None, "split_mag", "split_sed"},
        }
        for key, options in choice_dict.items():
            if args[key] not in options:
                raise ValueError(f"{key} must be in {options}, got {args[key]!r}")

        if "version_photometry" not in args and "data_table" not in args:
            raise ValueError(
                "Please pass either data_dir (for a directory containing all SNANA "
                "files such as a simulation output) or a combination of data_table "
                "and data_root."
            )
        if "data_table" in args and "data_root" not in args:
            raise ValueError(
                "If using data_table, please also pass data_root (which defines the "
                "location that the paths in data_table are defined with respect to)."
            )

    ######################
    ### Data Ingestion ###
    ######################
    def process_dataset(self, args: dict) -> None:
        """
        Processes a data set to be used by the numpyro model.

        This will read in SNANA-format files, either in text or FITS format.

        This will read through all light curves and work out the maximum number of data
        points for a single object - all others will then be padded to match this size.
        This is required because to benefit from the GPU, we need to have a fixed array
        structure allowing us to calculate flux integrals from parameter values across
        the whole sample in a single tensor operation. A mask is applied in the model
        to ensure that these padded values do not contribute to the likelihood.

        This method uses various arguments in the args dict to ingest data, apply cuts
        and various transformations, and sets attributes for data products that will be
        used for other purposes, including modelling with the run method.

        The args must include either "version_photometry", which points to a data
        directory containing an args["version_photometry"].LIST file, or the
        "data_root" and "data_table" arguments which together point to a table file that
        contains paths (relative to data_root) for light curve data files.

        Parameters
        ----------
        args
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        """
        survey_dict = {}
        if args.get("version_photometry"):
            data_dir, survey_dict = find_data_dir_in_SNANA(
                private_data_path=args["private_data_path"]
            ) if args["snana"] else (args["version_photometry"], {})
            list_file = Path(data_dir, Path(data_dir).name+".LIST")
            ds = SNDataset.from_snana_list(
                list_file,
                data_root=data_dir,
                keep_list=args["keep_list"],
                fluxcal_zpt=self.ZPT,
                peakmjd_key=args["peakmjd_key"],
                jobid=args["jobid"],
                njobtot=args["njobtot"],
                sim_prescale=args["sim_prescale"],
            )
        else:
            table_path = Path(args["data_root"], args["data_table"])
            data_root = args["data_root"]
            ds = SNDataset.from_table_file(
                table_path,
                data_root=data_root,
                fluxcal_zpt=self.ZPT,
                peakmjd_key=args["peakmjd_key"],
                jobid=args["jobid"],
                njobtot=args["njobtot"],
                sim_prescale=args["sim_prescale"],
            )
        if survey_dict == {}:
            survey_dict = {
                s: i for i,s in enumerate(np.unique(ds.survey)) if s is not None
            }
        surveys = [s for s in ds.survey if s is not None]
        if len(surveys) == 1:
            self.survey = surveys[0]
        elif len(surveys) and all([s == surveys[0] for s in surveys]):
            self.survey = surveys[0]
        else:
            self.survey = "NULL"
        self.survey_id = survey_dict.get(self.survey, 0)

        ds.apply_filter_map(args["map"])
        ds.drop_bands(args["drop_bands"])
        self._set_used_bands(ds.unique_bands)
        ds.drop_by_band_lims(self.band_lim_dict, wave_min=self.model_wave[0], wave_max=self.model_wave[-1])
        ds.cut_by_phot_numeric("phase", "<=", self.tau_knots[0])
        ds.cut_by_phot_numeric("phase", ">=", self.tau_knots[-1])
        ds.cut_by_meta_numeric("z_helio", ">", args["zlim"])
        ds.apply_error_floor(args["error_floor"])
        if args["keep_list"] is not None:
            ds.keep_according_to_list(args["keep_list"])

        # Making all_table before lc_cuts to store reasons why things were cut
        self.fitres_table, self.all_table = ds.make_fitres_table("version_photometry" in args, keep_dict=args["lc_cuts"])
        for key, val in args["lc_cuts"].items():
            ds.cut_by_meta_numeric(key, "<", val[0])
            ds.cut_by_meta_numeric(key, ">", val[1])

        self.dataset = ds
        self._set_used_bands(ds.unique_bands)
        self.lcplot_data = ds.make_lcplot_data(args["num_lcplot"])
        self.data = device_put(ds.make_bayesn_data(
            data_type=args["data_type"],
            band_dict=None,  # uses 1-based order in ds.unique_bands
            N_obs_max=args.get("N_obs_max"),
            cosmo=self.cosmo,
            negative_flux_mag_val=-99,
        ))

        t = self.data.mjd
        self.J_t = self.get_J_t(t)
        self.hsiao_interp = self.get_hsiao_interp(t)
        self.band_weights = self._calculate_band_weights(
            redshifts=self.data.z_hel,
            ebv=self.data.MWEBV,
            lam_shifts=np.zeros(len(self.dataset.unique_bands)+1),
        )

    def get_sn_slice_of_data(self, slice_: slice) -> ObsData:
        """
        Slices the SN axis of the attached ObsData to make a new subset ObsData.

        Parameters
        ----------
        slice_
            A 1D slice applied to the last axis of each array in self.data.

        Returns
        -------
        Sliced ObsData
        """
        if not isinstance(self.data, ObsData):
            raise ValueError("The data attribute has not been set yet.")
        return ObsData(*[arr[..., slice_] for arr in self.data])
    ###############################
    ### Astronomical Quantities ###
    ###############################
    @property
    def get_axav(self) -> Callable:
        """
        Public get_axav method of the loaded extinction relation.

        Returns
        -------
        Callable
            The ``DustExtRel.get_axav`` method of the loaded extinction relation.
        """
        return self.ext_rel.get_axav

    def _get_spectra(
        self,
        theta: ArrayLike,
        AV: ArrayLike,
        W0: ArrayLike,
        W1: ArrayLike,
        eps: ArrayLike,
        RV: ArrayLike,
        J_t: ArrayLike,
        hsiao_interp: ArrayLike,
        **kwargs: Any,
    ) -> Array:
        """
        Low-level method to calculate rest-frame spectra for given parameter values.

        Parameters
        ----------
        theta : (N_sn,)
            Set of theta values for each SN.
        AV : (N_sn,)
            Set of host extinction values for each SN.
        W0 : (N_l_knots, N_tau_knots) or (N_sn, N_l_knots, N_tau_knots)
            If 2D, global W0 matrix for all SNe.
            If 3D, SN-specific W0 matrix.
        W1 : (N_l_knots, N_tau_knots) or (N_sn, N_l_knots, N_tau_knots)
            If 2D, global W1 matrix for all SNe.
            If 3D, SN-specific W1 matrix.
        eps : (N_sn, N_l_knots, N_tau_knots)
            Set of epsilon values for each SN, describing residual colour variation.
        RV : (N_sn,)
            Set of R(V) values for each SN's host-galaxy extinction.
        J_t : (N_sn, N_tau_knots + 1, N_max_epochs)
            Matrix for cubic spline interpolation in time axis for each SN.
        hsiao_interp : (3, N_max_epochs, N_sn)
            Array containing Hsiao template spectra for each t value, comprising model
            for previous day, next day and t % 1 to allow for linear interpolation.
        **kwargs
            Additional arguments not used, but allowed to support passing kwarg dicts.

        Returns
        -------
        (N_sn, N_wl, N_max_epochs)
            Matrix containing model spectra for all SNe at all time-steps.
        """
        num_batch = theta.shape[0]
        W = W0 + theta[..., None, None] * W1 + eps
        padded_W = jnp.ones((W.shape[0], W.shape[1]+1, W.shape[2]+1))
        padded_W = padded_W.at[:, 1:, 1:].set(W)
        W_grid = jnp.matmul(self.J_l_T, jnp.matmul(padded_W, J_t))

        if not self.photoz:
            low_hsiao = self.hsiao_flux[:, hsiao_interp[0, ...].astype(int)]
            up_hsiao = self.hsiao_flux[:, hsiao_interp[1, ...].astype(int)]
            H_grid = (
                (1 - hsiao_interp[2, :]) * low_hsiao + hsiao_interp[2, :] * up_hsiao
            ).transpose(2, 0, 1)
        else:
            # pre-explosion clamps to the ~0 explosion-epoch (-20d) row
            # power-law tail beyond the late edge
            n_h = self.hsiao_flux.shape[1]
            low = jnp.clip(hsiao_interp[0, ...].astype(int), 0, n_h -1)
            up = jnp.clip(hsiao_interp[1, ...].astype(int), 0, n_h -1)
            H_in = (
                (1 - hsiao_interp[2, ...]) * self.hsiao_flux[:,low]
                + hsiao_interp[2, ...]*self.hsiao_flux[:,up]
            )
            t = hsiao_interp[0, ...] + self.hsiao_t[0] + hsiao_interp[2, ...]
            f_late = self.hsiao_flux[:, -1][:, None, None]
            slope_late = (self.hsiao_flux[:,-1] - self.hsiao_flux[:,-2])[:, None, None]
            dt_late = jnp.clip(t - self.hsiao_t[-1], 0., None)[None, ...]
            H_late = f_late * jnp.exp(
                -dt_late * jnp.abs(slope_late) / jnp.where(f_late > 0, f_late, 1.)
            )
            H_grid = jnp.where(
                t[None, ...] > self.hsiao_t[-1], H_late, H_in
            ).transpose(2, 0, 1)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)
        A = AV[..., None] * self.ext_rel._get_axav(RV=RV)
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[..., None]

        return model_spectra

    def get_spectra(
        self,
        theta: ArrayLike,
        AV: ArrayLike,
        W0: ArrayLike | None = None,
        W1: ArrayLike | None = None,
        eps: ArrayLike | None = None,
        t: ArrayLike | None = None,
        RV: ArrayLike | None = None,
        J_t: ArrayLike | None = None,
        hsiao_interp: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Array:
        """
        Wrapper for the low-level ``_get_spectra`` method.

        Flexible with argument shapes:

        * If arguments are scalars: returns one spectrum of shape ``(N_wl,)``.
        * If arguments are 1D arrays of length ``N_sn``: returns an array of shape ``(N_sn, N_wl)``.
        * If ``t`` is a 2D array of shape ``(N_sn, N_max_epochs)`` or ``J_t`` is defined: returns an array of shape ``(N_sn, N_wl, N_max_epochs)``.

        Parameters
        ----------
        theta : scalar or (N_sn,)
            The theta value to be used for all spectra, or an array of N_sn values.
        AV : scalar or (N_sn,)
            The host extinction value to be used for all spectra, or an array of N_sn values.
        W0 : None or (N_l, N_tau) or (N_sn, N_l, N_tau)
            If None, use ``SEDmodel.W0`` for all spectra. If 2D, the global W0 matrix.
            If 3D, SN-specific W0 matrices.
        W1 : None or (N_l, N_tau) or (N_sn, N_l, N_tau)
            If None, use ``SEDmodel.W1`` for all spectra. If 2D, the global W1 matrix.
            If 3D, SN-specific W1 matrices.
        eps : (N_l, N_tau) or (N_sn, N_l, N_tau)
            Residual perturbations to the SED surface. If 2D, shared across all SNe.
            If 3D, SN-specific perturbations. Should not actually be None, but follows
            arguments with default values to match signature order used elsewhere.
        t : scalar or (N_sn,) or (N_sn, N_max_epochs), optional
            If ``J_t`` and ``hsiao_interp`` are not provided, t is used to generate the
            phase interpolation matrix and Hsiao interpolation coordinates.
            Broadcasted to (N_sn, N_max_epochs).
        RV : None or scalar or (N_sn,)
            Host extinction R(V) value. If None, use ``SEDmodel.RV`` or
            ``SEDmodel.mu_R`` for all spectra, otherwise broadcast to (N_sn,).
        J_t : (N_sn, N_tau + 1, N_max_epochs), optional
            Phase interpolation matrix, required if ``t`` is None.
        hsiao_interp : (3, N_max_epochs, N_sn), optional
            Linear interpolation coordinates for the Hsiao template, required if ``t``
            is None.
        **kwargs
            Additional arguments not used, but allowed to support passing kwarg dicts.

        Returns
        -------
        (N_wl,) or (N_sn, N_wl) or (N_sn, N_wl, N_max_epochs)
            Model spectra corresponding to the dimensionality of input arguments.
        """

        if eps is None:
            raise TypeError("Missing required positional argument: 'eps'")
        def set_vals_to_1d(params, max_length):
            for key, val in params.items():
                if len(val.shape) == 0 or (len(val.shape) == 1 and len(val) == 1):
                    params[key] = np.full(max_length, val)
            return params

        if W0 is None:
            W0 = self.W0
        if W1 is None:
            W1 = self.W1
        params = {"theta": theta, "AV": AV, "RV": RV}
        multi_dim_params = {"W0": W0, "W1": W1, "eps": eps}
        ndim = []
        for key, val in params.items():
            params[key] = val = np.squeeze(val)
            ndim.append(len(val.shape))
        for key, val in multi_dim_params.items():
            multi_dim_params[key] = val = np.squeeze(val)
            ndim.append(len(val.shape) - 2)
        max_ndim = max(ndim)
        if max_ndim > 1:
            raise ValueError("Arguments have more dimensions than expected.")
        elif max_ndim == 1:
            lengths = np.array(
                [jnp.atleast_1d(val).shape[0] for val in params.values()]
                + [mdp.shape[0] for mdp in multi_dim_params.values() if len(mdp.shape) == 3]
            )
            max_length = int(max(lengths))
            if not all((lengths == 1) + (lengths == max_length)):
                raise ValueError(
                    "Arguments include multiple 1D arrays with inconsistent lengths"
                )
            params = set_vals_to_1d(params, max_length)
        else:
            params = set_vals_to_1d(params, 1)

        if J_t is not None and hsiao_interp is not None:
            N_sn, _, N_max_epochs = J_t.shape
            if hsiao_interp.shape != (3, N_max_epochs, N_sn):
                raise ValueError(
                    "J_t and hsiao_interp have incompatible shapes. They should be "
                    "(N_sn, N_tau_knots, N_max_epochs) and (3, N_max_epochs, N_sn) "
                    f"respectively, but the input arguments have shapes {J_t.shape} "
                    f"and {hsiao_interp.shape}."
                )
            if max_ndim == 1 and max_length != N_sn:
                raise ValueError(
                    "Input arguments include multiple 1D arrays with lengths "
                    f"{max_length} which is inconsistent with J_t and hsiao_interp "
                    f"which expect arrays of length {N_sn}."
                )
            if max_ndim == 0:
                max_length = N_sn
                params = set_vals_to_1d(params, max_length)
            max_ndim = 2
        elif t is not None:
            if len(t.shape) > 2:
                raise ValueError("Input t has more dimensions than expected.")
            t = np.squeeze(t)
            t_dim = len(t.shape)
            t = jnp.atleast_2d(t)
            if max_ndim == 1 and t.shape[1] > 1 and max_length != t.shape[1]:
                raise ValueError("Input arguments include multiple 1D arrays with inconsistent lengths")
            if max_ndim == 0 and t_dim >= 1:  # scalar params need to be made 1D.
                params = set_vals_to_1d(params, t.shape[1])
            elif max_ndim == 1 and t_dim == 0:  # scalar t needs to be made 1D.
                t = np.full((1, max_length), t)
            max_ndim = max(max_ndim, t_dim)
            J_t = self.get_J_t(t)
            hsiao_interp = self.get_hsiao_interp(t)
        else:
            raise TypeError(
                "Either 't' must be defined (scalar or ArrayLike) or both 'J_t' "
                "(ArrayLike (N_sn, N_tau_knots, N_max_epochs)) and 'hsiao_interp' "
                "(ArrayLike (3, N_max_epochs, N_sn) must be defined."
            )

        model_spectra = self._get_spectra(
            eps=eps,
            theta=params["theta"],
            AV=params["AV"],
            RV=params["RV"],
            W0=W0,
            W1=W1,
            J_t=J_t,
            hsiao_interp=hsiao_interp,
        )
        if max_ndim == 2:
            return model_spectra
        elif max_ndim == 1:
            return model_spectra[..., 0]
        return model_spectra[0, ..., 0]

    def get_flux_batch(
        self,
        model_spectra: ArrayLike,
        M0: Number | ArrayLike,
        Ds: ArrayLike,
        z: ArrayLike,
        ebv: ArrayLike,
        band_indices: ArrayLike,
        mask: ArrayLike,
        weights: ArrayLike,
        mag_shift: Number | ArrayLike,
        num_batch: int,
        **kwargs: Any,
    ) -> Array:
        """
        Calculates observer-frame fluxes for given parameter values.

        Parameters
        ----------
        model_spectra : (N_sn, N_wl, N_max_epochs)
            Array of SEDs indexed by SN, wavelengths, and epoch.
            Produced from ``SEDmodel._get_spectra`` or ``SEDmodel.get_spectra``.
        M0 : scalar or (N_sn,)
            Normalising constant to scale Hsiao template to correct order of magnitude.
        Ds : (N_sn,)
            Distance moduli for each SN.
        z : (N_sn,)
            Heliocentric redshifts for each SN.
        ebv : (N_sn,)
            MW E(B-V) values for each SN.
        band_indices : (N_max_epochs, N_sn)
            Array of integers describing which filter each observation is in.
        mask : (N_max_epochs, N_sn)
            Array containing booleans describing whether each observation should
            contribute to the posterior.
        weights : (N_sn, N_wl, N_bandpasses)
            Array containing band weights (transmission functions) to use for photometry.
        mag_shift : scalar or (N_bandpasses,)
            Zero-point offsets in magnitudes.
        num_batch
            Number of SNe (N_sn).
        **kwargs
            Additional arguments not used, but allowed to support passing kwarg dicts.

        Returns
        -------
        (N_max_epochs, N_sn)
            Matrix containing model fluxes for all SNe at all epochs.
        """
        num_observations = band_indices.shape[0]

        batch_indices = jnp.arange(num_batch).repeat(num_observations).astype(int)
        obs_band_weights = (
            weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .transpose(0, 2, 1)
        )

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=1).T
        mag_shift = jnp.empty(self.used_band_inds.shape[0]).at[:].set(mag_shift)
        zps = (self.used_zps + mag_shift)[band_indices]
        model_flux *= 10 ** (-0.4 * (M0 + Ds + zps - self.ZPT))
        model_flux *= mask
        return model_flux


    def get_mag_batch(
        self,
        model_spectra: ArrayLike,
        M0: Number | ArrayLike,
        Ds: ArrayLike,
        z: ArrayLike,
        ebv: ArrayLike,
        band_indices: ArrayLike,
        mask: ArrayLike,
        weights: ArrayLike,
        mag_shift: Number | ArrayLike,
        num_batch: int,
        **kwargs: Any,
    ) -> Array:
        """
        Calculates observer-frame magnitudes for given parameter values.

        Parameters
        ----------
        model_spectra : (N_sn, N_wl, N_max_epochs)
            Array of SEDs indexed by SN, wavelengths, and epoch.
        M0 : scalar or (N_sn,)
            Normalising constant to scale Hsiao template to correct order of magnitude.
        Ds : (N_sn,)
            Distance moduli for each SN.
        z : (N_sn,)
            Heliocentric redshifts for each SN.
        ebv : (N_sn,)
            MW E(B-V) values for each SN.
        band_indices : (N_max_epochs, N_sn)
            Array of integers describing which filter each observation is in.
        mask : (N_max_epochs, N_sn)
            Array containing booleans describing whether each observation should
            contribute to the posterior.
        weights : (N_sn, N_wl, N_bandpasses)
            Array containing band weights (transmission functions).
        mag_shift : scalar or (N_bandpasses,)
            Zero-point offsets in magnitudes.
        num_batch
            Number of SNe (N_sn).
        **kwargs
            Additional arguments not used, but allowed to support passing kwarg dicts.

        Returns
        -------
        (N_max_epochs, N_sn)
            Matrix containing model magnitudes for all SNe at all epochs.
        """
        model_flux = self.get_flux_batch(
            model_spectra,
            M0,
            Ds,
            z,
            ebv,
            band_indices,
            mask,
            weights,
            mag_shift,
            num_batch,
        )
        model_flux = (
            model_flux + (1 - mask) * 0.01
        )  # Masked data points are set to 0, set them to a small value
        # to avoid nans when logging

        model_mag = -2.5 * jnp.log10(model_flux) + self.ZPT
        model_mag *= mask  # Re-apply mask

        return model_mag

    def get_flux_from_chains(
        self,
        t: ArrayLike,
        bands: list[str] | list[list[str]],
        chains: str | Path | dict,
        zs: ArrayLike | Number,
        ebv_mws: ArrayLike | Number,
        lam_shifts: ArrayLike | Number = 0,
        mag_shifts: ArrayLike | Number = 0,
        mag: bool = True,
        num_samples: int | None = None,
        num_sne: int | None = None,
        mean: bool = False,
    ) -> Array:
        """
        Returns model photometry for posterior samples from BayeSN fits.

        Parameters
        ----------
        t : (N_phase,)
            Array of phases to evaluate model photometry at.
        bands
            List of bandpass names, or list of lists of bandpass names per SN.
        chains
            Path to file containing BayeSN fitting posterior samples, or dictionary
            of posterior samples with keys including 'theta', 'AV', 'eps', 'tmax',
            'mu', and 'delM'.
        zs : (N_sn,) or scalar
            Heliocentric redshifts corresponding to the SNe.
        ebv_mws : (N_sn,) or scalar
            Milky Way extinction values corresponding to the SNe.
        lam_shifts : (N_bands,) or scalar
            Wavelength shifts (Angstroms).
        mag_shifts : (N_bands,) or scalar
            Zero-point offsets in magnitudes.
        mag
            Whether to return magnitudes (True) or flux densities (False).
        num_samples
            Number of posterior samples to evaluate.
        num_sne
            Number of SNe.
        mean
            If True, generate only one flux time-series for each SN using mean parameter values.

        Returns
        -------
        (N_sn, num_samples, N_bands, N_phase)
            Array containing photometry for all SNe, posterior samples, bands, and phases.
        """
        if isinstance(chains, str | Path):
            with open(chains, "rb") as file:
                chains = pickle.load(file)

        chain_samples = chains["Ds"].shape[0] * chains["Ds"].shape[1]
        if num_sne is None:
            num_sne = chains["Ds"].shape[2]
        if mean:
            num_samples = 1
        elif num_samples is None:
            num_samples = chain_samples
        elif num_samples > chain_samples:
            raise ValueError(
                f"num_samples was specified as {num_samples}, but there are only "
                f"{chain_samples} samples to draw from. Either pass longer chains or "
                "decrease num_samples."
            )

        if isinstance(zs, Number | Array) and getattr(zs, "ndim", 0) == 0:
            zs = np.full(num_sne, zs)
        elif zs.shape != (num_sne,):
            raise ValueError(f"zs must be a number or 1D array with length {num_sne}.")
        if isinstance(ebv_mws, Number | Array) and getattr(ebv_mws, "ndim", 0) == 0:
            ebv_mws = np.full(num_sne, ebv_mws)

        band_list = isinstance(bands[0], list)
        if band_list:
            if len(bands) != num_sne:
                raise ValueError(
                    f"bands is a list of {len(bands)} lists, which cannot be mapped to "
                    f"num_sne={num_sne} SNe."
                )
            max_bands = np.max([len(sublist) for sublist in bands])
            unique_bands = list(set(b for sublist in bands for b in sublist))
        else:
            max_bands = len(bands)
            unique_bands = list(np.unique(bands))
        self._set_used_bands(unique_bands)
        self.band_weights = self._calculate_band_weights(zs, ebv_mws, lam_shifts=lam_shifts)

        if isinstance(lam_shifts, Number):
            lam_shifts = np.full(max_bands+1, lam_shifts)
        elif len(lam_shifts) == len(bands):
            lam_shifts = np.append(0, lam_shifts)  # NULL band padding
        elif len(lam_shifts) != len(bands)+1:
            raise ValueError(
                f"lam_shifts does not map to the unique bands {unique_bands}."
            )
        if isinstance(mag_shifts, Number):
            mag_shifts = np.full(max_bands+1, mag_shifts)
        elif len(mag_shifts) == len(bands):
            mag_shifts = np.append(0, mag_shifts)  # NULL band padding
        elif len(mag_shifts) != len(bands)+1:
            raise ValueError(
                f"mag_shifts does not map to the unique bands {unique_bands}."
            )

        flux_grid = jnp.zeros((num_sne, num_samples, max_bands, len(t)))
        print("Getting best fit light curves from chains...")
        for i in tqdm(np.arange(num_sne)):
            # Defining SN-specific terms
            if band_list:
                sn_bands = bands[i]
            else:
                sn_bands = bands
            # Padding for NULL_BAND
            sn_bp_inds = [0] + [self.used_band_dict[self.band_dict[b]] for b in sn_bands]
            sn_weights = self.band_weights[i:i+1, :, sn_bp_inds]
            sn_lam_shifts = np.array(lam_shifts)[sn_bp_inds]
            sn_mag_shifts = np.array(mag_shifts)[sn_bp_inds]

            params = {
                key: chains[key][..., i].flatten(order="F") if key in chains
                else None for key in ("theta", "AV", "tmax", "mu", "delM")
            }
            if "RV" in chains:
                if chains["RV"].ndim == 2:  # global
                    params["RV"] = chains["RV"].flatten(order="F")
                elif chains["RV"].ndim == 3:  # SN-specific or duped because not sampled
                    params["RV"] = chains["RV"][..., i].flatten(order="F")
            else:
                params["RV"] = np.full(num_samples, float(self.RV or self.mu_R))
            eps = chains["eps"][..., i]
            eps = eps.reshape((eps.shape[0] * eps.shape[1], eps.shape[2]), order="F")
            eps = eps.reshape(
                (eps.shape[0], self.l_knots.shape[0] - 2, self.tau_knots.shape[0]),
                order="F",
            )
            eps_full = jnp.zeros(
                (eps.shape[0], self.l_knots.shape[0], self.tau_knots.shape[0])
            )
            params["eps"] = eps_full.at[:, 1:-1, :].set(eps)

            theta, AV, mu, eps, del_M, tmax, RV = (
                params["theta"][:num_samples],
                params["AV"][:num_samples],
                params["mu"][:num_samples],
                params["eps"][:num_samples],
                params["delM"][:num_samples],
                params["tmax"][:num_samples],
                params["RV"][:num_samples],
            )
            if mean:
                theta, AV, mu, eps, del_M, tmax, RV = (
                    theta.mean()[None],
                    AV.mean()[None],
                    mu.mean()[None],
                    eps.mean(axis=0)[None],
                    del_M.mean()[None],
                    tmax.mean()[None],
                    RV.mean()[None],
                )

            lc, lc_err, params = self.simulate_light_curve(
                t,
                theta.shape[0],
                sn_bands,
                theta=theta,
                AV=AV,
                mu=mu,
                tmax=tmax,
                del_M=del_M,
                eps=eps,
                lam_shift=sn_lam_shifts,
                mag_shift=sn_mag_shifts,
                RV=RV,
                z=zs[i],
                write_to_files=False,
                ebv_mw=ebv_mws[i],
                yerr=0,
                mag=mag,
                band_weights=sn_weights,
            )
            lc = lc.T
            lc = lc.reshape(num_samples, len(sn_bands), len(t))
            flux_grid = flux_grid.at[i, :, : len(sn_bands), :].set(lc)

        return flux_grid
    #########################
    ### Numpyro Modelling ###
    #########################
    def _model(
        self,
        obs: ObsData,
        weights: ArrayLike,
        train_new_model: bool,
        infer_dust_properties: bool,
        shared_RV: bool,
        RV: Number | str,
        vary_redshift: bool,
        fix_tmax: bool,
        fix_eps: bool,
        vary_filter_shifts: bool,
        vary_offsets: bool,
        M_split: float,
        split_variant: str | None,
        data_type: str,
        photoz: bool,
        prior_only: bool = False,
        **kwargs,
    ) -> None:
        """
        Main NumPyro model function that dynamically calls modular sampling methods.

        Parameters
        ----------
        obs
            An instance of ObsData. The keys can be grouped by shape:

            * Arrays with shape (N_sn,): ``host_logmass``, ``z_hel``, ``z_hel_err``,
              ``muhat``, ``MWEBV``
            * Arrays with shape (N_max_epochs, N_sn): ``mjd``, ``flux``, ``flux_err``,
              ``band_indices``, ``mask``

            The ``flux`` and ``flux_err`` field names are used even when the data are in mags.
        weights : (N_sn, N_wl, N_bandpasses)
            Band weights based on filter responses and MW extinction curves for
            numerical flux integrals. Produced by ``SEDmodel._calculate_band_weights``.
        train_new_model
            If True, draw new samples for W0, W1, and L_Sigma.
            If False, return the model's pre-computed values.
        infer_dust_properties
            If True, samples sigma0 and tauA. If RV is "normal", also samples
            mu_R and sigma_R. If vary_redshift is True, also samples mu_z_grad
            and tau_z_grad. If False, uses pre-computed model attributes.
        shared_RV
            If True, all SNe will share a common RV that may be specified by a numeric
            RV argument, or drawn from a distribution specified by the RV argument and
            others (which probably should not be sampled).
        RV
            If "normal", RV(s) will be sampled from a truncated normal with mean mu_R,
            scale sigma_R, and lower limit 1.2. If "uniform", RV(s) will be sampled from
            U(uniform_RV_min, uniform_RV_max). If a number, all RVs will be that number
            regardless of shared_RV.
        vary_redshift
            If True, samples mu_z_grad and tau_z_grad. These parameters describe a linear
            trend correlation between redshift and mu_R and tauA.
        fix_tmax
            If True, uses the peak_mjd values taken from the data as tmax.
            If False, samples tmax for each SN.
        fix_eps
            If True, sets all eps matrices to 0 everywhere, otherwise sampled.
        vary_filter_shifts
            If True, samples wavelength shifts (Angstroms) for each filter.
            If False, treats all shifts as 0 Angstroms.
        vary_offsets
            If True, samples the magnitude shifts in each bandpasses' zero-point.
            If False, treats all shifts as 0 mag.
        M_split
            If split_variant is not None, the high- and low-mass populations will be
            determined by whether each host-galaxy mass is greater or less than M_split.
            Unitless since host-galaxy mass is log_10(M_stellar/M_sun).
        split_variant
            If not None, it can either be "split_mag" or "split_sed". The high- and low-mass
            galaxy populations will have their dust parameters sampled independently (e.g.
            sigma0, tauA, mu_R, sigma_R). If "split_mag", samples shifts to M0 for the high-
            and low-mass galaxy populations. If "split_sed", samples shifts to W0 for the
            high- and low-mass galaxy populations.
        data_type
            Either "flux" or "mag", indicating whether to treat the input data as fluxes
            or magnitudes.
        photoz
            If True, infer host galaxy redshift from photometric observations.
        prior_only
            If True, return after sampling all latents and skip the data-likelihood
            (``get_flux_batch`` and the obs sample). Used by ``_prior_pot`` to
            compute the prior log-density without the cost or memory footprint
            of running the model's flux computation. Default is False.

        Other Parameters
        ----------------
        fix_theta
            If not None, float(fix_theta) will be used as the theta value for all SNe.
            Default is None.
        fix_AV
            If not None, float(fix_AV) will be used as the AV value for all SNe.
            Default is None.
        AV_val
            If fix_AV is True, AV_val will be used for all SNe. Default is 0.
        muhat_err : (N_sn,) or scalar
            The error on muhat, which is cosmological distance inferred from the
            fiducial cosmology in SEDmodel.cosmo and cosmological redshift.
            If scalar, use the input value for all SNe. If None, calculate muhat_err
            for each SN from their redshift, redshift error, and the peculiar velocity
            from SEDmodel.sigma_pec. Default is None.
        fix_dist_limit
            The redshift limit below which errors on Ds are calculated from muhat_err
            and sigma0, and above which errors are forced to be fix_dist_Ds_err.
            The motivation is to relax the constraints on Ds at redshifts where
            cosmological effects may play a non-trivial role. Default is 0.08.
        fix_dist_Ds_err
            The Ds_err value to assume for SNe at redshifts above fix_dist_limit. Default is 5.
        mu_R
            The mean of the normal distribution of RV. Default is 3.
        sigma_R
            The standard deviation of the normal distribution of RV. Default is 0.5.
        mu_R_min
            If infer_dust_properties is True, the input mu_R parameter is ignored
            and instead sampled as mu_R ~ U(mu_R_min, mu_R_max). 1.2 is the value for pure
            Rayleigh Scattering (A propto wl^-4) as estimated in Draine 2003, 2003ARA&A..41..241D.
            Default is 1.2.
        mu_R_max
            If infer_dust_properties is True, the input mu_R parameter is ignored
            and instead sampled as mu_R ~ U(mu_R_min, mu_R_max). Very large grains would
            produce extinction curves with no theoretical upper limit on RV, but many extinction
            curves use data capping out around RV = 6 (e.g. Cardelli 1989, Fitzpatrick 1999/2019, etc.).
            Default is 6.
        sigma_sigma_R
            If infer_dust_properties is True, the input sigma_R parameter is ignored
            and instead sampled as sigma_R ~ HalfNormal(sigma_sigma_R). Default is 2.
        uniform_RV_min
            The lower bound of the uniform distribution of RV. Default is 1.
        uniform_RV_max
            The upper bound of the uniform distribution of RV. Default is 6.
        tau_z_min
            The lower bound of the uniform distribution of tau_z_grad. Default is -0.5.
        tau_z_max
            The upper bound of the uniform distribution of tau_z_grad. Default is 0.5.

        Notes
        -----
        If ``vary_redshift`` is True and ``infer_dust_properties`` is True:
        Linear correlations between redshift and mu_R / tauA will be sampled as
        ``mu_z_grad ~ U(mu_R_min - mu_R, mu_R_max - mu_R)`` and
        ``tau_z_grad ~ U(tau_z_min, tau_z_max)``.
        """
        N_sn = obs.mjd.shape[1]

        if train_new_model:
            W0, W1, L_Sigma = self._sample_model_params()
        else:
            W0, W1, L_Sigma = self.W0, self.W1, self.L_Sigma

        if infer_dust_properties:
            dust_pop = self._sample_dust_hyperparams(
                RV=RV,
                split_variant=split_variant,
                vary_redshift=vary_redshift,
                mass=obs.host_logmass,
                M_split=M_split,
                **kwargs,
            )
        else:
            dust_pop = self._get_fixed_dust_hyperparams(
                RV=RV,
                split_variant=split_variant,
                mass=obs.host_logmass,
                M_split=M_split,
                **kwargs,
            )

        if split_variant is None:
            M0 = self.M0
        elif split_variant == "split_mag":
            M0 = self._sample_M0_by_mass(HM_flag=dust_pop.HM_flag)
        elif split_variant == "split_sed":
            M0 = self.M0
            W0 = self._sample_W0_by_mass(HM_flag=dust_pop.HM_flag, W0=W0)

        if shared_RV:
            dust_pop = self._sample_shared_RV(dust_pop, RV)

        if vary_filter_shifts:
            lam_shift = numpyro.sample("lam_shift", dist.Normal(0, self.used_wave_sigmas))
        else:
            lam_shift = 0

        if vary_offsets:
            mag_shift = numpyro.sample("mag_shift", dist.MultivariateNormal(0, scale_tril=self.used_calib_chcov))
        else:
            mag_shift = 0

        with numpyro.plate("SNe", N_sn) as sn_index:
            N_max_epochs = obs.mjd.shape[0]
            band_indices = obs.band_indices[:, sn_index].astype(int)
            phot_mask = obs.mask[:, sn_index].astype(bool)
            if photoz:
                z = self._sample_z(
                    obs.z_hel[sn_index],
                    obs.z_hel_err[sn_index],
                    sn_index=sn_index,
                    z_icdf=kwargs.get("z_icdf"),
                )
            else:
                z = obs.z_hel[sn_index]
            if vary_filter_shifts or photoz:
                # In either case, observer frame transmissions need re-calculation
                # If we decide to sample E(B-V)_MW one day, that will also require
                # re-calculation.
                weights = self._calculate_band_weights(z, obs.MWEBV[sn_index], lam_shift)
            AV, RV = self._sample_split_SN_dust_params(
                dust_pop=dust_pop,
                redshift=z,
                shared_RV=shared_RV,
                **kwargs,
            )
            theta, eps, Ds = self._sample_SN_params(
                N_sn=N_sn,
                redshift=obs.z_hel[sn_index],
                redshift_error=obs.z_hel_err[sn_index],
                muhat=obs.muhat[sn_index],
                L_Sigma=L_Sigma,
                sigma0=dust_pop.sigma0,
                fix_eps=fix_eps,
                **kwargs,
            )

            if not fix_tmax:
                hsiao_interp, J_t, tmax = self._sample_SN_tmax(
                    t_all_sn=obs.mjd,
                    sn_index=sn_index,
                    z_obs=obs.z_hel[sn_index],
                    z_sampled=z,
                    **kwargs,
                )
            else:
                hsiao_interp, J_t, tmax = self.hsiao_interp, self.J_t, None

            if prior_only:
                return

            phot_epoch_spectra = self._get_spectra(theta, AV, W0, W1, eps, RV, J_t, hsiao_interp)
            data_fn = self.get_flux_batch if data_type == "flux" else self.get_mag_batch
            data = data_fn(
                model_spectra=phot_epoch_spectra,
                M0=M0,
                Ds=Ds,
                z=z,
                ebv=obs.MWEBV,
                band_indices=band_indices,
                mask=phot_mask,
                weights=weights,
                lam_shift=lam_shift,
                mag_shift=mag_shift,
                num_batch=N_sn,
            )
            with numpyro.plate("epochs", N_max_epochs, dim=-2):
                with numpyro.handlers.mask(mask=phot_mask):
                    numpyro.sample(
                        f"obs",
                        dist.Normal(data, obs.flux_err[..., sn_index]),
                        obs=obs.flux[..., sn_index],
                    )

    def _sample_model_params(self) -> tuple[Array, Array, Array]:
        """ Sample W0, W1, and L_Sigma

        Returns
        -------
        W0: numpyro Sample
        W1: numpyro Sample
        L_Sigma: numpyro Sample
        """
        W_mu = jnp.zeros(self.N_knots)
        W0 = numpyro.sample("W0", dist.MultivariateNormal(W_mu, jnp.eye(self.N_knots)))
        W1 = numpyro.sample("W1", dist.MultivariateNormal(W_mu, jnp.eye(self.N_knots)))
        W0 = jnp.reshape(
            W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
        )
        W1 = jnp.reshape(
            W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
        )

        sigmaepsilon = numpyro.sample("sigmaepsilon", dist.HalfCauchy(jnp.ones(self.N_knots_sig)))

        # Cholesky factors L of correlation matrix Omega
        # Covariance matrix Sigma = diag(sigmaepsilon) Omega diag(sigmaepsilon)
        # Or, Sigma = L_Sigma L_Sigma.T with L_Sigma = diag(sigmaepsilon) L_Omega
        L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(self.N_knots_sig))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)

        return W0, W1, L_Sigma

    def _sample_model_dust_params(
        self,
        suffix: str = "",
        vary_redshift: bool = False,
        RV: Number | str = "normal",
        mu_R: ArrayLike | None = None,
        sigma_R: ArrayLike | None = None,
        mu_R_min: float = 1.2,
        mu_R_max: float = 6,
        sigma_sigma_R: float = 2,
        tau_z_min: float = -0.5,
        tau_z_max: float = 0.5,
        uniform_RV_min: float = 1,
        uniform_RV_max: float = 6,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array | None, Array | None, Array | None, Array | None, Array | None, Array | None]:
        """Draw SN population level parameters that may vary by sub-population
        (e.g. split by galaxy mass)."""
        mu_z_grad, tau_z_grad = [0 for _ in range(2)]

        sigma0 = numpyro.sample(f"sigma0{suffix}", dist.HalfCauchy(0.1))
        tauA = numpyro.sample(f"tauA{suffix}", dist.HalfCauchy(1))

        if RV == "normal":
            mu_R = numpyro.sample(f"mu_R{suffix}", dist.Uniform(mu_R_min, mu_R_max))
            sigma_R = numpyro.sample(f"sigma_R{suffix}", dist.HalfNormal(sigma_sigma_R))

        if vary_redshift:
            mu_z_grad = numpyro.sample(
                f"mu_z_grad{suffix}",
                dist.Uniform(mu_R_min - mu_R, mu_R_max - mu_R)
            )
            tau_z_grad = numpyro.sample(
                f"tau_z_grad{suffix}",
                dist.Uniform(tau_z_min, tau_z_max)
            )
        return DustHyperParams(
            sigma0, tauA,
            mu_R, sigma_R,
            uniform_RV_min, uniform_RV_max,
            mu_z_grad, tau_z_grad, RV
        )

    def _get_fixed_model_dust_params(
        self,
        RV: Number | str | None = None,
        mu_R: ArrayLike | None = None,
        sigma_R: ArrayLike | None = None,
        uniform_RV_min: float = 1,
        uniform_RV_max: float = 6,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array | None, Array | None, Array | None, int, int, Array]:
        """Return pre-computed model dust parameters."""
        if RV is None:
            RV = getattr(self, "RV")
        if mu_R is None:
            mu_R = getattr(self, "mu_R")
        if sigma_R is None:
            sigma_R = getattr(self, "sigma_R")
        return DustHyperParams(
            self.sigma0, self.tauA,
            mu_R, sigma_R,
            uniform_RV_min, uniform_RV_max,
            0, 0, RV
        )

    def _sample_dust_hyperparams(
        self,
        RV: Number | str,
        split_variant: str | None,
        vary_redshift: bool,
        mass: ArrayLike,
        M_split: float,
        **kwargs: Any,
    ) -> DustPop:
        """Sample population dust parameters (across mass-split populations if specified)."""
        HM_flag = mass > M_split
        suffix = "_HM" if split_variant is not None else ""
        hm_dust = self._sample_model_dust_params(
            suffix=suffix,
            RV=RV,
            vary_redshift=vary_redshift,
            **kwargs,
        )
        lm_dust = None
        sigma0 = hm_dust.sigma0
        if split_variant is not None:
            lm_dust = self._sample_model_dust_params(
                suffix="_LM",
                RV=RV,
                vary_redshift=vary_redshift,
                **kwargs,
            )
            sigma0 = HM_flag * hm_dust.sigma0 + (1 - HM_flag) * lm_dust.sigma0

        dust_pop = DustPop(
            HM=hm_dust,
            LM=lm_dust,
            HM_flag=HM_flag,
            sigma0=sigma0,
        )
        return dust_pop

    def _get_fixed_dust_hyperparams(
        self,
        split_variant: str | None,
        mass: ArrayLike,
        M_split: float,
        **kwargs: Any,
    ) -> DustPop:
        """Retrieve pre-computed model dust parameters (and sample mass-split steps if specified)."""
        HM_flag = mass > M_split
        hm_dust = self._get_fixed_model_dust_params(**kwargs)
        lm_dust = None
        sigma0 = hm_dust.sigma0
        if split_variant is not None:
            lm_dust = self._get_fixed_model_dust_params(**kwargs)
            sigma0 = HM_flag * hm_dust.sigma0 + (1 - HM_flag) * lm_dust.sigma0

        dust_pop = DustPop(
            HM=hm_dust,
            LM=lm_dust,
            HM_flag=HM_flag,
            sigma0=sigma0,
        )
        return dust_pop

    def _sample_M0_by_mass(self, HM_flag: ArrayLike) -> ArrayLike:
        M_step_HM = numpyro.sample("M_step_HM", dist.Uniform(-0.2, 0.2))
        M_step_LM = numpyro.sample("M_step_LM", dist.Uniform(-0.2, 0.2))
        M0 = (
            jnp.full_like(HM_flag, self.M0, dtype=float)
            + HM_flag * M_step_HM
            + (1 - HM_flag) * M_step_LM
        )
        return M0

    def _sample_W0_by_mass(self, HM_flag: ArrayLike, W0: ArrayLike) -> ArrayLike:
        W_mu = jnp.zeros(self.N_knots)
        delW_HM = numpyro.sample(
            "delW_HM", dist.MultivariateNormal(W_mu, 0.1 * jnp.eye(self.N_knots))
        )
        delW_LM = numpyro.sample(
            "delW_LM", dist.MultivariateNormal(W_mu, 0.1 * jnp.eye(self.N_knots))
        )
        delW_HM = jnp.reshape(
            delW_HM, (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
        )
        delW_LM = jnp.reshape(
            delW_LM, (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
        )
        W0_HM = numpyro.deterministic("W0_HM", W0 + delW_HM)
        W0_LM = numpyro.deterministic("W0_LM", W0 + delW_LM)
        W0 = (
            HM_flag[:, None, None] * W0_HM[None, ...]
            + (1 - HM_flag)[:, None, None] * W0_LM[None, ...]
        )
        return W0

    def _sample_shared_RV(self, dust_pop: DustPop, RV: str | Number) -> DustPop:
        """ Sample the RV used by all SNe within a given population. """
        split = dust_pop.LM is not None
        suffix = "_HM" if split else ""
        hm_dust, lm_dust = dust_pop.HM, dust_pop.LM
        if RV == "uniform":
            hm_RV = numpyro.sample(
                f"RV{suffix}",
                dist.Uniform(hm_dust.uniform_RV_min, hm_dust.uniform_RV_max)
            )
            if split:
                lm_RV = numpyro.sample(
                    f"RV_LM",
                    dist.Uniform(lm_dust.uniform_RV_min, lm_dust.uniform_RV_max)
                )
        elif RV == "normal":
            hm_RV = numpyro.sample(
                f"RV{suffix}",
                dist.TruncatedNormal(hm_dust.mu_R, hm_dust.sigma_R, low=1.2),
            )
            if split:
                lm_RV = numpyro.sample(
                    f"RV_LM",
                    dist.TruncatedNormal(lm_dust.mu_R, lm_dust.sigma_R, low=1.2),
                )
        else:
            hm_RV = lm_RV = RV
        hm_dust = DustHyperParams(*hm_dust[:-1], hm_RV)
        if split:
            lm_dust = DustHyperParams(*lm_dust[:-1], lm_RV)
        dust_pop = DustPop(hm_dust, lm_dust, dust_pop.HM_flag, dust_pop.sigma0)
        return dust_pop

    def _sample_z(
        self,
        z_obs: ArrayLike,
        z_obs_err: ArrayLike,
        sn_index: ArrayLike,
        z_icdf: ArrayLike | None = None,
    ) -> Array:
        """Sample or interpolate redshift for each SN."""
        if self.z_icdf_grid is not None:  # per-SN host photo-z PDF via ICDF-reparam
            u = numpyro.sample('u', dist.Uniform(self.z_u_grid[0], self.z_u_grid[-1]))
            if z_icdf is not None:  # single per-SN row passed in (VI vmaps over SNe)
                z_sampled = numpyro.deterministic('z', jnp.interp(u, self.z_u_grid, z_icdf))
            else:  # MCMC: one plate over all SNe, index the shared table
                z_sampled = numpyro.deterministic('z', jax.vmap(jnp.interp, in_axes=(0, None, 0))(
                    u, self.z_u_grid, self.z_icdf_grid[sn_index]))
        else:  # Gaussian catalog prior
            ztform = numpyro.sample('ztform', dist.Normal(0, 1))
            z_sampled = numpyro.deterministic('z', z_obs + z_obs_err * ztform)
        # Rest-frame phase from the observer frame at the sampled z (time dilation)
        return z_sampled

    def _sample_SN_dust_params(
        self,
        dust_params: DustHyperParams | None = None,
        AV_dist: Callable[[float], Array] = dist.Exponential,
        fix_AV: float | None = None,
        shared_RV: bool = True,
        redshift: float | ArrayLike = 0,
        suffix: str = "",
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        """Sample AV and RV for each SN.

        Parameters
        ----------
        dust_params: DustHyperParams, optional
            NamedTuple containing population dust parameters.
        AV_dist:
            The stochastic function used to draw AV as fn(1/tauA).
        fix_AV: default None
            If not None, use float(fix_AV) as the AV value for all SNe.
        shared_RV: default True
            If True, use the same RV for all SNe, otherwise sample.
        redshift: default 0
            Needed if mu_z_grad or tau_z_grad are not 0.
        suffix: default ""
            This string is appended to the parameter naming scheme used by numpyro.

        Returns
        -------
            AV:
            RV:
        """
        suffix = f"_{suffix}".replace("__", "_").rstrip("_")
        if fix_AV is not None:
            AV = jnp.array([float(fix_AV)])
        else:
            AV = numpyro.sample(f"AV{suffix}", AV_dist(1 / (dust_params.tauA + redshift * dust_params.tau_z_grad)))

        z_drift = redshift * dust_params.mu_z_grad
        if shared_RV:
            RV = dust_params.RV + z_drift
        elif dust_params.RV == "uniform":
                RV = numpyro.sample(
                    f"RV{suffix}", dist.Uniform(
                        dust_params.uniform_RV_min + z_drift, dust_params.uniform_RV_max + z_drift
                    )
                )
        elif dust_params.RV == "normal":
            RV = numpyro.sample(
                f"RV{suffix}",
                dist.TruncatedNormal(
                    dust_params.mu_R + z_drift, dust_params.sigma_R, low=1.2
                ),
            )
        return AV, RV

    def _sample_split_SN_dust_params(
        self,
        dust_pop: DustPop,
        shared_RV: bool,
        redshift: ArrayLike,
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        """Sample AV and RV for each SN given population dust parameters."""
        HM_flag = dust_pop.HM_flag
        AV, RV = self._sample_SN_dust_params(
            dust_params=dust_pop.HM,
            shared_RV=shared_RV,
            redshift=redshift,
            suffix="HM" * (dust_pop.LM is not None),
            **kwargs,
        )
        if dust_pop.LM is not None:
            AV_LM, RV_LM = self._sample_SN_dust_params(
                dust_params=dust_pop.LM,
                shared_RV=shared_RV,
                redshift=redshift,
                suffix="LM",
                **kwargs,
            )
            AV = numpyro.deterministic("AV", HM_flag * AV + (1 - HM_flag) * AV_LM)
            RV = numpyro.deterministic("RV", HM_flag * RV + (1 - HM_flag) * RV_LM)
        return AV, RV

    def _sample_SN_params(
        self,
        N_sn: int,
        redshift: ArrayLike,
        redshift_error: ArrayLike,
        muhat: ArrayLike,
        # sn_obs: ArrayLike,
        L_Sigma: ArrayLike,
        sigma0: float | ArrayLike,
        fix_theta: float | None = None,
        muhat_err: float | None = None,
        fix_dist_limit: float = 0.08,
        fix_dist_Ds_err: float = 5,
        fix_eps: bool = False,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array]:
        """ Sample theta, eps, Ds for each SN.

        Parameters
        ----------
        N_sn:
            Total number of SN.
            This information is required because eps is delivered as a matrix of shape
            (N_sn, N_l_knots, N_tau_knots) where the N_sn broadcasting is handled by
            calling this function within a numpyro plate.
        redshift:
            Heliocentric redshift. Array of length N_sn
        redshift_error:
            Error on heliocentric redshift. Array of length N_sn
        muhat:
            Redshift/cosmology-based distance. Array of length N_sn
        L_Sigma:
            The covariance matrix for the prior of epsilon.
            The shape is (N_knots, N_knots) array where N_knots is the product of
            (N_l_knots - 2) and N_tau_knots. The - 2 is because the bluest and reddest
            bins in the full epsilon matrix are fixed to 0 at all phase bins.
        sigma0:
            Model-specific standard deviation in distance modulus to be added in
            quadrature to the error in cosmological distance modulus.
            If float-like, use the same sigma0 value for all SNe.
            If split_variant is not None, this will be an array of shape (N_sn,).
            The sigma0 values will be broadcast to each SN.
        fix_theta: default None
            If not None, float(fix_theta) will be used as the theta value for all SNe.
        muhat_err:
            If None, calculate muhat_err as 5*sqrt(z_err**2 + self.sigma_pec**2)/(z*ln(10))
            If scalar, use muhat_err to calculate Ds_err

        Returns
        -------
            theta:
            eps:
            Ds:
        """
        # redshift, redshift_error, muhat = sn_obs[5:8, 0]

        if fix_theta is not None:
            theta = jnp.array([float(fix_theta)])
        else:
            theta = numpyro.sample(f"theta", dist.Normal(0, 1.0))

        eps_full = jnp.zeros((N_sn, self.l_knots.shape[0], self.tau_knots.shape[0]))
        if not fix_eps:
            eps_mu = jnp.zeros(self.N_knots_sig)
            eps_tform = numpyro.sample(
                "eps_tform", dist.MultivariateNormal(eps_mu, jnp.eye(self.N_knots_sig))
            )
            eps_tform = eps_tform.T
            eps = numpyro.deterministic("eps", jnp.matmul(L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(
                eps,
                (N_sn, self.N_knots_sig_l, self.tau_knots.shape[0]),
                order="F",
            )
            eps = eps_full.at[:, 1:-1, :].set(eps)
        else:
            eps = eps_full

        # x * x seems more performant than x ** 2 or jnp.power(x, 2)
        if muhat_err is None:
            muhat_err = (
                5
                / (redshift * jnp.log(10))
                * jnp.sqrt(redshift_error * redshift_error + self.sigma_pec * self.sigma_pec)
            )
        Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
        fix_dist = redshift < fix_dist_limit
        Ds_err = Ds_err * fix_dist + fix_dist_Ds_err * (1 - fix_dist)
        Ds_tform = numpyro.sample("Ds_tform", dist.Normal(0, 1))
        Ds = numpyro.deterministic("Ds", muhat + Ds_tform * Ds_err)

        return theta, eps, Ds

    def _sample_SN_tmax(
        self,
        t_all_sn: ArrayLike,
        sn_index: ArrayLike,
        z_obs: ArrayLike,
        z_sampled: ArrayLike,
        fix_tmax: bool = False,
        tmax_min: float = -10,
        tmax_max: float = 10,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array | None]:
        """ Draw tmax samples for each SN and provide hsiao_interp and J_t.
        This only includes tmax.
        This does not sample theta, eps, or Ds, which come from _sample_SN_params.
        Nor does this sample AV and RV, which come from _sample_SN_dust_params.

        Parameters
        ----------
        t_all_sn: ArrayLike
            Shape (N_max_epochs, N_sn) array containing the phase of each epoch of
            photometry for each SN. N_max_epochs is the greatest number of epochs for
            any single SN in the data set, and the rest are padded with 0s for phases.
            These padded values are masked out during sampling.
        sn_index:
            The plate-level indices of the SN being sampled. The shape is (N_sn,).
            quadrature to the error in cosmological distance modulus.
        fix_tmax: default False
            If True, return the pre-calculated self.hsiao_interp and self.J_t values.
            If False, sample tmax ~ U(tmax_min, tmax_max) and adjust all phases from
            t_all_sn accordingly, then recalculate hsiao_interp and J_t.

        Returns
        -------
            hsiao_interp:
            J_t:
            tmax:
        """
        if fix_tmax:
            return self.hsiao_interp, self.J_t, None
        tmax = numpyro.sample("tmax", dist.Uniform(tmax_min, tmax_max))
        t_all_sn = t_all_sn * (1+z_obs)/(1+z_sampled) - tmax[None, sn_index]
        J_t = self.get_J_t(t_all_sn)
        hsiao_interp = self.get_hsiao_interp(t_all_sn)
        return hsiao_interp, J_t, tmax

    def initial_guess(
        self,
        args: dict,
        reference_model: str | Path,
        RNG_seed: int | None = None,
        jitter: Number = 0.01,
        single_sn: bool = False,
    ) -> dict[str, Any]:
        """
        Sets initialisation for training chains using parameter values from a reference model.

        Parameters
        ----------
        args
            Configuration dictionary.
        reference_model
            Previously-trained model to be used to set initialisation.
        RNG_seed
            Random seed for initialisation jitter.
        jitter
            Scale of Gaussian perturbation added to initial parameters.
        single_sn
            Whether initialisation is for a single supernova.

        Returns
        -------
        dict
            Dictionary containing initial parameter values for sampling.
        """
        # Set hyperparameter initialisations
        built_in_models = [f.name for f in self.__root_dir__.glob("model_files/*_model")]
        if isinstance(reference_model, str | Path) and Path(reference_model).exists():
            print(f"Using custom model at {reference_model} to initialise chains.")
            with open(reference_model, "r") as file:
                params = yaml.load(file)
        elif isinstance(reference_model, str) and reference_model in built_in_models:
            print(f'Loading built-in model {reference_model} to initialise chains')
            with open(self.__root_dir__ / f"model_files/{reference_model}/BAYESN.YAML", "r") as file:
                params = yaml.load(file)
        else:
            raise ValueError(
                "Invalid initialisation method, please choose either 'median' or "
                "'sample', or choose either one of the built-in models or a custom "
                "model to base the hyperparmeter initialisation on."
            )
        N_sn = 1 if single_sn else self.dataset.N_sn
        N_eps = self.N_knots_sig
        rng = np.random.default_rng(RNG_seed)  # different from main, no RNG_seed
        W0_init, W1_init = params["W0"], params["W1"]
        l_knots, tau_knots = params["L_KNOTS"], params["TAU_KNOTS"]
        tauA_init  = params["TAUA"]
        sigma0_init = params.get("SIGMA0", 0.1)  # different from main, was just 0.1
        sigmaepsilon_init = np.abs(rng.standard_cauchy(N_eps))  # different from main
        L_Omega_init = jnp.eye(N_eps)
        param_init = {}

        if args["train_new_model"]:  # parsing based on args is different from main
            # Cannot assume l/tau_knots of reference_model match self.
            # Interpolate to match new wavelength knots
            W0_init = interp1d(l_knots, W0_init, kind="cubic", axis=0, fill_value=0, bounds_error=False)(self.l_knots)
            W1_init = interp1d(l_knots, W1_init, kind="cubic", axis=0, fill_value=0, bounds_error=False)(self.l_knots)
            # Interpolate to match new time knots
            W0_init = interp1d(tau_knots, W0_init, kind="linear", axis=1, fill_value=0, bounds_error=False)(self.tau_knots)
            W1_init = interp1d(tau_knots, W1_init, kind="linear", axis=1, fill_value=0, bounds_error=False)(self.tau_knots)
            W0_init = W0_init.flatten(order="F")
            W1_init = W1_init.flatten(order="F")
            param_init["W0"] = jnp.array(rng.normal(W0_init, jitter))
            param_init["W1"] = jnp.array(rng.normal(W1_init, jitter))

            # param_init["sigmaepsilon_tform"] = jnp.arctan(
            #     sigmaepsilon_init + rng.normal(0, jitter, sigmaepsilon_init.shape) / 1.)
            param_init["sigmaepsilon"] = jnp.array(rng.normal(sigmaepsilon_init, jitter))

            param_init["L_Omega"] = jnp.array(L_Omega_init)

        if args["infer_dust_properties"]:
            param_init["tauA"] = jnp.array(truncnorm.rvs(
                -tauA_init/jitter, np.inf, loc=tauA_init,
                scale=jitter, size=1, random_state=rng
            )[0])
            param_init["sigma0"] = jnp.array(truncnorm.rvs(
                -sigma0_init/jitter, np.inf, loc=sigma0_init,
                scale=jitter, size=1, random_state=rng
            )[0])
            if isinstance(args["RV"], str) and args["RV"] == "normal":
                param_init["mu_R"] = jnp.array(params.get("MUR", params.get("RV", 3.)))
                param_init["sigma_R"] = jnp.array(params.get("SIGMAR", 0.5))
            elif isinstance(args["RV"], str) and args["RV"] == "uniform":
                param_init["uniform_RV_min"] = jnp.array(params.get("uniform_RV_min", 1))
                param_init["uniform_RV_max"] = jnp.array(params.get("uniform_RV_max", 6))
            else:
                param_init["RV"] = jnp.array(params.get("RV", params.get("MUR", 3.)))
            # param_init["tauA_tform"] = jnp.arctan(tauA_ / 1.)
            # param_init["sigma0_tform"] = jnp.arctan(sigma0_ / 0.1)

        param_init["Ds_tform"] = jnp.array(rng.normal(0, 1, N_sn))
        if args["fix_theta"] is None:
            param_init["theta"] = jnp.array(rng.normal(0, 1, N_sn))
        if args["fix_AV"] is None:
            # different from main, used tauA_ = param_init["tauA"], jittered tauA_init
            param_init["AV"] = jnp.array(rng.exponential(tauA_init, N_sn))

        if not args["fix_eps"]:
            # sigmaepsilon_init = 0.1 * np.ones(N_eps)
            # param_init["eps"] = rng.normal(0, 1, (N_sn, N_eps))
            L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon_init), jnp.eye(N_eps))
            # transposed to have N_sn be first axis for plate structure.
            param_init["eps_tform"] = jnp.matmul(np.linalg.inv(L_Sigma), rng.normal(0, 1, (N_eps, N_sn))).T

        return param_init

    def run(self, args: dict) -> None:
        """
        Main method to run BayeSN. The input yaml file allows for customisation of the
        sampling configuration and model via keyword arguments.

        Parameters
        ----------
        args
            Arguments from input yaml file before command line overrides,
            defines model wavelength range and data set to load.
        """
        args = self.parse_args(args)
        self.process_dataset(args)

        mode = args["mode"]
        fitting_mode = (
            mode.startswith("fit") or
            (
                mode == "custom"
                and not args["train_new_model"]
                and not args["infer_dust_properties"]
            )
        )
        # Set up initialisation for HMC chains
        # -------------------------
        if args["initialisation"] == "median":
            init_strategy = init_to_median()
        elif args["initialisation"] == "sample":
            init_strategy = init_to_sample()
        else:
            if not args["initialisation"].endswith("_model"):
                init_model = args["initialisation"] + "_model"
            init_strategy = init_to_value(
                values=self.initial_guess(
                    args, reference_model=init_model,
                    single_sn=fitting_mode  # need to init as if N_sn=1 for vmapping
                )
            )

        regularize_mass_matrix = fitting_mode
        step_size = 0.1 + 0.9*fitting_mode
        nuts_kernel = NUTS(
            self._model,
            init_strategy=init_strategy,
            regularize_mass_matrix=regularize_mass_matrix,
            step_size=step_size,
        )
        print(f"Preprocessing time: {time.time() - self.start_time:.2f} seconds")
        # z_hel and mjd chosen as representatives.
        print(f"self.data SN-arrays shape: {self.data.z_hel.shape} dtype: {self.data.z_hel.dtype}")
        print(f"self.data phot-arrays shape: {self.data.mjd.shape} dtype: {self.data.mjd.dtype}")
        print(f"self.data size: {sum(arr.nbytes for arr in self.data)/ 1024**2:.1f} MiB")
        print(f"self.band_weights shape: {self.band_weights.shape} dtype: {self.band_weights.dtype} "
            f"size: {self.band_weights.nbytes / 1024**2:.1f} MiB")
        print(f"Current mode: {args['mode']}")
        print("Running...")

        weights = self.band_weights
        start = timeit.default_timer()
        if (
            fitting_mode and args["fit_method"] == "mcmc"
        ):  # Use vmap to vectorise over individual fitting jobs
            samples = self._run_fit_mcmc(args, nuts_kernel)
        elif fitting_mode and args["fit_method"] == "vi":
            samples = self._run_fit_vi(args, init_strategy)
        else:
            samples = self._run_other(args, nuts_kernel)
        end = timeit.default_timer()
        print(f"Total inference runtime: {end - start:.2f} seconds")
        self.postprocess(samples, args)

    def _run_fit_mcmc(self, args, nuts_kernel):
        def fit_vmap_mcmc(data: ArrayLike, weights: ArrayLike, z_icdf) -> dict:
            """
            Short function-in-a-function just to allow you to do a vectorised map over multiple objects on a single
            device

            Parameters
            ----------
            obs: ArrayLike
                Data to fit, from output of process_dataset
            weights: ArrayLike
                Band-weights to calculate photometry

            Returns
            -------

            sample_dict: dict
                Samples and other information from MCMC fit

            """
            rng_key = PRNGKey(0)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=args["num_samples"],
                num_warmup=args["num_warmup"],
                num_chains=args["num_chains"],
                chain_method=args["chain_method"],
                progress_bar=args["chain_method"] == "vectorized",
            )
            if args["photoz"] and self.z_icdf_grid is not None:
                args["z_icdf"] = z_icdf
            # Vmapping over SN axis, need to pad back to expected shapes.
            padded_data = ObsData(*[arr[..., None] for arr in data])
            mcmc.run(rng_key, padded_data, weights[None, ...],  **args)
            return {
                **mcmc.get_samples(group_by_chain=True),
                **mcmc.get_extra_fields(group_by_chain=True),
            }

        vmap = jax.vmap(fit_vmap_mcmc, in_axes=(-1, 0, 0))
        n_sne = self.data.z_hel.shape[0]
        if args["photoz"] and self.z_icdf_grid is not None:
            z_icdf_all = np.asarray(self.z_icdf_grid)
        else:
            z_icdf_all = np.zeros((n_sne, 1))
        samples = vmap(self.data, self.band_weights, z_icdf_all)
        for key, val in samples.items():
            val = np.asarray(val)
            # drop the size-1 SNe-plate dim from the event axes (>=3), keeping n_sne/chains/draws (0/1/2)
            squeeze_axes = tuple(ax for ax in range(3, val.ndim) if val.shape[ax] == 1)
            if squeeze_axes:
                val = np.squeeze(val, axis=squeeze_axes)
            # vmap adds n_sne as axis 0; move it last to the (chains, draws, [event], n_sne) layout
            samples[key] = np.moveaxis(val, 0, -1)
        return samples

    def _run_fit_vi(self, args, init_strategy):
        try:
            RV = float(args["RV"])
        except ValueError:
            RV = self.RV or self.mu_R
        noeps_kwargs, vi_kwargs = [copy.deepcopy(args) for _ in range(2)]
        noeps_kwargs.update({"fix_eps": True, "rv_type": "global", "RV": RV})
        vi_kwargs.update({
            "fix_eps": False,
            "AV_dist": zltn.My_Exponential,
            "muhat_err": 5,
            "prior_only": False,
            "rv_type": "global",
            "RV": RV,
        })
        if args['laplace_method'] == 'lm':
            from numpyro.infer.util import initialize_model
            self._lm_model_info = initialize_model(
                PRNGKey(0), self._model,
                init_strategy=init_strategy, dynamic_args=True,
                model_args=(self.get_sn_slice_of_data(slice(0,1)), self.band_weights[0:1, ...]),
                model_kwargs=noeps_kwargs,
            )
            self._vi_model_info = initialize_model(
                PRNGKey(0), self._model,
                init_strategy=init_strategy, dynamic_args=True,
                model_args=(self.get_sn_slice_of_data(slice(0,1)), self.band_weights[0:1, ...]),
                model_kwargs=vi_kwargs,
            )

        def fit_vmap_vi(data: ArrayLike, weights: ArrayLike, z_icdf) -> dict:
            """
            Short function-in-a-function just to allow you to do a vectorised map over multiple objects on a single
            device

            Parameters
            ----------
            obs: ArrayLike
                Data to fit, from output of process_dataset
            weights: ArrayLike
                Band-weights to calculate photometry

            Returns
            -------

            sample_dict: dict
                Samples and other information from MCMC fit

            """
            if args["photoz"]:
                z_loc = "u" if self.z_icdf_grid is not None else "ztform"
                sample_locs = ["AV", "theta", "tmax", z_loc, "eps_tform", "Ds_tform"]
                # per-SN host photo-z quantiles threaded through the vmap (empty for the Gaussian case)
                z_kwargs = {"z_icdf": z_icdf} if self.z_icdf_grid is not None else {}
                # z-latent starts at unconstrained 0 (Normal mean / Uniform prior midpoint)
                extra_template = {z_loc: jnp.array([0.0])}
            else:
                sample_locs = ["AV", "theta", "tmax", "eps_tform", "Ds_tform"]
                z_kwargs = {}
                extra_template = {}

            # Vmapping over SN axis, need to pad back to expected shapes.
            padded_data = ObsData(*[arr[..., None] for arr in data])

            warm_scale_tril = None
            if args['laplace_method'] == 'lm':
                model_args = (padded_data, weights[None, ...])
                # Stage 1: Gauss-Newton LM MAP for (AV, theta, tmax, [redshift], Ds)
                # under the Exponential prior.
                mi = self._lm_model_info
                # pot_fn_noeps = mi.potential_fn(padded_data, weights[None, ...], **z_kwargs)
                # post_fn_noeps = mi.postprocess_fn(padded_data, weights[None, ...], **z_kwargs)
                mi_kwargs = copy.deepcopy(noeps_kwargs)
                mi_kwargs.update(z_kwargs)
                pot_fn_noeps = mi.potential_fn(padded_data, weights[None, ...], **mi_kwargs)
                post_fn_noeps = mi.postprocess_fn(padded_data, weights[None, ...], **mi_kwargs)
                # predict_fn_noeps = lambda z: _predict(noeps_model, model_args, {}, z)
                # prior_pot_fn_noeps = lambda z: _prior_pot(noeps_model, model_args, {}, z)
                predict_fn_noeps = lambda z: _predict(self._model, model_args, mi_kwargs, z)
                prior_pot_fn_noeps = lambda z: _prior_pot(self._model, model_args, mi_kwargs, z)
                # Per-SN init: prior medians for AV/theta/tmax, this SN's muhat for Ds.
                z_template_s1 = {
                    'AV': jnp.array([jnp.log(self.tauA * jnp.log(2.0))]),
                    # 'Ds': padded_data.muhat[0:1],
                    'Ds_tform': jnp.array([0.0]),
                    'theta': jnp.array([0.0]),
                    'tmax': jnp.array([0.0]),
                    **extra_template,
                }
                noeps_median, _, z_unc_noeps = run_lm_laplace_gn(
                    predict_fn_noeps, prior_pot_fn_noeps, post_fn_noeps, z_template_s1,
                    maxiter=args['lm_maxiter'],
                    lam_init=args['lm_lam_init'],
                    use_linesearch=args['lm_use_linesearch'],
                )
                # Stage 2: Gauss-Newton LM on the full VI model, warm-started from the Stage 1 MAP
                vi_mi = self._vi_model_info
                vi_mi_kwargs = copy.deepcopy(vi_kwargs)
                vi_mi_kwargs.update(z_kwargs)
                post_fn_vi = vi_mi.postprocess_fn(*model_args, **vi_mi_kwargs)
                predict_fn = lambda z: _predict(self._model, model_args, vi_mi_kwargs, z)
                prior_pot_fn = lambda z: _prior_pot(self._model, model_args, vi_mi_kwargs, z)
                z_start_vi = {**vi_mi.param_info.z, **z_unc_noeps,
                              "AV": noeps_median["AV"]}
                z_start_vi["eps_tform"] = jnp.zeros_like(z_start_vi["eps_tform"])
                if args["stage2_tmax_prior_std"] is not None:
                    tmax_anchor = z_unc_noeps["tmax"]
                    tmax_var = args["stage2_tmax_prior_std"] ** 2
                    def prior_pot_anchored(z):
                        delta = z["tmax"] - tmax_anchor
                        return prior_pot_fn(z) + 0.5 * jnp.sum(delta * delta) / tmax_var
                else:
                    prior_pot_anchored = prior_pot_fn
                laplace_median, _, z_unc_vi = run_lm_laplace_gn(
                    predict_fn, prior_pot_anchored, post_fn_vi, z_start_vi,
                    maxiter=args["lm_maxiter"],
                    lam_init=args["lm_lam_init"],
                    use_linesearch=args["lm_use_linesearch"],
                )
                warm_scale_tril = compute_gn_scale_tril(
                    predict_fn, prior_pot_anchored, z_unc_vi)
            else:
                optimizer = Adam(0.01)
                # laplace_guide = AutoLaplaceApproximation(noeps_model, init_loc_fn=init_strategy)
                laplace_guide = AutoLaplaceApproximation(self._model, init_loc_fn=init_strategy)
                svi = SVI(self._model, laplace_guide, optimizer, loss=Trace_ELBO(5))
                svi_kwargs = copy.deepcopy(noeps_kwargs)
                svi_kwargs.update(z_kwargs)
                svi_result = svi.run(PRNGKey(123), 15000, padded_data, weights[None, ...], progress_bar=False, **svi_kwargs)
                params, losses = svi_result.params, svi_result.losses
                laplace_median = laplace_guide.median(params)

            # Initialise the ZLTN guide loc from the Laplace MAP.
            new_init_dict = {
                k: jnp.array([laplace_median[k][0]])
                for k in sample_locs
                if k in laplace_median
            }
            if "eps_tform" not in new_init_dict:
                new_init_dict["eps_tform"] = jnp.zeros(
                    (1, (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0])
                )
            zltn_guide = zltn.AutoMultiZLTNGuide(
                self._model,
                init_loc_fn=init_to_value(values=new_init_dict),
                init_scale_tril=warm_scale_tril
            )
            if args['zltn_lr_final'] == args['zltn_lr']:
                step_size = args['zltn_lr']
            else:
                decay_base = (args['zltn_lr_final'] / args['zltn_lr']) ** (1.0 / args['num_zltn_iter'])
                step_size = lambda t: args['zltn_lr'] * decay_base ** t
            svi = SVI(self._model, zltn_guide, Adam(step_size), Trace_ELBO(args["zltn_particles"]))
            svi_kwargs = copy.deepcopy(vi_kwargs)
            svi_kwargs.update(z_kwargs)
            svi_result = svi.run(PRNGKey(123), args["num_zltn_iter"], padded_data, weights[None, ...], progress_bar=False, **svi_kwargs)
            params, losses = svi_result.params, svi_result.losses
            predictive = Predictive(
                zltn_guide, params=params, num_samples=4 * args["num_samples"]
            )
            samples = predictive(PRNGKey(123), data=None)
            if args["photoz"]: # surface z (a deterministic, so not in the guide samples)
                if self.z_icdf_grid is not None:
                    samples['z'] = jnp.interp(samples['u'], self.z_u_grid, z_icdf)
                else:
                    samples['z'] = data.z_hel + data.z_hel_err * samples['ztform']
            samples['eps'] = jnp.matmul(self.L_Sigma[None, ...], samples['eps_tform'].transpose(0, 2, 1))
            muhat_err = args.get("muhat_err", 5)
            if muhat_err is None:
                muhat_err = (
                    5
                    / (data.z_hel * jnp.log(10))
                    * jnp.sqrt(data.z_hel_err * data.z_hel_err + self.sigma_pec * self.sigma_pec)
                )
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            fix_dist = data.z_hel < args.get("fix_dist_limit", 0.08)
            Ds_err = Ds_err * fix_dist + args.get("fix_dist_Ds_err", 5) * (1 - fix_dist)
            samples['Ds'] = data.muhat + samples['Ds_tform'] * Ds_err
            # samples['losses'] = losses
            return {**samples}

        batched_map = jax.vmap(fit_vmap_vi, in_axes=(-1, 0, 0))
        n_sne = self.data.z_hel.shape[0]  # not using dataset since fit method does not
        # per-SN host photo-z quantiles threaded through the vmap (dummy zeros otherwise)
        if args['photoz'] and self.z_icdf_grid is not None:
            z_icdf_all = np.asarray(self.z_icdf_grid)
        else:
            z_icdf_all = np.zeros((n_sne, 1))
        batch_size = args["batch_size"] if args.get("batch_size") else n_sne
        n_batches = (n_sne + batch_size - 1) // batch_size

        chunks = []
        for b in tqdm(range(n_batches), desc='VI batches', disable=n_batches == 1):
            lo, hi = b * batch_size, min((b + 1) * batch_size, n_sne)
            n_real = hi - lo
            if n_real == batch_size:
                batch_data = self.get_sn_slice_of_data(slice(lo, hi))
                batch_weights = self.band_weights[lo:hi]
                batch_zicdf = z_icdf_all[lo:hi]
            else:
                # Pad final batch by replicating SN 0; padded outputs discarded.
                real_batch_data = self.get_sn_slice_of_data(slice(lo, hi))
                dupe_batch_data = self.get_sn_slice_of_data(slice(0, 1))
                n_dupe = batch_size - n_real
                batch_data = ObsData(*[
                    np.append(real, np.repeat(dupe, n_dupe, axis=1), axis=1)
                    if dupe.ndim == 2 else np.append(real, np.repeat(dupe, n_dupe))
                    for real, dupe in zip(real_batch_data, dupe_batch_data)
                ])
                batch_weights = np.empty(
                    (batch_size, *self.band_weights.shape[1:]), dtype=self.band_weights.dtype
                )
                batch_weights[:n_real] = self.band_weights[lo:hi]
                batch_weights[n_real:] = self.band_weights[0:1]
                batch_zicdf = np.empty((batch_size, z_icdf_all.shape[1]), dtype=z_icdf_all.dtype)
                batch_zicdf[:n_real] = z_icdf_all[lo:hi]
                batch_zicdf[n_real:] = z_icdf_all[0:1]
            chunk = batched_map(batch_data, batch_weights, batch_zicdf)
            chunks.append({k: np.asarray(v)[:n_real] for k, v in chunk.items()})

        samples = {k: np.concatenate([c[k] for c in chunks], axis=0)
                   for k in chunks[0]}
        del samples["_auto_latent"]
        expand_dim = False
        for key, val in samples.items():
            val = np.squeeze(val)
            if val.ndim == 1:  # In case fitting only one object
                expand_dim = True
            if expand_dim:
                val = val[None, ...]
            if val.ndim == 3:
                samples[key] = val.transpose(1, 2, 0)
            else:
                samples[key] = val.transpose()
            samples[key] = samples[key].reshape(
                4, args["num_samples"], *samples[key].shape[1:]
            )
        return samples

    def _run_other(self, args, nuts_kernel):
        mcmc = MCMC(
            nuts_kernel,
            num_samples=args["num_samples"],
            num_warmup=args["num_warmup"],
            num_chains=args["num_chains"],
            chain_method=args["chain_method"],
            progress_bar=True,
        )
        rng = PRNGKey(0)
        mcmc.run(
            rng, self.data, self.band_weights, **args, extra_fields=("potential_energy",),
        )
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        return samples

    def fit_from_file(
        self,
        path: str | Path,
        filt_map: dict = {},
        peakmjd_key: str = "SEARCH_PEAKMJD",
        print_summary: bool = True,
        file_prefix: str | None = None,
        drop_bands: list = [],
        mag: bool = False,
        photoz: bool = False,
        z_prior_err: float | None = None,
        z_pdf: Any = None,
        z_quantiles: Any = None,
        chain_method: str = "parallel",
        ext_rel: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, tuple]:
        """
        Fit a light curve contained in an SNANA-format text file using the BayeSN model.

        Parameters
        ----------
        path
            Path to SNANA-format text file containing data to be fit.
        filt_map
            Dictionary providing mapping between filter names in file and BayeSN filters.
        peakmjd_key
            Key to be used for peak MJD in SNANA text file metadata.
        print_summary
            Specifies whether to print fit summary.
        file_prefix
            Prefix of name for output files containing summary table and MCMC samples.
            If None, output files will not be saved and only returned.
        drop_bands
            List of bands to be ignored during fitting.
        mag
            Specifies whether data is mag or flux.
        photoz
            Whether to treat redshift as a sampled parameter.
        z_prior_err
            Prior uncertainty on redshift.
        z_pdf
            Photo-z prior PDF object.
        z_quantiles
            Precomputed photo-z quantiles.
        chain_method
            Method for parallel MCMC chain execution.
        ext_rel
            Extinction relation name to use.
        **kwargs
            Additional keyword arguments passed to ``fit``.

        Returns
        -------
        samples : dict
            Dictionary containing parameter names as keys and MCMC samples as values.
        sn_props
            2-tuple containing SN redshift and MW E(B-V).
        """
        meta, lcdata = read_snana_ascii(path)

        t = lcdata.MJD.values
        flux = lcdata.flux.values
        flux_err = lcdata.flux_err.values
        filters = lcdata.FLT.values
        peak_mjd = meta[peakmjd_key]
        z = meta["REDSHIFT_HELIO"]
        ebv_mw = meta["MWEBV"]
        zpt = meta.get("ZP_FLUXCAL", self.ZPT)  # header ZP overrides the configured default
        if zpt != self.ZPT:
            warn(UserWarning(
                f"Overwriting the current ZPT ({self.ZPT}) with zpt={zpt}."
            ))
            self.ZPT = zpt
        if z_prior_err is None:
            z_prior_err = meta.get("REDSHIFT_HELIO_ERR", 0.)

        return self.fit(
            t,
            flux,
            flux_err,
            filters,
            z,
            ebv_mw=ebv_mw,
            peak_mjd=peak_mjd,
            filt_map=filt_map,
            print_summary=print_summary,
            file_prefix=file_prefix,
            drop_bands=drop_bands,
            mag=mag,
            photoz=photoz,
            z_prior_err=z_prior_err,
            z_pdf=z_pdf,
            z_quantiles=z_quantiles,
            chain_method=chain_method,
            ext_rel=ext_rel,
            zpt=zpt,
            **kwargs,
        )

    def fit(
        self,
        t: ArrayLike,
        flux: ArrayLike,
        flux_err: ArrayLike,
        filters: ArrayLike,
        z: float,
        ext_rel: str | None = None,
        host_logmass: Number = -9,
        ebv_mw: float = 0,
        peak_mjd: float | None = None,
        filt_map: dict = {},
        print_summary: bool = True,
        file_prefix: str | None = None,
        drop_bands: list = [],
        mag: bool = False,
        photoz: bool = False,
        z_prior_err: float | None = None,
        z_pdf: Any = None,
        z_quantiles: Any = None,
        verbose: bool = False,
        num_samples: int = 250,
        num_warmup: int = 250,
        num_chains: int = 4,
        chain_method: str = "parallel",
        zpt: Number | None = None,
        RNG_seed: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict, tuple]:
        """
        Fit light curve data loaded into memory with the BayeSN model.

        Parameters
        ----------
        t : (N_obs,)
            MJDs or rest-frame phases for light curve data to be fit.
        flux : (N_obs,)
            Fluxes (or magnitudes if ``mag=True``) for light curve data.
        flux_err : (N_obs,)
            Flux or magnitude errors for light curve data.
        filters : (N_obs,)
            Filter name corresponding to each observation.
        z
            Heliocentric redshift of the SN.
        ext_rel
            Extinction relation name to use.
        host_logmass
            Host galaxy log stellar mass.
        ebv_mw
            Milky Way E(B-V) extinction value.
        peak_mjd
            Fiducial value for maximum MJD of SN, used to convert phases to rest-frame.
        filt_map
            Dictionary mapping filter names in data to BayeSN filters.
        print_summary
            Specifies whether to print fit summary.
        file_prefix
            Prefix of name for output files containing summary table and MCMC samples.
        drop_bands
            List of bands to be ignored during fitting.
        mag
            Whether input data are in magnitudes rather than fluxes.
        photoz
            Whether to sample the redshift.
        z_prior_err
            Uncertainty on the redshift prior.
        z_pdf
            Photo-z prior PDF object.
        z_quantiles
            Precomputed photo-z quantiles.
        verbose
            Prints updates to stdout if True.
        num_samples
            Number of MCMC samples to draw per chain.
        num_warmup
            Number of warmup steps per chain.
        num_chains
            Number of parallel MCMC chains.
        chain_method
            Method for executing chains in parallel.
        zpt
            Instrumental zero-point override.
        RNG_seed
            Random number generator seed.
        **kwargs
            Additional arguments passed to ``_model``.

        Returns
        -------
        samples : dict
            Dictionary containing parameter names as keys and MCMC samples as values.
        sn_props
            2-tuple containing SN redshift and MW E(B-V).
        """
        if zpt is not None and self.ZPT != zpt:
            warn(UserWarning(
                f"Overwriting the current ZPT ({self.ZPT}) with zpt={zpt}."
            ))
            self.ZPT = zpt
        if ext_rel is not None and self.ext_rel.name != ext_rel:
            self.load_ext_rel(ext_rel_name=ext_rel)
        if isinstance(drop_bands, str):
            drop_bands = [drop_bands]
        t, flux, flux_err, filters = (
            np.array(t),
            np.array(flux),
            np.array(flux_err),
            np.array(filters),
        )
        if not (len(t) == len(flux) == len(flux_err) == len(filters)):
            raise ValueError("t, flux, flux_err, and filters must all be the same length.")
        if mag:  # Convert data from mag into FLUXCAL
            flux = np.power(10, (self.ZPT - flux) / 2.5)
            flux_err = (np.log(10) / 2.5) * flux * flux_err
        if peak_mjd is not None:
            t = (t - peak_mjd) / (1 + z)

        self.photoz = kwargs["photoz"] = photoz
        self.z_icdf_grid = None
        if photoz and z_quantiles is not None:  # (probs, vals), or bare z-values with even 0..1 levels
            zq = z_quantiles
            probs, vals = zq if (len(zq) == 2 and np.ndim(zq[0]) > 0) else (np.linspace(0., 1., len(zq)), zq)
            self.z_u_grid = jnp.asarray(probs)
            self.z_icdf_grid = jnp.atleast_2d(jnp.asarray(vals))
        elif photoz and z_pdf is not None:
            self.z_u_grid = jnp.linspace(0., 1., 101)
            self.z_icdf_grid = jnp.atleast_2d(jnp.asarray(z_pdf.icdf(self.z_u_grid)))
        if photoz:
            # Loose cut: drop epochs pre-explosion across the +/-3 sigma prior z and tmax range
            if self.z_icdf_grid is not None:
                z_lo, z_hi = float(self.z_icdf_grid[0, 0]), float(self.z_icdf_grid[0, -1])
            elif z_prior_err is not None:
                z_lo, z_hi = z - 3 * z_prior_err, z + 3 * z_prior_err
            else:
                raise ValueError(
                    "photoz is True but z_quantiles, z_pdf, and z_prior_err are all "
                    "None, so it is not clear how redshift should be sampled."
                )
            p1, p2 = t * (1 + z) / (1 + z_lo), t * (1 + z) / (1 + z_hi)
            keep = np.maximum(p1, p2) + 10 > float(self.hsiao_t[0])
        else:
            keep = (t > self.tau_knots.min()) & (t < self.tau_knots.max())
            z_prior_err = 0
        flux, flux_err, filters, t = flux[keep], flux_err[keep], filters[keep], t[keep]

        if verbose and any(~keep):
            print(
                f"Cutting {len(t) - sum(keep)} data due to rest-frame phases "
                "beyond model limits."
            )

        # Prepare filters
        filters = np.array([filt_map.get(filt, filt) for filt in filters])
        self._set_used_bands(filters)
        for f in np.unique(filters):
            if f not in self.filter_dict["filters"]:
                raise ValueError(
                    f"""
                    Filter "{f}" not defined in BayeSN, either add a mapping to
                    filt_map to ensure that your filter names match up with ones
                    built-in or add some custom filters if you want to use your own.
                    """
                )
            if photoz:
                if self.band_lim_dict[f][1] / (1 + z_lo) < self.min_wave or self.band_lim_dict[f][0] / (1 + z_hi) > self.max_wave:
                    if verbose:
                        print(
                            f"Dropping filter {f} due to rest-frame transmission "
                            "beyond model definition."
                        )
                    drop_bands.append(f)
            elif z > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or z < (
                self.band_lim_dict[f][1] / self.l_knots[-1] - 1
            ):
                if verbose:
                    print(
                        f"Dropping filter {f} due to rest-frame transmission "
                        "beyond model definition."
                    )
                drop_bands.append(f)
        for f in drop_bands:
            inds = filters != f
            t = t[inds]
            flux = flux[inds]
            flux_err = flux_err[inds]
            filters = filters[inds]
        self._set_used_bands(filters)  # dropping drop_bands
        band_indices = np.array(
            [self.used_band_dict[self.band_dict[filt]] for filt in filters]
        )

        n_data = len(t)
        if n_data == 0:
            raise ValueError(
                "No data in rest-frame phase range covered by model, maybe you gave "
                "the wrong peak MJD?"
            )
        # Set up and populate data array
        self.data = ObsData(
            host_logmass=jnp.full(1, host_logmass),
            z_hel=jnp.full(1, z),
            z_hel_err=jnp.full(1, z_prior_err),
            muhat=jnp.full(1, self.cosmo.distmod(z).value),
            MWEBV=jnp.full(1, ebv_mw),
            mjd=jnp.array(t[:, None]),
            flux=jnp.array(flux[:, None]),
            flux_err=jnp.array(flux_err[:, None]),
            band_indices=jnp.array(band_indices[:, None]),
            mask=jnp.ones_like(t)[:, None]
        )

        self.band_weights = self._calculate_band_weights(
            self.data.z_hel,
            self.data.MWEBV,
            lam_shifts=0
        )
        self.J_t = self.get_J_t(self.data.mjd)
        self.hsiao_interp = self.get_hsiao_interp(self.data.mjd)

        # Required for parse_args to not throw an error.
        kwargs.update({"data_root": ".", "data_table": None})
        kwargs = self.parse_args(kwargs, verbose=verbose)
        kwargs["muhat_err"] = 5

        nuts_kernel = NUTS(
            self._model,
            init_strategy=init_to_median(),
        )
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
        )
        rng = PRNGKey(0)

        mcmc.run(
            rng,
            self.data,
            self.band_weights,
            **kwargs,
            extra_fields=("potential_energy",),
        )
        if print_summary:
            mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        if kwargs["fix_tmax"]:
            samples["tmax"] = jnp.zeros((num_chains, num_samples, 1))
        if kwargs["fix_eps"]:
            samples["eps_tform"] = jnp.zeros((num_chains, num_samples, self.N_knots_sig, 1))
        if kwargs["fix_theta"] is not None:
            samples["theta"] = jnp.full((num_chains, num_samples, 1), kwargs["fix_theta"])
        if kwargs["fix_AV"] is not None:
            samples["AV"] = jnp.full((num_chains, num_samples, 1), kwargs["fix_AV"])
        if isinstance(kwargs["RV"], Number | Array) and getattr(kwargs["RV"], "ndim", 0) == 0:
            samples["RV"] = jnp.full((num_chains, num_samples, 1), kwargs["RV"])

        if peak_mjd is not None:
            samples["peak_MJD"] = peak_mjd + samples["tmax"] * (1 + (samples["z"] if photoz else z))
        if not photoz:
            # muhat-shrinkage ties Ds to distmod(z); skip for cosmology-independent photo-z
            muhat = self.cosmo.distmod(z).value
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            samples["mu"] = np.random.default_rng(RNG_seed).normal(
                loc=(
                    samples["Ds"] * np.power(muhat_err, 2)
                    + muhat * np.power(self.sigma0, 2)
                ) / np.power(Ds_err, 2),
                scale=np.sqrt(
                    (
                        np.power(self.sigma0, 2) * np.power(muhat_err, 2)
                    ) / np.power(Ds_err, 2)
                )
            )
            samples["delM"] = samples["Ds"] - samples["mu"]

        if file_prefix is not None:
            summary = arviz.summary(samples)
            summary.to_csv(f"{file_prefix}_fit_summary.csv")
            with open(f"{file_prefix}_chains.pkl", "wb") as file:
                pickle.dump(samples, file)

        sn_props = (z, ebv_mw)
        return samples, sn_props

    def postprocess(
        self,
        samples: dict,
        args: dict,
        l_knot_1: Number = 6200.0,
        tau_knot_0: Number = 0.0,
        tau_knot_1: Number = 10.0,
        RNG_seed: int | None = None,
    ) -> None:
        """
        Function to postprocess BayeSN output. Applies transformations to some
        parameters e.g. ensuring consistency for W1 and theta, as flipping the sign in
        front of W1 and theta will lead to an identical result. Saves output chains
        and calculates a fit summary.

        Parameters
        ----------
        samples
            Output of MCMC, dictionary containing posterior samples for each parameter
            with parameter names as keys.
        args
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        l_knot_1
            Wavelength knot used for sign degeneracy fixing.
        tau_knot_0
            First phase knot used for sign degeneracy fixing.
        tau_knot_1
            Second phase knot used for sign degeneracy fixing.
        RNG_seed
            Random seed for postprocessing computations.
        """
        rng = np.random.default_rng(RNG_seed)
        start = time.time()
        if args["fix_tmax"]:
            samples["tmax"] = np.zeros_like(samples["Ds"])
        if args["fix_eps"]:
            samples["eps_tform"] = np.zeros((*samples["Ds"].shape[:2], self.N_knots_sig, samples["Ds"].shape[2]))
        if args["fix_theta"] is not None:
            samples["theta"] = np.full_like(samples["Ds"], args["fix_theta"])
        if args["fix_AV"] is not None:
            samples["AV"] = np.full_like(samples["Ds"], args["fix_AV"])
        if isinstance(args["RV"], Number | Array) and getattr(args["RV"], "ndim", 0) == 0:
            samples["RV"] = np.full_like(samples["Ds"], float(args["RV"]))
        if args["train_new_model"]:
            # Breaking the degeneracy will change the samples, so save a copy first.
            with open(args["outputdir"] / "initial_chains.pkl", "wb") as file:
                pickle.dump(samples, file)
            samples["theta"], samples["W1"] = self._fix_W1_theta_sign_degen(
                theta=samples["theta"],
                W1=samples["W1"],
                l_knot_1=l_knot_1,
                tau_knot_0=tau_knot_0,
                tau_knot_1=tau_knot_1
            )
            self._make_yaml_from_samples(samples=samples, args=args)

        fitting_mode = args["mode"].startswith("fit") or (not args["train_new_model"] and not args["infer_dust_properties"])

        if fitting_mode:
            muhat_err = 5
            Ds_err = np.hypot(muhat_err, self.sigma0)
            if args["photoz"]:
                # Cosmology-independent: report the fitted light-curve distance Ds directly,
                # without the muhat (cosmo(z_hubble) distmod) shrinkage that would inject a fiducial cosmology
                samples["mu"] = samples["Ds"]
                samples["delM"] = np.zeros_like(samples["Ds"])
            else:
                samples["mu"] = rng.normal(
                    loc=(
                        samples["Ds"] * np.power(muhat_err, 2)
                        + self.data.muhat * np.power(self.sigma0, 2)
                    ) / np.power(Ds_err, 2),
                    scale=self.sigma0 * muhat_err / Ds_err
                )
                samples["delM"] = samples["Ds"] - samples["mu"]
            if not args["fix_tmax"]:  # Convert tmax samples into peak_MJD samples
                # Time dilation at the fitted z for photo-z, else the fixed catalog z
                z_dilation = samples["z"] if args["photoz"] else self.data.z_hel[None, None, :]
                samples["peak_MJD"] = self.dataset.peak_mjd[None, None, :] + samples["tmax"] * (1 + z_dilation)

            # Compute FITPROB (must be before LCPLOT generation which corrupts self.band_weights)
            fitprob, fitchi2, ndof = self.get_fitprob(samples, batch_size=args.get("batch_size"))

            # Add flux from chains to lcplot_data and create lcplot file
            # Real data will have DATA_FLAG=1 and model flux 0
            t = np.arange(self.tau_knots[0], self.tau_knots[-1], 2)
            if args["num_lcplot"] is None:
                num_lcplot = self.dataset.N_sn
            else:
                num_lcplot = args["num_lcplot"]

            if num_lcplot > 0:
                bands_by_cid = self.lcplot_data.groupby("CID")["FLT"].unique().to_dict()
                bands = [list(bands_by_cid.get(sn, [])) for sn in self.dataset.snid]
                f = self.get_flux_from_chains(
                    t,
                    bands,
                    samples,
                    self.data.z_hel,
                    self.data.MWEBV,
                    num_samples=None,
                    num_sne=num_lcplot,
                    mag=False,
                    mean=not args["save_fit_errors"],
                )
                f, ferr = f.mean(axis=1), f.std(axis=1)

                self.lcplot_data["DATA_FLAG"] = 1
                z_hel = self.data.z_hel
                fit_dfs = []
                for i, sn in enumerate(self.lcplot_data.CID.unique()):
                    fit_df = pd.DataFrame()
                    fit_df["MJD"] = (self.dataset.peak_mjd[i] + t * (1 + z_hel[i])).repeat(
                        len(bands[i])
                    )
                    fit_df["FLUXCAL"] = f[i, : len(bands[i]), :].flatten(order="F")
                    fit_df["FLUXCALERR"] = ferr[i, : len(bands[i]), :].flatten(order="F")
                    fit_df["FLT"] = np.tile(bands[i], len(t))
                    fit_df["CID"] = sn
                    fit_df["DATA_FLAG"] = 0
                    fit_dfs.append(fit_df)
                self.lcplot_data = pd.concat([self.lcplot_data] + fit_dfs, ignore_index=True)

                self.lcplot_data = self.lcplot_data.sort_values(
                    by=["CID", "DATA_FLAG", "MJD"]
                )
                self.lcplot_data.to_csv(
                    args["outputdir"] / f"{args['outfile_prefix']}.LCPLOT",
                    index=False,
                )

            # Create FITRES file
            # if args["snana"]:
            # fetch snana version that includes tag + commit;
            # e.g., v11_05-4-gd033611.
            # Use same git command as in Makefile for C code
            SNANA_DIR = os.environ.get("SNANA_DIR", "NULL")
            if SNANA_DIR != "NULL":
                cmd = f"cd {SNANA_DIR}; git describe --always --tags"
                ret = subprocess.run(
                    [cmd], cwd=os.getcwd(), shell=True, capture_output=True, text=True
                )
                snana_version = ret.stdout.replace("\n", "")
            else:
                snana_version = "NULL"
            self.fitres_table.meta = {
                "#\n# SNANA_VERSION:": snana_version,
                "# VERSION_PHOTOMETRY:": args.get(
                    "version_photometry", args.get("data_table")
                ),
                "# TABLE NAME:": "FITRES\n#",
            }

            n_sn = samples["mu"].shape[-1]
            drop_keys = ["diverging", "_auto_latent"]
            for key in drop_keys:
                if key in samples:
                    del samples[key]
            if args["save_summary"]:
                summary = arviz.summary(samples)
                summary.to_csv(args["outputdir"] / f"{args['outfile_prefix']}.SUMMARY.TEXT")
                summary_subset = summary[~summary.index.str.contains("tform")]
                rhat = summary_subset.r_hat.values
                sn_rhat = np.array([rhat[i::n_sn] for i in range(n_sn)])
                self.fitres_table["MEANRHAT"] = sn_rhat.mean(axis=1)
                self.fitres_table["MAXRHAT"] = sn_rhat.max(axis=1)
            self.fitres_table["MU_LCFIT"] = samples["mu"].mean(axis=(0, 1))
            self.fitres_table["MUERR_LCFIT"] = samples["mu"].std(axis=(0, 1))
            for key in ("theta", "AV", "peak_MJD"):
                self.fitres_table[key.upper().replace("_", "")] = samples[key].mean(axis=(0, 1))
                self.fitres_table[key.upper().replace("_", "")+"ERR"] = samples[key].std(axis=(0, 1))
            if args["photoz"]:
                # fitted photo-z posterior (catalog zHEL/zHD columns keep the host prior)
                self.fitres_table['ZPHOT_FIT'] = samples['z'].mean(axis=(0, 1))
                self.fitres_table['ZPHOT_FITERR'] = samples['z'].std(axis=(0, 1))
            self.fitres_table['FITCHI2'] = np.array(fitchi2)
            self.fitres_table['NDOF'] = ndof
            self.fitres_table['FITPROB'] = fitprob
            # if not args["fit_method"] == "vi":
            self.fitres_table.round(4)

            drop_count = pd.isna(self.fitres_table["MU_LCFIT"]).sum()
            self.fitres_table = self.fitres_table[
                ~pd.isna(self.fitres_table["MU_LCFIT"])
            ]

            # Reorder to put SIM columns last
            new_cols = [
                col for col in self.fitres_table.columns if "SIM" not in col
            ] + [col for col in self.fitres_table.columns if "SIM" in col]
            self.fitres_table = self.fitres_table[new_cols]

            sncosmo.write_lc(
                data=self.fitres_table,
                fname=args["outputdir"] / f"{args['outfile_prefix']}.FITRES.TEXT",
                fmt="snana",
                metachar="",
            )
            if hasattr(self, "all_table"):
                sncosmo.write_lc(
                    data=self.all_table,
                    fname=args["outputdir"] / f"{args['outfile_prefix']}.LCSUMMARY.TEXT",
                    fmt="snana",
                    metachar="",
                )

        if args["snana"]:
            self.end_time = time.time()
            cpu_time = self.end_time - self.start_time
            # Output yaml
            out_dict = {
                "ABORT_IF_ZERO": 1,
                "SURVEY": getattr(self, "survey", "NULL"),
                "IDSURVEY": int(getattr(self, "survey_id", 0)),
                "NEVT_TOT": self.data.z_hel.shape[0],
                "NEVT_LC_CUTS": self.data.z_hel.shape[0],
                "NEVT_LCFIT_CUTS": int(self.data.z_hel.shape[0] - drop_count),
                "CPU_MINUTES": round(cpu_time / 60, 2),
            }
            with open(f"{args['outfile_prefix']}.YAML", "w") as file:
                yaml.dump(out_dict, file)

            if fitting_mode:
                return

        # Save convergence data for each parameter to csv file
        summary = arviz.summary(samples)
        summary.to_csv(args["outputdir"] / "fit_summary.csv")

        with open(args["outputdir"] / "chains.pkl", "wb") as file:
            pickle.dump(samples, file)

        dump_args = copy.deepcopy(args)
        for key, val in dump_args.items():
            if isinstance(val, Path):
                dump_args[key] = str(val)
            elif isinstance(val, Array) and getattr(val.ndim, 1) == 1:
                dump_args[key] = float(val)
            elif isinstance(val, Array | np.ndarray):
                dump_args[key] = val.tolist()
            elif val == dist.Exponential:
                dump_args[key] = "dist.Exponential"
            elif val == zltn.My_Exponential:
                dump_args[key] = "zltn.My_Exponential"
        with open(args["outputdir"] / "input.yaml", "w") as file:
            yaml.dump(dump_args, file)

    def _fix_W1_theta_sign_degen(
        self,
        theta: ArrayLike,
        W1: ArrayLike,
        l_knot_1: Number = 6200.,
        tau_knot_0: Number = 0.,
        tau_knot_1: Number = 10.,
    ) -> dict:
        """The model has multiple symmetries that will lead to unreliable parameter
        estimation if not corrected.

        The model is degenerate under the transformation (theta, W1) -> (-theta, -W1).
        We force the W1 surface evaluated at wavelength l_knot_1 to increase between
        the phases tau_knot_0 and tau_knot_1.

        The model is also degenerate under (theta, W1) -> (A*theta, W1/A) for constant A.
        (the previous degeneracy is a special case A=-1). Although theta is sampled from
        N(0, 1), we force std(theta) = 1 in the posterior to break this degeneracy.
        """
        J_R = spline_coeffs([float(l_knot_1)], self.l_knots, invKD(self.l_knots))
        J_0 = spline_coeffs([float(tau_knot_0)], self.tau_knots, invKD(self.tau_knots))
        J_10 = spline_coeffs([float(tau_knot_1)], self.tau_knots, invKD(self.tau_knots))
        W1_grid = np.reshape(
            W1,
            (W1.shape[0], W1.shape[1], self.l_knots.shape[0], self.tau_knots.shape[0],),
            order="F",
        )
        N_chains = W1_grid.shape[0]
        sign = np.zeros(N_chains)
        for chain in range(N_chains):
            chain_W1 = np.mean(W1_grid[chain, ...], axis=0)
            padded_chain_W1 = np.ones((chain_W1.shape[0]+1, chain_W1.shape[1]+1))
            padded_chain_W1[1:,1:] = chain_W1
            chain_sign = np.sign(
                np.squeeze(np.matmul(J_R, np.matmul(padded_chain_W1, J_10.T)))
                - np.squeeze(np.matmul(J_R, np.matmul(padded_chain_W1, J_0.T)))
            )
            sign[chain] = chain_sign
        theta = theta * sign[:, None, None]
        W1 = W1 * sign[:, None, None]
        # Modify W1 and theta----------------
        theta_std = np.std(theta, axis=2)
        theta = theta / theta_std[..., None]
        W1 = W1 * theta_std[..., None]
        return theta, W1

    def _make_yaml_from_samples(
        self,
        samples: dict,
        args: dict,
        outfile: str | Path = "bayesn.yaml"
    ):
        """Save best fit global params to files for easy inspection and reading in.
        """
        W_shape = (self.l_knots.shape[0], self.tau_knots.shape[0])
        W0 = np.mean(samples["W0"], axis=[0, 1]).reshape(W_shape, order="F")
        W1 = np.mean(samples["W1"], axis=[0, 1]).reshape(W_shape, order="F")
        L_Sigma = np.matmul(
            np.diag(np.mean(samples["sigmaepsilon"], axis=[0, 1])),
            np.mean(samples["L_Omega"], axis=[0, 1]),
        )
        yaml_data = {
            "M0": float(self.M0),
            "TAU_KNOTS": self.tau_knots.tolist(),
            "L_KNOTS": self.l_knots.tolist(),
            "L_SIGMA_EPSILON": L_Sigma.tolist(),
            "W1": W1.tolist(),
        }
        if args["split_variant"] == "split_sed":
            for suffix in ("_HM", "_LM"):
                delW = np.mean(
                    samples[f"delW{suffix}"], axis=[0, 1]
                ).reshape(W_shape, order="F")
                yaml_data[f"W0{suffix}"] = (W0 + delW).tolist()
        else:
            yaml_data["W0"] = W0.tolist()
        for key in (
            # The keys that may vary by mass split, see _sample_model_dust_params
            "sigma0", "tauA", "mu_R", "sigma_R", "uniform_RV_min", "uniform_RV_max",
            "mu_z_grad", "tau_z_grad", "M_step"
        ):
            for suffix in ("", "_HM", "_LM"):
                if key + suffix in samples:
                    yaml_data[key.upper()+suffix] = float(np.mean(samples[key + suffix]))

        if not args["infer_dust_properties"]:
            yaml_data.update({"SIGMA0": float(self.sigma0), "TAUA": float(self.tauA)})

        if isinstance(args["RV"], Number):
            yaml_data["RV"] = float(args["RV"])
        elif args["shared_RV"]:
            yaml_data["RV"] = float(np.mean(samples["RV"]))

        for key in ("MU_R", "SIGMA_R"):  # for backwards compatibility, need
            if key in yaml_data:
                yaml_data[key.replace("_", "")] = yaml_data.pop(key)

        with open(args["outputdir"] / outfile, "w") as file:
            yaml.dump(yaml_data, file)

    #################
    ### Utilities ###
    #################
    def get_J_t(self, t: ArrayLike, extrap: int | str | None = 1) -> Array:
        """
        Matrix for cubic spline interpolation in time axis for each SN.

        Parameters
        ----------
        t : (N_max_epochs, N_sn)
            Array of phase values used to calculate J_t.
        extrap
            Integer 1, 2, or 3 describing polynomial order of extrapolation beyond phase knots.

        Returns
        -------
        (N_sn, N_tau_knots + 1, N_max_epochs)
            Matrix for cubic spline interpolation in time axis for each SN.
        """
        keep_shape = t.shape
        return self.J_t_map(t.flatten(), self.tau_knots, self.KD_t, extrap).reshape((*keep_shape, self.tau_knots.shape[0]+1)).transpose(1, 2, 0)

    def get_hsiao_interp(self, t: ArrayLike) -> Array:
        """
        Coordinates for linear interpolation of the Hsiao template spectra.

        Parameters
        ----------
        t : (N_max_epochs, N_sn) or (...)
            Phases at which to evaluate Hsiao template interpolation indices.

        Returns
        -------
        (3, N_max_epochs, N_sn) or (3, ...)
            Array containing floor(phase), ceil(phase), and fractional phase remainder.
        """
        return jnp.array([self.hsiao_offset + jnp.floor(t), self.hsiao_offset + jnp.ceil(t), jnp.remainder(t, 1)])

    def get_fitprob(
        self, samples: dict, batch_size: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FITPROB for each SN using posterior mean parameters.

        Uses the BayeSN joint test statistic T = chi2_data + chi2_epsilon, where
        chi2_data is the standard flux chi-squared at the posterior mean (does not
        include "model" uncertainties, only data) and chi2_epsilon is defined as
        sum(means["eps_tform"]^2) to penalize extreme epsilon values.

        Parameters
        ----------
        samples
            Posterior samples dictionary with keys including 'theta', 'AV', 'tmax',
            'Ds', and 'eps_tform'.
        batch_size
            Batch size for evaluation, if applicable.

        Returns
        -------
        fitprob : (N_sn,)
            Goodness-of-fit probability for each SN.
        fitchi2 : (N_sn,)
            Joint test statistic T = chi2_data + chi2_epsilon.
        ndof : (N_sn,)
            Effective degrees of freedom: N_obs + alpha*N_eps - N_params.
        """
        n_sne = self.data.z_hel.shape[0]

        # --- 1. Posterior means ---
        means = {
            key: np.array(samples[key].mean(axis=(0, 1))) for key in
            # These should always be in samples due to postprocessing.
            ("theta", "AV", "tmax", "Ds", "eps_tform", "RV")
        }
        # These may not be in samples, default to 0.
        lam_shift = np.array(samples.get("lam_shift", [[0]])).mean(axis=(0, 1))
        mag_shift = np.array(samples.get("mag_shift", [[0]])).mean(axis=(0, 1))

        # eps_tform: mean over chains/samples -> ensure (N_knots_sig, n_sne)
        if means["eps_tform"].shape[0] == n_sne and means["eps_tform"].shape[1] == self.L_Sigma.shape[0]:
            means["eps_tform"] = means["eps_tform"].T
        # Reconstruct eps via L_Sigma transform (mirrors fit_model_globalRV lines 864-869)
        eps = np.matmul(self.L_Sigma, means["eps_tform"])
        eps = eps.T
        eps = eps.reshape(
            (n_sne, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F'
        )
        eps_full = jnp.zeros((n_sne, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full = eps_full.at[:, 1:-1, :].set(eps)

        # --- 2. Rebuild J_t, hsiao_interp, band_weights ---
        obs_times = self.data.mjd
        z_obs = np.asarray(self.data.z_hel)
        if "z" in samples:
            z_mean = np.array(samples['z'].mean(axis=(0, 1)))
            t = obs_times * (1 + z_obs[None, :]) / (1 + z_mean[None, :]) - means["tmax"][None, :]
        else:
            z_mean = z_obs
            t = obs_times - means["tmax"][None, :]
        if "z" in samples or "lam_shift" in samples:
            weights = self._calculate_band_weights(z_mean, self.data.MWEBV, lam_shifts=lam_shift)
        else:
            weights = self.band_weights
        hsiao_interp = self.get_hsiao_interp(t)
        J_t = self.get_J_t(t)

        # --- 3. Inputs in (N_obs, n_sne) convention ---
        band_indices = self.data.band_indices.astype(int)
        mask = self.data.mask.astype(bool)

        # --- 4. Model flux — batched over SN axis to keep peak memory bounded ---
        bs = batch_size or n_sne
        rv_is_array = isinstance(means["RV"], ArrayLike) and getattr(means["RV"], 'ndim', 0) > 0
        chunks = []
        for lo in range(0, n_sne, bs):
            hi = min(lo + bs, n_sne)
            chunk_spectra = self._get_spectra(
                means["theta"][lo:hi], means["AV"][lo:hi], self.W0, self.W1,
                eps_full[lo:hi], means["RV"][lo:hi] if rv_is_array else means["RV"],
                J_t[lo:hi], hsiao_interp[:, :, lo:hi]
            )
            chunk = self.get_flux_batch(
                model_spectra=chunk_spectra, M0=self.M0, Ds=means["Ds"][lo:hi],
                z=self.data.z_hel[lo:hi], ebv=self.data.MWEBV[lo:hi],
                band_indices=band_indices[:, lo:hi], mask=mask[:, lo:hi],
                weights=weights[lo:hi], lam_shift=lam_shift, mag_shift=mag_shift,
                num_batch=hi-lo,
            )
            chunks.append(np.asarray(chunk))
        model_flux = np.concatenate(chunks, axis=-1)

        # --- 5. chi2_data ---
        obs_flux = self.data.flux
        obs_err = self.data.flux_err
        residuals_sq = (obs_flux - model_flux) ** 2
        chi2_per_obs = jnp.where(mask, residuals_sq / obs_err ** 2, 0.0)
        chi2_data = jnp.sum(chi2_per_obs, axis=0)

        # --- 6. chi2_epsilon: sum of squared whitened epsilon ---
        N_knots_sig = self.N_knots_sig
        chi2_epsilon = np.sum(means["eps_tform"] ** 2, axis=0)

        # --- 7. Host photo-z prior term -2 ln p_host(z), added to the joint chi2 (matches SNANA) ---
        chi2_z = 0.0
        if "z" in samples:
            if self.z_icdf_grid is None:  # Gaussian prior: pull^2 (drop the log-norm const, as SNANA does)
                z_obs_err = np.asarray(self.data.z_hel_err)
                chi2_z = ((z_mean - z_obs) / z_obs_err) ** 2
            else:  # quantile prior: p_host is dCDF/dz, the finite-difference slope of the quantiles
                zq, pl = np.asarray(self.z_icdf_grid), np.asarray(self.z_u_grid)
                slopes = np.diff(pl) / np.diff(zq, axis=1)  # dP/dz per quantile bin
                k = np.clip(np.sum(zq <= z_mean[:, None], axis=1) - 1, 0, pl.shape[0] - 2)
                p_host = slopes[np.arange(len(z_mean)), k]
                chi2_z = np.where(p_host > 0, -2.0 * np.log(p_host), 1000.0)

        # --- 8. Joint statistic, NDOF, FITPROB ---
        T_joint = np.array(chi2_data) + chi2_epsilon + chi2_z
        # NDOF = N_obs + alpha*N_eps - N_params.
        # Physical params (tmax, theta, AV, Ds, +RV) consume DOF from data,
        # but some may be fixed at specific values and not contribute to N_params.
        N_params = sum([
            1 if np.std(samples[key]) else 0
            for key in ("tmax", "theta", "AV", "Ds", "RV")
        ])
        # Epsilon knots are prior-constrained, so their effective DOF is
        # between 0 (fully free) and N_knots_sig (fully prior-determined).
        # alpha=0.75 calibrated on Foundation DR1 sample.
        # If eps was fixed to 0 everywhere, alpha goes to 0.
        alpha = 0.75 if np.std(samples["eps_tform"]) else 0
        # Sampled redshifts are net-zero DOF: the host-z prior adds one prior point
        # while the sampled z consumes one parameter (as SNANA does), leaving ndof
        # unchanged.
        # lam_shift and mag_shift consume a DOF per bandpass.
        N_bps = weights.shape[-1]
        N_params -= weights.shape[-1] * (int("lam_shift" in samples) + int("mag_shift" in samples))
        ndof = (np.sum(np.array(mask), axis=0) + alpha * N_knots_sig - N_params).astype(int)
        fitprob = np.where(
            (ndof > 0) & (T_joint > 0),
            chi2.sf(T_joint, ndof),
            1.0
        )
        return fitprob, T_joint, ndof

    def sample_del_M(self, N: int) -> np.ndarray:
        """
        Samples grey offset del_M from model prior.

        Parameters
        ----------
        N
            Number of objects to sample for.

        Returns
        -------
        (N,)
            Sampled del_M values.
        """
        del_M = np.random.normal(0, self.sigma0, N)
        return del_M

    def sample_AV(self, N: int) -> np.ndarray:
        """
        Samples AV from model prior.

        Parameters
        ----------
        N
            Number of objects to sample for.

        Returns
        -------
        (N,)
            Sampled AV values.
        """
        AV = np.random.exponential(self.tauA, N)
        return AV

    def sample_RV(self, N: int) -> np.ndarray:
        """
        Samples RV from model prior if a population model.
        Otherwise, returns the SEDmodel.RV attribute N times.

        Parameters
        ----------
        N
            Number of objects to sample for.

        Returns
        -------
        (N,)
            Sampled RV values.
        """
        if self.mu_R is None:
            return np.full(N, self.RV)
        else:
            return truncnorm.rvs(-self.trunc_val/self.sigma_R, np.inf, loc=self.mu_R, scale=self.sigma_R, size=N)

    def sample_theta(self, N: int) -> np.ndarray:
        """
        Samples theta from model prior.

        Parameters
        ----------
        N
            Number of objects to sample for.

        Returns
        -------
        (N,)
            Sampled theta values.
        """
        theta = np.random.normal(0, 1, N)
        return theta

    def sample_epsilon(self, N: int) -> np.ndarray:
        """
        Samples epsilon from model prior.

        Parameters
        ----------
        N
            Number of objects to sample for.

        Returns
        -------
        (N, N_l_knots, N_tau_knots)
            Sampled epsilon values.
        """
        eps_mu = jnp.zeros(self.N_knots_sig)
        eps_tform = np.random.multivariate_normal(eps_mu, np.eye(self.N_knots_sig), N)
        eps_tform = eps_tform.T
        eps = np.matmul(self.L_Sigma, eps_tform)
        eps = eps.T
        eps = np.reshape(eps, (N, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order="F")
        eps_full = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full[:, 1:-1, :] = eps
        return eps_full

    def sample_lam_shift(self, *args: Any) -> np.ndarray:
        """
        Samples lam_shift from model prior.

        Returns
        -------
        (N_used_bands,)
            Sampled lam_shift values.
        """
        lam_shift = np.random.normal(0, self.used_wave_sigmas)
        return lam_shift

    def sample_mag_shift(self, *args: Any) -> np.ndarray | Array:
        """
        Samples mag_shift from model prior.

        Returns
        -------
        (N_used_bands,)
            Sampled mag_shift values.
        """
        if not len(self.used_calib_chcov):
            return jnp.zeros(1)
        mag_shift = np.random.multivariate_normal(np.zeros(len(self.used_calib_cov)), self.used_calib_cov)
        return mag_shift

    def simulate_spectrum(
        self,
        t: ArrayLike,
        N: int,
        dl: int = 10,
        z: float | ArrayLike = 0,
        mu: float | ArrayLike | str = 0,
        ebv_mw: float | ArrayLike = 0,
        RV: float | ArrayLike | None = None,
        logM: float | ArrayLike | None = None,
        del_M: float | ArrayLike | None = None,
        AV: float | ArrayLike | None = None,
        theta: float | ArrayLike | None = None,
        eps: ArrayLike | int | None = None,
    ) -> tuple[np.ndarray, Array, dict]:
        """
        Simulates spectra for given parameter values in the observer-frame.

        Scalar arguments for SN parameters (such as ``z``, ``mu``, ``ebv_mw``,
        ``RV``, ``del_M``, ``AV``, ``theta``) are broadcast to all ``N`` simulated
        supernovae. Alternatively, a 1D array of length ``N`` can be passed to
        specify individual values for each supernova.

        Parameters passed as ``None`` (such as ``del_M``, ``AV``, ``theta``, ``eps``,
        and ``RV`` for population models) are sampled from their respective model priors.

        ``eps`` can be passed as ``None`` (sampled from prior), scalar or 0 (no perturbation),
        2D array of shape ``(N_l_knots, N_tau_knots)`` (shared perturbation across all SNe),
        or 3D array of shape ``(N, N_l_knots, N_tau_knots)`` (individual perturbations).

        Parameters
        ----------
        t : (N_t,)
            Set of t values to simulate spectra at.
        N
            Number of separate objects to simulate spectra for.
        dl
            Wavelength spacing for simulated spectra in rest-frame. Default is 10 AA.
        z : (N,) or scalar
            Redshift to simulate spectra at, affecting observer-frame wavelengths and
            reducing spectra by factor of (1+z).
        mu : (N,) or scalar or str
            Distance modulus to simulate spectra at. If set to "z", distance moduli
            corresponding to the redshift values passed in the default model cosmology
            will be used. Technically these are heliocentric redshifts rather than
            Hubble diagram redshifts so it's not perfect, but can be useful sometimes.
        ebv_mw : (N,) or scalar
            Milky Way E(B-V) values for simulated spectra.
        RV : (N,) or scalar or None
            Host galaxy dust R(V) values.
        logM : (N,) or scalar or None
            Host galaxy log stellar mass (currently unused).
        del_M : (N,) or scalar or None
            Grey offset del_M value to be used for each SN.
        AV : (N,) or scalar or None
            Host extinction AV value to be used for each SN.
        theta : (N,) or scalar or None
            Theta value to be used for each SN.
        eps : (N_l, N_tau) or (N, N_l, N_tau) or scalar or None
            Epsilon perturbation values to be used for each SN.

        Returns
        -------
        l_o : (N_wl,)
            Array of observer-frame wavelength values.
        spectra : (N, N_wl, N_t)
            Array of simulated spectra.
        param_dict : dict
            Dictionary of corresponding parameter values for each simulated object.
        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.array(del_M)
            if len(del_M.shape) == 0:
                del_M = del_M.repeat(N)
            elif del_M.shape[0] != N:
                raise ValueError(
                    "If not providing a scalar del_M value, array must be of same length as the number of "
                    "objects to simulate, N"
                )
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.array(AV)
            if len(AV.shape) == 0:
                AV = AV.repeat(N)
            elif AV.shape[0] != N:
                raise ValueError(
                    "If not providing a scalar AV value, array must be of same length as the number of "
                    "objects to simulate, N"
                )
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.array(theta)
            if len(theta.shape) == 0:
                theta = theta.repeat(N)
            elif theta.shape[0] != N:
                raise ValueError(
                    "If not providing a scalar theta value, array must be of same length as the number of "
                    "objects to simulate, N"
                )
        if eps is None:
            eps = self.sample_epsilon(N)
        else:
            eps = np.array(eps)
            if len(eps.shape) == 0:
                if eps == 0:
                    eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
                else:
                    raise ValueError(
                        "For epsilon, please pass an ArrayLike object of shape (N, l_knots, tau_knots). The only scalar "
                        "value accepted is 0, which will effectively remove the effect of epsilon"
                    )
            elif (
                len(eps.shape) == 2
                and eps.shape[0] == self.l_knots.shape[0]
                and eps.shape[1] == self.tau_knots.shape[0]
            ):
                eps = eps[None, ...].repeat(N, axis=0)
            elif (
                len(eps.shape) != 3
                or eps.shape[0] != N
                or eps.shape[1] != self.l_knots.shape[0]
                or eps.shape[2] != self.tau_knots.shape[0]
            ):
                raise ValueError(
                    "For epsilon, please pass an ArrayLike object of shape (N, l_knots, tau_knots)"
                )
        ebv_mw = np.array(ebv_mw)
        if len(ebv_mw.shape) == 0:
            ebv_mw = ebv_mw.repeat(N)
        elif ebv_mw.shape[0] != N:
            raise ValueError(
                "For ebv_mw, either pass a single scalar value or an array of values for each of the N simulated objects"
            )
        if RV is None:
            RV = self.RV
        RV = np.array(RV)
        if len(RV.shape) == 0:
            RV = RV.repeat(N)
        elif RV.shape[0] != N:
            raise ValueError(
                "For RV, either pass a single scalar value or an array of values for each of the N simulated objects"
            )
        z = np.array(z)
        if len(z.shape) == 0:
            z = z.repeat(N)
        elif z.shape[0] != N:
            raise ValueError(
                "For z, either pass a single scalar value or an array of values for each of the N simulated objects"
            )
        mu = np.array(mu)
        if len(mu.shape) == 0:
            mu = mu.repeat(N)
        elif mu.shape[0] != N:
            raise ValueError(
                "For mu, either pass a single scalar value or an array of values for each of the N simulated objects"
            )
        param_dict = {
            "del_M": del_M,
            "AV": AV,
            "theta": theta,
            "eps": eps,
            "z": z,
            "mu": mu,
            "ebv_mw": ebv_mw,
            "RV": RV,
        }
        l_r = np.arange(min(self.l_knots), max(self.l_knots) + dl, dl, dtype=float)
        l_r = l_r[l_r <= max(self.l_knots)]
        l_o = l_r[None, ...].repeat(N, axis=0) * (1 + z[:, None])

        old_model_wave = self.model_wave
        self.model_wave = l_r
        KD_l = invKD(self.l_knots)
        self.J_l_T = device_put(spline_coeffs(self.model_wave, self.l_knots, KD_l))
        self._load_hsiao_template()
        self.load_ext_rel(self.ext_rel.name)

        t = jnp.array(t)
        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = self.get_hsiao_interp(t)
        J_t = self.get_J_t(t)
        spectra = self._get_spectra(
            theta, AV, self.W0, self.W1, eps, RV, J_t, hsiao_interp
        )

        # cleaning
        self.model_wave = old_model_wave
        KD_l = invKD(self.l_knots)
        self.J_l_T = device_put(spline_coeffs(self.model_wave, self.l_knots, KD_l))
        self._load_hsiao_template()
        self.load_ext_rel(self.ext_rel.name)
        return l_o, spectra, param_dict

    def simulate_light_curve(
        self,
        t: ArrayLike,
        N: int,
        bands: list[str],
        yerr: float | ArrayLike = 0,
        err_type: str = "mag",
        z: float | ArrayLike = 0,
        zerr: float = 1e-4,
        mu: float | ArrayLike | str = 0,
        ebv_mw: float | ArrayLike = 0,
        lam_shift: float | ArrayLike = 0,
        mag_shift: float | ArrayLike = 0,
        RV: float | ArrayLike | None = None,
        logM: float | ArrayLike | None = None,
        tmax: float | ArrayLike = 0,
        del_M: float | ArrayLike | None = None,
        AV: float | ArrayLike | None = None,
        theta: float | ArrayLike | None = None,
        eps: ArrayLike | int | None = None,
        mag: bool = True,
        write_to_files: bool = False,
        output_dir: str | Path | None = None,
        band_weights: Array | None = None,
        ext_rel: str = "F19",
    ) -> tuple[Array, Array, dict]:
        """
        Simulate light curves from the BayeSN model in either mag or flux space
        and save them to SNANA-format text files if requested.

        Scalar arguments for SN parameters (such as ``z``, ``mu``, ``ebv_mw``,
        ``RV``, ``tmax``, ``del_M``, ``AV``, ``theta``) are broadcast to all ``N``
        simulated supernovae. Alternatively, a 1D array of length ``N`` can be passed
        to specify individual values for each supernova.

        Parameters passed as ``None`` (such as ``del_M``, ``AV``, ``theta``, ``eps``,
        and ``RV`` for population models) are sampled from their respective model priors.

        ``eps`` can be passed as ``None`` (sampled from prior), scalar or 0 (no perturbation),
        2D array of shape ``(N_l_knots, N_tau_knots)`` (shared perturbation across all SNe),
        or 3D array of shape ``(N, N_l_knots, N_tau_knots)`` (individual perturbations).

        ``yerr`` can be a scalar (same uncertainty for all observations), a 1D array
        (same error profile across epochs for each SN), or a 2D array of shape ``(N, N_obs)``.

        Parameters
        ----------
        t : (N_t,) or (N_bands,)
            Set of t values to simulate spectra at.
            If len(t) == len(bands), will assume the t values correspond to the bands.
            Otherwise, will simulate photometry at each value of t for each band.
        N
            Number of separate objects to simulate spectra for.
        bands
            List of bands in which to simulate photometry.
        yerr : scalar or (N_t,) or (N, N_t)
            Observational uncertainties. If 0, exact model photometry is returned.
        err_type
            Specifies which type of error you are passing: "mag" or "flux".
        z : (N,) or scalar
            Redshift to simulate spectra at.
        zerr
            Error on z. Only needed when saving to SNANA-format light curve files.
        mu : (N,) or scalar or str
            Distance modulus to simulate spectra at. If set to "z", distance moduli
            corresponding to the redshift values passed in the default model cosmology
            will be used. Technically these are heliocentric redshifts rather than
            Hubble diagram redshifts so it's not perfect, but can be useful sometimes.
        ebv_mw : (N,) or scalar
            Milky Way E(B-V) values for simulated spectra.
        lam_shift : (N_bands,) or scalar
            Wavelength shifts in Angstroms.
        mag_shift : (N_bands,) or scalar
            Magnitude zero-point offsets.
        RV : (N,) or scalar or None
            Host galaxy dust R(V) values.
        logM : (N,) or scalar or None
            Host galaxy log stellar mass (currently unused).
        tmax : (N,) or scalar
            Time of maximum in rest-frame days.
        del_M : (N,) or scalar or None
            Grey offset del_M value to be used for each SN.
        AV : (N,) or scalar or None
            Host extinction AV value to be used for each SN.
        theta : (N,) or scalar or None
            Theta value to be used for each SN.
        eps : (N_l, N_tau) or (N, N_l, N_tau) or scalar or None
            Epsilon perturbation values to be used for each SN.
        mag
            Determines whether returned values are mags (True) or fluxes (False).
        write_to_files
            Determines whether to save simulated light curves to SNANA-format light
            curve files.
        output_dir
            Path to output directory to save simulated SNANA-format files.
        band_weights : (N, N_wl, N_bands) or None
            Array containing band weights (transmission functions) to use for photometry.
        ext_rel
            Extinction relation name to use.

        Returns
        -------
        data : (N_obs_total, N)
            Array containing simulated flux or mag values.
        yerr : (N_obs_total, N)
            Array containing corresponding errors for each data point.
        param_dict : dict
            Dictionary of corresponding parameter values for each simulated object.
        """
        def clean_input(
            var : float | ArrayLike,
            sample_fn : Callable[[int | None], Array] | None,
            N : int,
            name : str
        ) -> Array:
            # z and ebv_mw to not have sampling functions by design.
            # If they are not provided with a scalar or an array of shape N
            # An error should be raised.
            if var is None:
                if sample_fn is None:
                    raise ValueError(
                        f"{name} must be defined with either a single scalar to be "
                        f"used for all objects, or an array of scalars of length N={N}."
                    )
                var = sample_fn(N)
            else:
                var = np.array(var)
                if np.isscalar(var) or (
                    isinstance(var, (Array, np.ndarray))
                    and not len(var.shape)
                ):
                    var = var.repeat(N)
                elif name not in ("lam_shift", "mag_shift") and var.shape[0] != N:
                    raise ValueError(
                        f"If not providing a scalar {name} value, array must be of "
                        f"same length as the number of objects to simulate, N={N}."
                    )
                elif name in ("lam_shift", "mag_shift") and var.shape[0] != len(bands)+1:
                    raise ValueError(
                        f"If not providing a scalar {name} value, array must be of "
                        f"same length as the number of bandpasses to simulate (and the "
                        f"NULL band) len(bands)={len(bands)+1}."
                    )
            return var

        self._set_used_bands(bands)
        if self.ext_rel.name != ext_rel:
            self.load_ext_rel(ext_rel)
        param_dict = {
            "del_M": (del_M, self.sample_del_M, N),
            "AV": (AV, self.sample_AV, N),
            "theta": (theta, self.sample_theta, N),
            "z": (z, None, N),
            "ebv_mw": (ebv_mw, None, N),
            "RV": (RV, self.sample_RV, N),
            "lam_shift": (lam_shift, self.sample_lam_shift, len(self.used_wave_sigmas)),
            "mag_shift": (mag_shift, self.sample_mag_shift, len(self.used_calib_cov)),
        }
        for key, val in param_dict.items():
            param_dict[key] = clean_input(val[0], val[1], val[2], key)
        if isinstance(mu, str) and mu == "z":
            param_dict["mu"] = self.cosmo.distmod(z).value
        else:
            param_dict["mu"] = clean_input(mu, None, N, "mu")
        # eps is special since it needs to be a (N, N_l_knots, N_tau_knots) shaped-array
        # rather than just (N,) shaped.
        if eps is None:
            eps = self.sample_epsilon(N)
        elif np.isscalar(eps) or (
            isinstance(eps, (ArrayImpl, np.ndarray))
            and not len(eps.shape)
        ):
            eps = np.array(eps)
            if eps == 0:
                eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
            else:
                raise ValueError(
                    "For epsilon, please pass an ArrayLike object of shape (N, "
                    "N_l_knots, N_tau_knots). The only scalar value accepted is 0, "
                    "which will effectively remove the effect of epsilon."
                )
        elif (
            eps.ndim != 3
            or eps.shape[0] != N
            or eps.shape[1] != self.l_knots.shape[0]
            or eps.shape[2] != self.tau_knots.shape[0]
        ):
            raise ValueError(
                "For epsilon, please pass an ArrayLike object of shape (N, N_l_knots, N_tau_knots)"
            )
        param_dict["eps"] = eps
        param_dict["Ds"] = param_dict["mu"] + param_dict["del_M"]
        param_dict["ebv"] = param_dict["ebv_mw"]

        if t.shape[0] == np.array(bands).shape[0]:
            band_indices = np.array([self.used_band_dict[self.band_dict[band]] for band in bands])
            band_indices = band_indices[:, None].repeat(N, axis=1).astype(int)
        else:
            t = jnp.array(t)
            num_per_band = t.shape[0]
            num_bands = len(bands)
            band_indices = np.zeros(num_bands * num_per_band)
            t = t[:, None].repeat(num_bands, axis=1).flatten(order="F")
            for i, band in enumerate(bands):
                if band not in self.band_dict:
                    raise ValueError(f"{band} is present in filters yaml file")
                band_indices[i * num_per_band : (i + 1) * num_per_band] = (
                    self.used_band_dict[self.band_dict[band]]
                )
            band_indices = band_indices[:, None].repeat(N, axis=1).astype(int)
        mask = np.ones_like(band_indices)
        t = jnp.repeat(t[..., None], N, axis=1)
        tmax = np.atleast_1d(tmax)
        if len(tmax) == 1 and N > 1:
            tmax = tmax.repeat(N)
        param_dict["tmax"] = tmax
        t = t - tmax[None, :]
        J_t = self.get_J_t(t)
        hsiao_interp = self.get_hsiao_interp(t)

        phot_epoch_spectra = self._get_spectra(
            W0=self.W0,
            W1=self.W1,
            J_t=J_t,
            hsiao_interp=hsiao_interp,
            **param_dict,
        )
        if mag:
            fn = self.get_mag_batch
        else:
            fn = self.get_flux_batch

        if band_weights is None:
            band_weights = self._calculate_band_weights(
                param_dict["z"],
                param_dict["ebv_mw"],
                lam_shifts=param_dict["lam_shift"]
            )
        else:
            if band_weights.shape == (1, len(self.model_wave), len(bands)+1) and N > 1:
                if np.std(param_dict["z"]):
                    raise ValueError(
                        "band_weights was provided as an array of shape "
                        f"{band_weights.shape}, which would normally be interpreted "
                        "as the weight array to repeat for all SNe, but the given z "
                        f"values are not all the same, meaning band_weights should "
                        "differ between targets. Please provide a band_weights array "
                        f"of shape (N, N_model_wave, N_bands) = ({N}, "
                        f"{len(self.model_wave)}, {len(bands)})."
                    )
                weights = jnp.repeat(band_weights, N, axis=0)
            elif band_weights.shape != (N, len(self.model_wave), len(bands)+1):
                raise ValueError(
                    "band_weights was provided as an array of shape "
                    f"{band_weights.shape}, which does not match the expected shape "
                    f"(N, N_wl, N_bp) = ({N}, {len(self.model_wave)}, {len(bands)+1})."
                )

        data = fn(
            model_spectra=phot_epoch_spectra,
            M0=self.M0,
            band_indices=band_indices,
            mask=mask,
            weights=band_weights,
            num_batch=N,
            **param_dict
        )
        # Apply error if specified
        yerr = jnp.array(yerr)
        if len(yerr.shape) == 0 and yerr == 0:
            yerr = np.zeros_like(data)
        elif err_type == "mag" and not mag:
            # if data is negative, yerr will become negative.
            # abs is not quite right, but there's no right answer when converting
            # negative fluxes to magnitudes
            yerr = np.abs(yerr * (np.log(10) / 2.5) * data)
        if len(yerr.shape) == 0:  # Single error for all data points
            yerr = np.ones_like(data) * yerr
        elif len(yerr.shape) == 1:
            assert data.shape[0] == yerr.shape[0], (
                f"If passing a 1d array, shape of yerr must match number of "
                f"simulated data points per objects, {data.shape[0]}"
            )
            yerr = np.repeat(yerr[..., None], N, axis=1)
        else:
            assert data.shape == yerr.shape, (
                f"If passing a 2d array, shape of yerr must match generated data shape"
                f" of {data.shape}"
            )
        data = np.random.normal(data, yerr)

        if write_to_files and mag:
            if output_dir is None:
                raise ValueError(
                    "If writing to SNANA files, please provide an output directory"
                )
            if not Path(output_dir).exists():
                os.mkdir(output_dir)
            sn_names, sn_files = [], []
            for i in range(N):
                sn_name = f"{i}"
                sn_t, sn_mag, sn_mag_err, sn_z, sn_ebv_mw = (
                    t[:, i],
                    data[:, i],
                    yerr[:, i],
                    param_dict["z"][i],
                    param_dict["ebv_mw"][i],
                )
                sn_t = sn_t * (1 + sn_z)
                sn_tmax = 0
                inv_used = {v: self.inv_band_dict[k] for k, v in self.used_band_dict.items()}
                sn_flt = [inv_used[f] for f in band_indices[:, i]]
                sn_file = write_snana_lcfile(
                    output_dir=output_dir,
                    snname=sn_name,
                    mjd=sn_t,
                    flt=sn_flt,
                    mag=sn_mag,
                    mag_err=sn_mag_err,
                    tmax=sn_tmax,
                    z_helio=sn_z,
                    z_cmb=sn_z,
                    z_cmb_err=zerr,
                    ebv_mw=sn_ebv_mw,
                )
                sn_names.append(sn_name)
                sn_files.append(sn_file)
        elif write_to_files:
            raise ValueError("If writing to SNANA files, please generate mags")
        return data, yerr, param_dict
    def list_bandpasses(
        self, group_by: str = "bp_name", return_dict: bool = False
    ) -> dict[str, list[str]] | None:
        """
        Convenience method for listing recognised bandpasses.

        Parameters
        ----------
        group_by
            Grouping criterion: "bp_name" (default) or "site".
        return_dict
            If True, return the dictionary of bandpass names.

        Returns
        -------
        dict or None
            If ``return_dict`` is True, returns dictionary mapping groups to bandpass names.
        """
        full_names = self.filter_dict["filters"].keys()
        paths = [val["path"].strip(
            str(self.__root_dir__ / "bayesn-filters/filters/")
            ) for val in self.filter_dict["filters"].values()]
        ret_dict = {}
        match group_by:
            case "bp_name":
                for bp in set(name.split("_")[0] for name in full_names):
                    ret_dict[bp] = [name for name in full_names if name.startswith(bp+"_") or name == bp]
            case "site":
                for site in [path.split('/')[0] for path in paths]:
                    ret_dict[site] = [name for i,name in enumerate(full_names) if paths[i].startswith(site)]
            case _:
                raise ValueError(
                    "Unrecognised 'group_by' argument. "
                    "Valid options are 'bp_name' and 'site'"
                )
        for key, val in ret_dict.items():
            print(key+":")
            print(val)
            print()
        if return_dict:
            return ret_dict

    @property
    def inv_band_dict(self) -> dict[int, str]:
        """
        Inverse mapping from filter indices to BayeSN bandpass names.

        Returns
        -------
        dict[int, str]
            Dictionary mapping integer band index to bandpass name.
        """
        return {val: key for key, val in self.band_dict.items()}

    def get_model_params(self) -> np.ndarray:
        """
        Return 1D array of scalar model parameters.

        Parameters are packed in the order: ``L_Omega``, ``W0``, ``W1``,
        ``sigma0``, ``tauA``, followed by ``(RV,)`` or ``(mu_RV, sigma_RV)``
        depending on whether a population RV model is used.

        Returns
        -------
        (N_params,)
            Flattened array of model parameter values.
        """
        N_L = int(self.N_knots_sig*(self.N_knots_sig+1)/2)
        N_W = self.l_knots.shape[0]*self.tau_knots.shape[0]
        N_params = N_L + 2*N_W + 3 + int(self.sigma_R is not None)
        model_params = np.zeros(N_params)
        k = 0
        for i in range(self.N_knots_sig):
            for j in range(self.N_knots_sig):
                if j > i:
                    continue
                model_params[k] = self.L_Sigma[i,j]
                k += 1
        for key in ("W0", "W1"):
            for i in range(N_W):
                model_params[k] = getattr(self, key)[i%self.l_knots.shape[0], i//self.l_knots.shape[0]]
                k += 1
        for param in ("sigma0", "tauA"):
            model_params[k] = getattr(self, param)
            k += 1
        if self.mu_R is not None:
            model_params[k] = float(self.mu_R)
            model_params[k+1] = float(self.sigma_R)
        else:
            model_params[k] = float(self.RV)
        return model_params
