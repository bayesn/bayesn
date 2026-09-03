"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model.
"""

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
import sncosmo
import timeit
from ruamel.yaml import YAML
import time
from tqdm import tqdm
from typing import Any, NamedTuple

from .lm_optim import run_lm_laplace_gn, compute_gn_scale_tril
from .spline_utils import invKD, spline_coeffs, spline_coeffs_step
from .extinction_relations import DustExtRel
from .io import write_snana_lcfile, read_snana_spectra
from bayesn.datasets import SNDataset
from bayesn.utils import _predict, _prior_pot
import bayesn.zltn_utils as zltn
from bayesn import constants

yaml = YAML(typ="safe")
yaml.default_flow_style = False

jax.config.update("jax_enable_x64", True)  # Enables 64 computation

np.seterr(divide="ignore", invalid="ignore")  # Disable divide by zero warnings

# jax.config.update("jax_platform_name", "cpu")  # Forces CPU

BASE_DIR: Path = Path(__file__).parent.absolute()
with open(BASE_DIR.parent / "defaults.yaml", "r") as file:
    default_kwargs = yaml.load(file)
default_kwargs["AV_dist"] = dist.Exponential


class DustParams(NamedTuple):
    """Container for population-level dust parameters."""
    sigma0: ArrayLike
    tauA: ArrayLike
    mu_R: ArrayLike | None
    sigma_R: ArrayLike | None
    phi_alpha_R: ArrayLike | None
    mu_z_grad: ArrayLike | float
    tau_z_grad: ArrayLike | float
    global_RV: ArrayLike | float


class DustPop(NamedTuple):
    """Container for high-mass and low-mass population dust parameters."""
    HM: DustParams
    LM: DustParams | None = None
    HM_flag: ArrayLike | None = None
    sigma0: ArrayLike | float | None = None
    split_variant: str | None = None


class SEDmodel(object):
    """
    BayeSN-SED Model

    Class which imports a BayeSN model, and allows one to fit or simulate
    Type Ia supernovae based on this model.

    Methods
    -------
    get_flux_batch:
        Get integrated fluxes across a large number of SNe, phases, and bands.
    def get_flux_from_chains:
        Get model photometry for posterior samples from model fitting chains.
    get_mag_batch:
        Get magnitudes across a large number of SNe, phases, and bands.
    get_spectra:
        Get spectra for a large number of SNe and phases. Wraps a more performant,
        lower-level method that has more restrictive arguments.
    initial_guess:
        Defined method used to initialise chains for model training.
    parse_args:
        Parse the args from the input yaml file along with any command line arguments
        to define the job being run.
    postprocess:
        Postprocess the output of the MCMC run if required and save the chains and
        summaries.
    process_dataset:
        Process a set of data for use by the BayeSN model. Calls lower-level methods
        depending on the input yaml for use with version photometry or a data table.
    run:
        Run an inference job using the BayeSN model.
    sample_AV:
        Sample AV from the population distribution based on a pre-trained tauA.
    sample_del_M:
        Sample delta_M from the population distribution based on a pre-trained sigma0.
    sample_epsilon:
        Sample epsilon from the population distribution based on a pre-trained L_Sigma.
    sample_theta:
        Sample theta from the standard normal distribution.
    simulate_light_curve:
        Simulate a light curve or set of light curves from the BayeSN SED model.
    simulate_spectrum:
        Simulate a specrum or set of spectra from the BayeSN SED model.
    _model:
        Defines the BayeSN parameters to infer conditioned on input data.
        Calls helper functions that specify the components in a modular fashion.

    Attributes
    ----------
    cosmo: astropy.cosmology.FlatLambdaCDM
        Defines the fiducial cosmology assumed by the model when training
    RV_MW: Scalar
        RV value for calculating Milky Way extinction
    sigma_pec: Scalar
        Peculiar velocity to be used in calculating redshift uncertainties, default = 150 km/s
    l_knots: Array
        Array of wavelength knots which the model is defined at
    t_knots: Array
        Array of time knots which the model is defined at
    W0: Array shape (N_l_knots, N_tau_knots)
        W0 matrix for loaded model
    W1: Array shape (N_l_knots, N_tau_knots)
        W1 matrix for loaded model
    L_Sigma: Array shape (N_knots_sig_l, N_knots_sig_l)
        Covariance matrix describing epsilon distribution for loaded model
        N_knots_sig_l = (N_l_knots-2) * N_tau_knots
    M0: scalar
        Reference absolute magnitude for scaling Hsiao template
    sigma0: Scalar
        Standard deviation of grey offset parameter for loaded model
    RV: Scalar
        Global host extinction value for loaded model
    tauA: Scalar
        Global tauA value for exponential AV prior for loaded model
    spectrum_bins: int
        Number of wavelength bins used for modelling spectra and calculating photometry. Based on ParSNiP as presented
        in Boone+21
    hsiao_flux: Array shape (N_wl, 105)
        Grid of flux values for Hsiao template interpolated to SEDmodel.model_wave.
    hsiao_t: Array shape (105,)
        Time values corresponding to Hsiao template grid
    hsiao_l: Array shape (2401)
        Wavelength values corresponding to Hsiao template grid
    """
    ######################
    ### Initialisation ###
    ######################
    def __init__(
        self,
        num_devices: int = 4,
        load_model: str = "T21_model",
        filter_yaml: str | None = None,
        fiducial_cosmology: dict[str, float] = {"H0": 73.24, "Om0": 0.28},
        load_ext_rel: str = "G23",
        apply_dovekie_mag_shifts: bool = True,
        apply_mag_shifts: bool = False,
        apply_lam_shifts: bool = False,
        shift_file: str | None = None,
        fluxcal_zpt: float = 27.50,
    ):
        """
        Initialises the BayeSN SED model by loading a pre-computed model, transmission
        functions, and dust extinction relation.

        Parameters
        ----------
        num_devices :
                If running on a CPU, numpyro will by default see it as a single device.
                This argument will set the number of available cores for numpyro to use
                e.g. set to 4, you can train 4 chains on 4 cores in parallel.
        load_model :
            Can be either a pre-defined BayeSN model name (see table below), or
            a path to directory containing a set of .txt files from which a
            valid model can be constructed. Currently implemented default models
            are listed below.

            "G26_model": Grayling+26 BayeSN model (arXiv:2606.19429).
                         Covers rest wavelength range of 2800-10800A (ugriz). Intended
                         for cosmology; jointly fits filter wavelength and zero-point
                         cross-calibration shifts alongside the SED, using Dovekie as
                         an informative prior. Population RV distribution. Trained on
                         1024 SNe Ia from the Kenworthy+21 compilation with Dovekie
                         calibration updates (Foundation, CfA3, CfA4, CSP, SDSS, PS1,
                         DES, SNLS).
            "G25_model": Grayling+26 phase-extended optical+NIR BayeSN model
                         (MNRAS 548, stag340; arXiv:2510.11719; BayeSN-TD). Covers
                         rest wavelength range of 2800-18500A (UBgVrizYJH) and phase
                         range -10 to +85 days, motivated by fitting late-time
                         observations of strongly-lensed SNe Ia.  Population RV
                         distribution. Trained on 278 SNe Ia combining Avelino+19 low-z
                         compilation, Foundation DR1 (Foley+18, Jones+19), and CSP-I.
            "W22_model": Ward+22 No-Split BayeSN model (ApJ 956, 111; arXiv:2209.10558).
                         Covers rest wavelength range of 3000-18500A (BVRIYJH). No
                         treatment of host mass effects.  Global RV assumed. Trained on
                         Foundation DR1 (Foley+18, Jones+19) and low-z Avelino+19
                         (ApJ, 887, 106) compilation of CfA, CSP and others.
            "T21_model": Thorp+21 No-Split BayeSN model (arXiv:2102:05678).
                         Covers rest wavelength range of 3500-9500A (griz). No
                         treatment of host mass effects.  Global RV assumed. Trained on
                         Foundation DR1 (Foley+18, Jones+19).
            "M20_model": Mandel+20 BayeSN model (arXiv:2008.07538).
                         Covers rest wavelength range of 3000-18500A (BVRIYJH). No
                         treatment of host mass effects.  Global RV assumed. Trained on
                         low-z Avelino+19 (ApJ, 887, 106) compilation of CfA, CSP and
                         others.
        fiducial_cosmology :
            Dictionary containg keys "H0" and "Om0" for initialising an
            astropy.cosmology.FlatLambdaCDM instance.
            Default from Riess+16 (ApJ, 826, 56).
        filter_yaml :
            Path to yaml file containing details on filters and standards to use.
            If not specified, will look for a file called filters.yaml in directory that
            BayeSN is called from.
        load_ext_rel :
            Name of dust extinction relation to load
            Available choices are listed below
            "CCM89"
            "C94"
            "O94"
            "F99"
            "F99_AVGLMC"
            "F99_LMC2"
            "F99_SNANA"
            "C00"
            "F04"
            "VCG04"
            "FM07"
            "GCC09"
            "F19"
            "D22"
            "G23"
        apply_dovekie_mag_shifts :
            Argument passed to SEDmodel.load_band_weights
            Shift the zero-points of bandpasses used in the Dovekie analysis
            (Popovic et al. 2025)
        apply_mag_shifts :
            Argument passed to SEDmodel.load_band_weights
        apply_lam_shifts :
            Argument passed to SEDmodel.load_band_weights
        shift_file :
            Argument passed to SEDmodel.load_band_weights
        """

        # Settings for jax/numpyro
        numpyro.set_host_device_count(num_devices)
        self.start_time = time.time()
        self.end_time = None
        # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        print("Current devices:", jax.devices())

        self.__root_dir__ = BASE_DIR
        print(f"Currently working in {os.getcwd()}")

        self.filter_yaml = filter_yaml
        built_in_models = [f.name for f in self.__root_dir__.glob("model_files/*_model")]

        # Model-independent terms
        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.RV_MW = device_put(jnp.array(3.1))
        self.sigma_pec = device_put(jnp.array(150 / 3e5))
        self.trunc_val = 1.2  # lower limit for RV based on pure Rayleigh Scattering
        self.ZPT = fluxcal_zpt  # Common fluxcal zero point for all bands
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
        if "RV" in params:
            self.RV_type = "global"
            self.RV = jnp.array(params["RV"])
        elif "MUR" in params:
            self.RV_type = "pop"
            self.mu_R = jnp.array(params["MUR"])
            self.sigma_R = jnp.array(params["SIGMAR"])
        # Build the model wavelengths in log space
        self.min_wave = min(self.l_knots)
        self.max_wave = max(self.l_knots)
        model_log_wave = jnp.linspace(
            jnp.log10(self.min_wave), jnp.log10(self.max_wave), self.spectrum_bins
        )
        self.model_wave = 10 ** model_log_wave
        self.dlambda = jnp.diff(self.model_wave)

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
            "hsiao_interp",
            "sn_list",
            "z_u_grid",    # CDF probability levels of the host photo-z quantiles
            "z_icdf_grid", # (N_sn, len(z_u_grid)) per-SN z at those levels, or None
        ):
            setattr(self, attr, None)

    def load_ext_rel(self, ext_rel_name: str):
        """ Changing the ext_rel attribute would not recalculate mw_ext.
        This method allows the user to change extinction relations cleanly.
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
        self.used_band_inds = jnp.array(list(self.band_dict.values()))
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
            if not Path(self.filter_yaml).exists():
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
                        path = Path(self.filter_yaml).absolute().parent / path
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
                    path = Path(self.filter_yaml).absolute().parent / path
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
        apply_dovekie_mag_shifts: bool = True,
        shift_df: pd.DataFrame | None = None,
        apply_mag_shifts: bool = False,
        apply_lam_shifts: bool = False
        ):
        if name not in self.filter_dict["filters"]:
            raise ValueError(
                f"Unrecognised bandpass name: {name}. Valid options can be found by"
                "calling SEDmodel.list_bandpasses()."
            )
        ret_dict = self.filter_dict["filters"][name]
        ret_dict["defined_mag"] = ret_dict.pop("magzero", 0)
        lam_shift = ret_dict.pop("lam_shift", 0) * int(apply_lam_shifts)
        lam, trans = np.loadtxt(ret_dict["path"]).T
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
        apply_dovekie_mag_shifts: bool = True,
        shift_file: None | str | Path = None,
        apply_mag_shifts: bool = False,
        apply_lam_shifts: bool = False
    ) -> None:
        """
        Sets up the interpolation for the band weights used for photometry as well as
        calculating the zero points for each band. This code is partly based off
        ParSNiP from Boone+21.

        Parameters
        ----------
        apply_dovekie_mag_shifts :
            Boolean indicating whether to add the mag_update and mag_cal elements from
            the bayesn/bayesn-filters/filters.yaml file to each defined magnitude.
        shift_file :
            If not None, then a path to a csv file with columns
                BAND : strings
                    bandpass names in BayeSN convention
                MAG_SHIFT : scalar
                    Magnitude value to add to defined magnitude (zero-point)..
                LAM_SHIFT : scalar
                    Angstrom value to add to the wavelengths of the transmission fn.
            These values override the mag_shift and lam_shift values taken from
            bayesn/bayesn-filters/filters.yaml.
        apply_mag_shifts :
            Whether the defined magnitude should be shifted.
        apply_lam_shifts :
            Whether the transmission functions should be shifted.
        """
        if shift_file is not None:
            if not Path(shift_file).exists():
                raise FileNotFoundError(f'Specified shift file {shift_file} does not exist')
            shift_file = pd.read_csv(shift_file, comment='#')

        def ab_standard_flam(l):  # Can just use analytic function for AB spectrum
            f = (const.c.to("AA/s").value / 1e23) * (l**-2) * 10 ** (-48.6 / 2.5) * 1e23
            return f

        dlambda = jnp.diff(self.hires_wave)
        dlambda = jnp.r_[dlambda, dlambda[-1]]
        # Load filters------------------------------
        # If not working with lam_shifts, the transmission functions can be
        # pre-processed a bit. Thus, band_weights and band_weights_shift are both
        # calculated and stored in band_interpolate_weights(_shift).
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
                shift_df=shift_file,
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
        dovekie_inds = jnp.array([self.band_dict[band] for band in self.dovekie_labels if band in self.band_dict])
        for dov_ind_1, ind_1 in enumerate(dovekie_inds):
            for dov_ind_2, ind_2 in enumerate(dovekie_inds):
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
        obs_frame_wave = self.model_wave[None,:,None] * (1+redshifts)[:,None,None] + lam_shifts[None, None, :]
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
    def parse_args(self, args: dict, cmd_args: dict | argparse.Namespace, verbose: bool = True) -> dict:
        """
        Parameters
        ----------
        args:
            Arguments from input yaml file before command line overrides,
            defines model wavelength range and data set to load.
        cmd_args:
            dict-like of command line arguments, which overrides yaml file if specified
        """
        args = self._cmd_arg_overrides(args, cmd_args)
        args.pop("CONFIG", None)
        args.pop("config", None)
        args = self._parse_mode(args)
        self.RV_type = args["rv_type"] = self._get_rv_type(args, verbose=True)

        # Print out when relevant parameters are being assigned their default values.
        # p2 is only relevant if substr is in p1.
        for substr, p1, p2 in zip(
            *np.array([
                ("split", "mode", "M_split"),
                ("pop", "rv_type", "mu_R"),
                ("pop", "rv_type", "sigma_R"),
                ("pop", "rv_type", "mu_R_min"),
                ("pop", "rv_type", "mu_R_max"),
                ("pop", "rv_type", "sigma_sigma_R"),
                ("uniform", "rv_type", "uniform_RV_min"),
                ("uniform", "rv_type", "uniform_RV_max"),
                ("True", "vary_redshift", "tau_z_min"),
                ("True", "vary_redshift", "tau_z_max"),
            ]).T
        ):
            comparison = args.get(p1)
            if not verbose or comparison is None or substr not in str(comparison) or p2 in args:
                continue
            print(
                f"{p1} is {comparison}, but {p2} is not in the input args. "
                f"Setting {p2} to its default of {default_kwargs[p2]}."
            )

        for param, default in default_kwargs.items():
            args[param] = args.pop(param, default)

        # Defaults based on model values.
        for key in ("RV", "mu_R"):
            if args[key] == "default":
                args[key] = float(getattr(self, "RV", 3))
        for key in ("l_knots", "tau_knots"):
            if args[key] == "default":
                args[key] = args.get(key, getattr(self, key).tolist())

        # The VI fitting method uses a modified exponential for AV.
        if args.get("fit_method") == "vi":
            args["AV_dist"] = zltn.My_Exponential

        if args["outputdir"] == "default":
            args["outputdir"] = "."
        args["outputdir"] = Path(args["outputdir"]).absolute()
        args["photoz"] = args.get("photoz", False)
        args["num_zltn_iter"] = args.get("num_zltn_iter", 4000 if args["photoz"] else 1500)
        if args["keep_list"] is not None:
            keep_list = pd.read_csv(args["keep_list"], comment="#", sep=r"\s+")
            if keep_list.shape[1] == 1:
                keep_list = pd.read_csv(args["keep_list"], header=None)[0].astype(str).values
            else:
                if "CID" in keep_list.columns:
                    keep_list = keep_list.CID.values
                elif "SNID" in keep_list.columns:
                    keep_list = keep_list.SNID.values
            args["SNID_keep_list"] = keep_list.astype(str)
        else:
            args["SNID_keep_list"] = None

        for key in {"mode", "fit_method", "laplace_method", "lm_solver"}:
            args[key] = args[key].lower()
        pdp = args.get("private_data_path", [])
        args["private_data_path"] = [pdp] if isinstance(pdp, str) else pdp
        if args["jobsplit"] is not None:
            args["snana"] = True
        else:
            args["jobsplit"] = [1, 1]
            args["snana"] = False
        args["jobid"] = args["jobsplit"][0]
        args["njobtot"] = args["jobsplit"][1] * args["sim_prescale"]

        if not (args["mode"].startswith("fit") and args["snana"]):
            try:
                if not args["outputdir"].exists():
                    args["outputdir"].mkdir()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Requested output directory does not exist and could not be created"
                )
        self._check_args_valid(args)

        if args["train_new_model"]:
            self.l_knots = device_put(np.array(args["l_knots"], dtype=float))
            KD_l = invKD(self.l_knots)
            self.J_l_T = device_put(
                spline_coeffs(self.model_wave, self.l_knots, KD_l)
            )
            self.tau_knots = device_put(np.array(args["tau_knots"], dtype=float))
            self.KD_t = device_put(invKD(self.tau_knots))
        return args

    def _cmd_arg_overrides(self, args: dict, cmd_args: dict | argparse.Namespace) -> dict:
        """
        Parameters
        ----------
        args:
            Arguments from input yaml file before command line overrides,
            defines model wavelength range and data set to load.
        cmd_args:
            dict-like of command line arguments, which overrides yaml file if specified

        Returns
        -------
        args: dict
            Original args dict with cmd_arg overrides.
        """
        # Command line overrides, if present
        if isinstance(cmd_args, dict):
            dict_cmd_args = cmd_args
        else:
            dict_cmd_args = vars(cmd_args)
        for arg in dict_cmd_args:
            if arg in ["input", "filters"]:
                continue
            arg_val = getattr(cmd_args, arg)
            if arg_val is not None:
                if arg == "map":
                    filt_map = np.loadtxt(cmd_args.map, dtype=str)
                    arg_val = {row[0]: row[1] for row in filt_map}
                args[arg] = arg_val
        return args

    def _parse_mode(self, args: dict, verbose: bool = True) -> dict:
        """
        This method interprets the mode config parameter.
        Mode was a macro for several configurations of kwargs.
        If something is in the yaml but conflicts with the mode, raise an error.
        Otherwise, populate args with the model's appropriate kwargs as if they
        were in the yaml.

        The supported modes and (TODO: their corresponding kwargs) are
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
        args:
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
                if (
                    substr == "dust"
                    and "redshift" in args["mode"]
                    and key == "vary_redshift"
                ):
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
            if args.get("rv_type", "pop") != "pop":
                raise ValueError(
                    f"mode is {mode} but rv_type is {args['rv_type']}. "
                    f"This is not consistent with typical {mode} behavior. "
                    "Consider changing the mode or setting it to 'custom'. "
                )
            args["rv_type"] = "pop"

        # RV type was previously provided through the mode parameter.
        # Check for backwards compatibility.
        supported_RV_types = ("global", "pop", "uniform")
        if "rv" not in mode:  # e.g. not fit_poprv, train_global_rv
            return args

        if "rv_type" in args and args["rv_type"].lower() not in mode:
            raise ValueError(
                f"The rv_type parameter was provided as {args['rv_type']}. "
                f"However, the mode {mode} seems to indicate that a different "
                "rv_type should be used. Please remove the inconsistency."
            )
        for RV_type in supported_RV_types:
            if RV_type not in mode:
                continue
            args["rv_type"] = RV_type
        if "rv_type" not in args:
            raise ValueError(
                f"rv_type was not in the input yaml, but 'rv' is in the mode. However, "
                f"none of the substrings {supported_RV_types} are in {args['mode']}. "
                f"Please provide one of the supported rv_type in the input yaml."
            )
        if verbose:
            print(
                f"The rv_type was inferred as {args['rv_type']} from the mode "
                f"{args['mode']}. Including rv_type is supported for backwards "
                "compatibility, but in the future please consider including the "
                "rv_type parameter in the input yaml file."
            )
        return args

    def _get_rv_type(self, args: dict, verbose: bool = True) -> str:
        """
        rv_type can be passed in the input yaml as its own keyword or as a mode.
        This method parses the args and figures out what the intended rv_type is and
        raises an error if there are ambiguous signs.

        Parameters
        ----------
        args:
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        verbose:

        Returns
        -------
        detected_rv_type: str
            Either "pop", "global", or "uniform"
        """
        if "rv_type" in args:
            return args.pop("rv_type")
        elif (
            isinstance(args.get("RV", False), str)
            and args["RV"] in ("global", "pop", "uniform")
        ):
            return args.pop("RV")

        # If rv_type is not provided and is not in the mode, infer its value.
        uniform = ("uniform_RV_min" in args or "uniform_RV_max" in args)
        pop = ("mu_R" in args or "sigma_R" in args)
        glbl = ("RV" in args)
        if uniform + pop + glbl > 1:
            err = (
                "rv_type is not specified, and the rv_type cannot be inferred from the"
                " other arguments. "
            )
            if uniform:
                err += "uniform_RV_min/max suggest uniform. "
            if pop:
                err += "mu_R/sigma_R suggest pop. "
            if glbl:
                err += "RV suggests global. "
            err += "Please specify rv_type or provide only one set of arguments."
            raise ValueError(err)
        elif uniform:
            detected_RV_type = "uniform"
            if verbose:
                print("Inferring uniform RV based on uniform_RV_min/max.")
        elif pop:
            detected_RV_type = "pop"
            if verbose:
                print("Inferring pop RV based on mu_R/sigma_R.")
        elif glbl:
            detected_RV_type = "global"
            if verbose:
                print("Inferring global RV based on RV.")
        else:
            detected_RV_type = "global"
            if verbose:
                print(f"Inferring global RV with RV={self.RV} as a default")
        return detected_RV_type

    def _check_args_valid(self, args: dict) -> None:
        """
        Validates the input args by looking for parameters that will cause problems
        downstream, e.g. unsupported data_type/fit_method.

        Parameters
        ----------
        args:
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        """
        def check_scalar(val: Any, val_name: str) -> None:
            try:
                float(val)
            except (ValueError, TypeError):
                raise TypeError(
                    f"{val_name} must be a float-like value. Instead got {val}."
                )

        # RV_type
        supported_RV_types = ("global", "pop", "uniform")
        if args["rv_type"] not in supported_RV_types:
            raise ValueError(
                f"rv_type is {args['rv_type']}, which is not a supported"
                f"option. Please set rv_type to something from {supported_RV_types}."
            )
        if args["rv_type"] == "global":
            check_scalar(args["RV"], "RV")
        elif args["rv_type"] == "pop":
            [check_scalar(args[name], name) for name in ("mu_R", "sigma_R")]
            if float(args["sigma_R"]) == 0:
                raise ValueError(
                    "sigma_R cannot be 0. Consider using rv_type: 'global'."
                )
        elif args["rv_type"] == "uniform":
            [check_scalar(args[name], name) for name in ("uniform_RV_min", "uniform_RV_max")]
            if float(args["uniform_RV_min"]) == float(args["uniform_RV_max"]):
                raise ValueError(
                    "uniform_RV_min cannot equal uniform_RV_max. "
                    "Consider using rv_type: 'global'."
                )

        if args["data_type"] not in ("flux", "mag"):
            raise ValueError(
                f"Requested data_type, {args['args']['data_type']}, is not "
                "supported. Please set data_type to either 'flux' or 'mag'."
            )
        if args["fit_method"] not in ("vi", "mcmc"):
            raise ValueError(
                f"Requested fitting method, {args['fit_method']}, is not supported. "
                "Please set fit_method to either 'mcmc' or 'vi'."
            )
        if args["laplace_method"] not in {"svi", "lm"}:
            raise ValueError(f"laplace_method must be 'svi' or 'lm', got {args['laplace_method']!r}")
        if args["lm_solver"] not in {"gn", "hvp_cg"}:
            raise ValueError(f"lm_solver must be 'gn' or 'hvp_cg', got {args['lm_solver']!r}")

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
        args:
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        """
        if args.get("version_photometry"):
            data_dir, survey_dict = utils.find_data_dir_in_SNANA(
                private_data_path=args["private_data_path"]
            ) if args["snana"] else args["version_photometry"], {}
            list_file = Path(data_dir, Path(data_dir).name+".LIST")
            ds = SNDataset.from_snana_list(
                list_file,
                data_root=data_dir,
                keep_list=args["SNID_keep_list"],
                fluxcal_zpt=self.ZPT,
                peakmjd_key=args["peakmjd_key"],
                jobid=args["jobid"],
                njobtot=args["njobtot"],
            )
            surveys = [s for s in ds.idsurvey if s is not None]
            if len(surveys) == 1:
                self.survey = surveys[0]
            else:
                self.survey = "NULL"
            self.survey_id = survey_dict.get(self.survey, 0)
        else:
            table_path = Path(args["data_root"], args["data_table"])
            data_root = args["data_root"]
            ds = SNDataset.from_table_file(
                table_path,
                data_root=data_root,
                fluxcal_zpt=self.ZPT,
                jobid=args["jobid"],
                njobtot=args["njobtot"],
            )

        ds.apply_filter_map(args["map"])
        self._set_used_bands(ds.unique_bands)
        ds.drop_bands(args["drop_bands"])
        ds.drop_by_band_lims(self.band_lim_dict, wave_min=self.model_wave[0], wave_max=self.model_wave[-1])
        ds.cut_by_phot_numeric("phase", "<=", self.tau_knots[0])
        ds.cut_by_phot_numeric("phase", ">=", self.tau_knots[-1])
        ds.cut_by_meta_numeric("z_helio", ">", args["zlim"])
        if args["keep_list"]:
            ds.keep_according_to_list(args["keep_list"])
        if args["SNID_keep_list"]:
            ds.keep_according_to_list(args["SNID_keep_list"])
        ds.apply_error_floor(args["error_floor"])

        self.ds = ds
        self.fitres_table, self.all_table = ds.make_fitres_table("version_photometry" in args, keep_dict=args["lc_cuts"])
        self.lcplot_data = ds.make_lcplot_data(args["num_lcplot"])
        sn_data, obs_data = ds.make_bayesn_data(
            data_type=args["data_type"],
            band_dict=None,  # uses 1-based order in ds.unique_bands
            N_obs_max=args.get("N_obs_max"),
            cosmo=self.cosmo,
            negative_flux_mag_val=-99,
        )
        # TODO: Eventually sn_data and obs_data should be assigned as attributes and the
        # numpyro model should look for them as arguments, but for now we can recreate
        # the old (10, N_obs_max, N_sn) array to maintain compatibility.
        # self.sn_data = device_put(jnp.array(sn_data))
        # self.obs_data = device_put(jnp.array(obs_data))
        big_data_block = np.zeros((10, obs_data.shape[1], obs_data.shape[2]))
        big_data_block[np.array([0, 1, 2, 4, 9])] = obs_data
        big_data_block[np.array([3, 5, 6, 7, 8])] = sn_data[:,None,:]*obs_data[4, :, :]
        self.data = device_put(big_data_block)

        t = self.data[0, ...]
        self.J_t = self.get_J_t(t)
        self.hsiao_interp = self.get_hsiao_interp(t)
        self.band_weights = self._calculate_band_weights(
            redshifts=self.ds.z_helio,
            ebv=self.ds.mwebv,
            lam_shifts=np.zeros(len(self.ds.unique_bands)+1),
        )

    ###############################
    ### Astronomical Quantities ###
    ###############################
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
        return self.ext_rel._get_axav(RV)

    def get_axav(self, RV: ArrayLike) -> Array:
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
        return self.ext_rel.get_axav(RV)

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
        Calculates rest-frame spectra for given parameter values

        Parameters
        ----------
        theta: ArrayLike shape (N_sn,)
            Set of theta values for each SN
        AV: ArrayLike shape (N_sn,)
            Set of host extinction values for each SN
        W0: ArrayLike shape (N_l_knots, N_tau_knots) or (N_sn, N_l_knots, N_tau_knots)
            If 2D, global W0 matrix for all SN.
            If 3D, SN-specific W0 matrix.
        W1: ArrayLike shape (N_l_knots, N_tau_knots) or (N_sn, N_l_knots, N_tau_knots)
            If 2D, global W0 matrix for all SN.
            If 3D, SN-specific W0 matrix.
        eps: ArrayLike shape (N_sn, N_l_knots, N_tau_knots)
            Set of epsilon values for each SN, describing residual colour variation
        RV: ArrayLike shape (N_sn,)
            Set of R(V) values for each SN's host-galaxy extinction
        J_t: ArrayLike shape (N_sn, N_tau_knots, N_max_epochs)
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: ArrayLike shape (3, N_max_epochs, N_sn)
            Array containing Hsiao template spectra for each t value, comprising model
            for previous day, next day and t % 1 to allow for linear interpolation.

        Returns
        -------
        model_spectra: ArrayLike shape (N_sn, N_wl, N_max_epochs)
            Matrix containing model spectra for all SNe at all time-steps

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
        A = AV[..., None] * self._get_axav(RV=RV)
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
        """ A wrapper for the low-level _get_spectra method. That method is designed to
        work fast in jax, while this method is more flexible with the shape of
        parameters it can accept.
        If the arguments are float-likes:
            return one spectrum of shape (N_wl)
        If they are 1D arrays of length N_sn:
            return an array of shape (N_sn, N_wl)
        If t is a 2D arrays of shape (N_sn, N_max_epochs) or J_t is defined:
            return an array of shape (N_sn, N_wl, N_max_epochs)

        Parameters
        ----------
        theta:
            The theta value to be used for all spectra, or an array of N values.
        AV:
            The host extinction value to be used for all spectra, or an array of
            N_sn values.
        W0:
            If None, use SEDmodel.W0 for all spectra. If 2D, the global W0 matrix
            for all spectra. If 3D, the W0 matrix for each spectra.
        W1:
            If None, use SEDmodel.W1 for all spectra. If 2D, the global W1 matrix
            for all spectra. If 3D, the W1 matrix for each spectra.
        eps:
            A term describing the variation in SED evolution not captured in the
            theta * W1 term. If 2D, a set of epsilon values to use for all SNe.
            If 3D, a set of epsilon values for each SN. Should not be None, the
            default value is only provided to match the order of _get_spectra
            while setting W0 and W1 to have defaults of None.
        t:
            If J_t is not provided, this is used to generate a J_t matrix.
            If scalar, phase to be used for all spectra. If 1D, an phase for each
            SN. If 2D, an array of up to N_max_epochs phases for each SN.
        RV:
            Global R_V value for host extinction, or an array of N_sn values.
            Should not be None, the default value is only provided to match the
            order of _get_spectra while setting W0 and W1 to have defaults of
            None.
        J_t:
            If None, this will be calculated from t. Matrix for cubic spline
            interpolation in time axis for each SN.
        hsiao_interp:
            If None, this will be calculated from t. Array containing Hsiao
            template spectra for each t value, comprising model for previous day,
            next day and t % 1 to allow for linear interpolation.

        Returns
        -------
        model_spectra: ArrayLike shape (N, N_wl, N_epochs)
            Matrix containing model spectra for all SNe at all time-steps

        """

        if eps is None:
            raise TypeError("SEDmodel.get_spectra() missing required positional arguments: 'eps'")
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
            raise ValueError("Input arguments have more dimensions than expected.")
        elif max_ndim == 1:
            lengths = np.array(
                [jnp.atleast_1d(val).shape[0] for val in params.values()]
                + [mdp.shape[0] for mdp in multi_dim_params.values() if len(mdp.shape) == 3]
            )
            max_length = int(max(lengths))
            if not all((lengths == 1) + (lengths == max_length)):
                raise ValueError("Input arguments include multiple 1D arrays with inconsistent lengths")
            params = set_vals_to_1d(params, max_length)
        else:
            params = set_vals_to_1d(params, 1)

        if J_t is not None and hsiao_interp is not None:
            N_sn, _, N_max_epochs = J_t.shape
            if hsiao_interp.shape != (3, N_max_epochs, N_sn):
                raise ValueError(
                    f"""
                    Input J_t and hsiao_interp have incompatible shapes. They should be
                    (N_sn, N_tau_knots, N_max_epochs) and (3, N_max_epochs, N_sn)
                    respectively, but the input arguments have shapes {J_t.shape} and
                    {hsiao_interp.shape}.
                    """
                )
            if max_ndim == 1 and max_length != N_sn:
                raise ValueError(
                    f"""
                    Input arguments include multiple 1D arrays with lengths {max_length}
                    which is inconsistent with J_t and hsiao_interp which expect arrays
                    of length {N_sn}.
                    """
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
            raise TypeError("""
                SEDmodel.get_spectra() requires either "t" to be defined (scalar or ArrayLike)
                or "J_t" (jax.Array of shape (N_sn, N_tau_knots, N_max_epochs) and "hsiao_interp"
                (jax.Array of shape (3, N_max_epochs, N_sn) to be defined.
                """
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
        model_spectra: Array,
        M0: ArrayLike,
        Ds: ArrayLike,
        z: ArrayLike,
        ebv: ArrayLike,
        band_indices: ArrayLike,
        mask: ArrayLike,
        weights: Array,
        mag_shift: Number | ArrayLike,
        num_batch: int,
        **kwargs: Any,
    ) -> Array:
        """
        Calculates observer-frame fluxes for given parameter values

        Parameters
        ----------
        model_spectra: ArrayLike shape (N_sn, N_wl, N_max_epochs)
            Array of SEDs indexed by SN, wavelengths, epoch.
            Produced from SEDmodel._get_spectra or its wrapper SEDmodel.get_spectra.
        M0: ArrayLike shape (N_sn,)
            Normalising constant to scale Hsiao template to correct order of magnitude. Typically fixed to -19.5
            although can be inferred separately for different bins in a mass split analysis
        Ds: ArrayLike shape (N_sn,)
            Set of distance moduli for each SN
        z: ArrayLike shape (N_sn,)
            Set of heliocentric redshifts for each SN
        ebv: ArrayLike shape (N_sn,)
            Set of MW E(B-V) values for each SN
        band_indices: ArrayLike shape (N_sn, N_max_epochs)
            Array of integers describing which filter each observation is in.
            The integers map to the values of SEDmodel.used_band_dict.
        mask: ArrayLike shape (N_sn, N_max_epochs)
            Array containing booleans describing whether each observations should
            contribute to the posterior.
        J_t: ArrayLike shape (N_sn, N_tau_knots, N_max_epochs)
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: ArrayLike shape (3, N_max_epochs, N_sn)
            Array containing Hsiao template spectra for each t value, comprising model
            for previous day, next day and t % 1 to allow for linear interpolation.
        weights: array_like shape (N_sn, N_wl, N_bandpasses)
            Array containing band weights (transmission functions) to use for photometry.
        num_batch:
            N_sn

        Returns
        -------
        model_flux: ArrayLike shape (N_max_epochs, N_sn)
            Matrix containing model fluxes for all SNe at all time-steps
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
        zps = self.used_zps[band_indices]
        model_flux *= 10 ** (-0.4 * (M0 + Ds + zps + mag_shift - self.ZPT))
        model_flux *= mask
        return model_flux


    def get_mag_batch(
        self,
        model_spectra: Array,
        M0: ArrayLike,
        Ds: ArrayLike,
        z: ArrayLike,
        ebv: ArrayLike,
        band_indices: ArrayLike,
        mask: ArrayLike,
        weights: Array,
        mag_shift: ArrayLike | None,
        num_batch: int,
        **kwargs: Any,
    ) -> Array:
        """
        Calculates observer-frame magnitudes for given parameter values
        See SEDmodel.get_flux_batch for Parameters and Return values.
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
        bands: ArrayLike,
        chains: str | ArrayLike,
        zs: ArrayLike | float,
        ebv_mws: ArrayLike | float,
        mag: bool = True,
        num_samples: int | None = None,
        num_sne: int | None = None,
        mean: bool = False
    ) -> Array:
        """
        Returns model photometry for posterior samples from BayeSN fits, which can be
        used to make light curve fit plots.

        Parameters
        ----------
        t :
            Array of phases to evaluate model photometry at
        bands :
            List of bandpasses to evaluate model photometry in. Photometry will be
        chains :
            If a str, path to file containing BayeSN fitting posterior samples you
            wish to obtain photometry for. If not a str, then the posterior samples.
        zs :
            Array of heliocentric redshifts corresponding to the SNe you are obtaining
            model fit light curves for.
        ebv_mws :
            Array containing Milky Way extincion values corresponding to the SNe you
            are obtaining model fit light curves for.
        mag :
            Boolean to specify whether you want magnitude or flux data. If True,
            magnitudes will be returned. If False, flux densities (f_lambda) will be
            returned. Default to True i.e. mag data.
        num_samples :
            An optional keyword argument to specify the number of posterior samples
            you wish to obtain photometry for. Might be useful in testing if you are
            looking at lots of SNe, as otherwise this function will take a while to
            generate e.g. photometry for 1000 posterior samples across 1000 SNe.
            Default to None, meaning that photometry will be calculated for all
            posterior samples in chains provided.
        mean :
            If True, generate only one flux time-series for each SN using the mean
            values for SN parameters.

        Returns
        -------
        flux_grid : Array shape (N_sn, num_samples, len(bands), len(t))
            Array containing photometry for all SNe, posterior samples, bands and
            phases requested.
        """
        if type(chains) == str:
            with open(chains, "rb") as file:
                chains = pickle.load(file)

        if num_sne is None:
            num_sne = chains["theta"].shape[2]
        if num_samples is None:
            num_samples = chains["theta"].shape[0] * chains["theta"].shape[1]

        if np.isscalar(zs):
            zs = np.array([zs])
        if np.isscalar(ebv_mws):
            ebv_mws = np.array([ebv_mws])

        if mean:
            num_samples = 1

        band_list = isinstance(bands[0], list)
        if band_list:
            max_bands = np.max([len(b) for b in bands])
        else:
            max_bands = len(bands)
        if self.band_weights is None:
            self.band_weights = self._calculate_band_weights(zs, ebv_mws)

        flux_grid = jnp.zeros((num_sne, num_samples, max_bands, len(t)))
        print("Getting best fit light curves from chains...")
        for i in tqdm(np.arange(num_sne)):
            if band_list:
                fit_bands = bands[i]
            else:
                fit_bands = bands
            theta = chains["theta"][..., i].flatten(order="F")
            AV = chains["AV"][..., i].flatten(order="F")
            tmax = chains["tmax"][..., i].flatten(order="F")
            if "RV" in chains:
                RV = chains["RV"][..., i].flatten(order="F")
            else:
                RV = None
            if "lam_shift" in chains:
                lam_shift = chains["lam_shift"][..., i].flatten(order="F")
            else:
                # Not None because simulate_light_curve interprets constants as a value
                # to use whereas None leads to sampling from priors.
                lam_shift = 0
            if "mag_shift" in chains:
                mag_shift = chains["mag_shift"][..., i].flatten(order="F")
            else:
                # See above lam_shift comment.
                mag_shift = 0
            mu = chains["mu"][..., i].flatten(order="F")
            eps = chains["eps"][..., i]
            eps = eps.reshape((eps.shape[0] * eps.shape[1], eps.shape[2]), order="F")
            eps = eps.reshape(
                (eps.shape[0], self.l_knots.shape[0] - 2, self.tau_knots.shape[0]),
                order="F",
            )
            eps_full = jnp.zeros(
                (eps.shape[0], self.l_knots.shape[0], self.tau_knots.shape[0])
            )
            eps = eps_full.at[:, 1:-1, :].set(eps)
            del_M = chains["delM"][..., i].flatten(order="F")

            theta, AV, mu, eps, del_M, tmax = (
                theta[:num_samples],
                AV[:num_samples],
                mu[:num_samples],
                eps[:num_samples],
                del_M[:num_samples],
                tmax[:num_samples],
            )
            if "RV" in chains:
                RV = RV[:num_samples]
            if "lam_shift" in chains:
                lam_shift = lam_shift[:num_samples]
            if "mag_shift" in chains:
                mag_shift = mag_shift[:num_samples]
            if mean:
                theta, AV, mu, eps, del_M, tmax = (
                    theta.mean()[None],
                    AV.mean()[None],
                    mu.mean()[None],
                    eps.mean(axis=0)[None],
                    del_M.mean()[None],
                    tmax.mean()[None],
                )
                if "RV" in chains:
                    RV = RV.mean()[None]
                if "lam_shift" in chains:
                    lam_shift = lam_shift.mean()[None]
                if "mag_shift" in chains:
                    mag_shift = mag_shift.mean()[None]

            lc, lc_err, params = self.simulate_light_curve(
                t,
                theta.shape[0],
                fit_bands,
                theta=theta,
                AV=AV,
                mu=mu,
                tmax=tmax,
                del_M=del_M,
                eps=eps,
                lam_shift=lam_shift,
                mag_shift=mag_shift,
                RV=RV,
                z=zs[i],
                write_to_files=False,
                ebv_mw=ebv_mws[i],
                yerr=0,
                mag=mag,
                band_weights=self.band_weights[i : i + 1],
            )
            lc = lc.T
            lc = lc.reshape(num_samples, len(fit_bands), len(t))
            flux_grid = flux_grid.at[i, :, : len(fit_bands), :].set(lc)

        return flux_grid
    #########################
    ### Numpyro Modelling ###
    #########################
    def _model(
        self,
        obs: ArrayLike,
        weights: ArrayLike,
        train_new_model: bool = False,
        infer_dust_properties: bool = False,
        vary_redshift: bool = False,
        fix_tmax: bool = True,
        vary_filter_shifts: bool = True,
        vary_offsets: bool = True,
        M_split: float = 10,
        split_variant: str | None = None,
        data_type: str = "flux",
        photoz: bool = False,
        **kwargs: Any,
    ) -> None:
        # TODO: Split functionality by primary use cases (fitting / training)
        """
        Modular numpyro sampling functions are defined and organized based on common
        use cases. The input kwargs are then parsed and the appropriate functions are
        called.

        Parameters
        ----------
        obs: ArrayLike shape (10, N_max_epochs, N_sn)
            Data to fit, produced and attached to SEDmodel by process_dataset
            The first dimension indexes 10 parameters, they are
                phase: scalars
                flux or mag: scalars
                flux_err or mag_err: positive scalars
                host-galaxy mass: positive scalars
                band_indices: integers
                redshift: scalars
                    Heliocentric redshift
                redshift_error: positive scalars
                muhat: scalars
                MWEBV: scalars
                mask: bool
        weights: ArrayLike shape (N_sn, N_wl, N_bandpasses)
            Band weights based on filter responses and MW extinction curves for
            numerical flux integrals. Produced by SEDmodel._calculate_band_weights.
        train_new_model:
            If True, draw new samples for W0, W1, and L_Sigma.
            If False, return the model's pre-computed values.
        infer_dust_properties:
            If True, samples sigma0, tauA.
            If True and SEDmodel.RV_type is "pop", also samples mu_R, sigma_R
            If True and vary_redshift is True, also samples mu_z_grad and tau_z_grad.
            If False, uses pre-computed model attributes.
        vary_redshift:
            If True and infer_dust_properties is True, samples mu_z_grad and tau_z_grad.
            These parameters describe a linear trend correlation between redshift and
            mu_R and tauA.
        fix_tmax:
            If True, uses the peak_mjd values taken from the data as tmax.
            If False, samples tmax for each SN.
        vary_filter_shifts:
            If True, samples wavelength shifts (Angstroms) for each filter.
            If False, treats all shifts as 0 Angstroms.
        vary_offsets:
            If True, samples the magnitude shifts in each bandpasses' zero-point.
            If False, treats all shifts as 0 mag.
        M_split:
            If split_variant is not None, the high- and low-mass populations will be
            determined by whether each host-galaxy mass is greater or less than M_split.
            Unitless since host-galaxy mass is log_10(M_stellar/M_sun).
        split_variant:
            If not None, it can either be "split_mag" or "split_sed".
            If not None, the high- and low-mass galaxy populations will have their dust
            parameters sampled independently (e.g. sigma0, tauA, mu_R, sigma_R)
            If "split_mag", samples shifts to M0 for the high- and low-mass galaxy populations.
            If "split_sed", samples shifts to W0 for the high- and low-mass galaxy populations.
        data_type:
            Either "flux", or "mag", indicating whether to treat the input data as
            fluxes or magnitudes.

        Recognized Keyword Arguments
        ----------------------------
        fix_theta: default None
            If not None, float(fix_theta) will be used as the theta value for all SNe.
        fix_AV: default None
            If not None, float(fix_AV) will be used as the AV value for all SNe.
        AV_val: default=0
            If fix_AV is True, AV_val will be used for all SNe.
        muhat_err: default None
            The error on muhat, which is cosmological distance inferred from the
            fiducial cosmology in SEDmodel.cosmo and cosmological redshift.
            If scalar, use the input value for all SNe.
            If None, calculate muhat_err for each SN from their redshift,
            redshift error, and the peculiar velocity from SEDmodel.sigma_pec.
        fix_dist_limit: default 0.08
            The redshift limit below which errors on Ds are calculated from muhat_err
            and sigma0, and above which errors are forced to be fix_dist_Ds_err.
            The motivation is to relax the constraints on Ds at redshifts where
            cosmological effects may play a non-trivial role.
        fix_dist_Ds_err: default 5
            The Ds_err value to assume for SNe at redshifts above fix_dist_limit.

        If SEDmodel.rv_type == "global":
            Each SN's RV value RV_s = RV
            RV: default SEDmodel.RV if defined or 3 if not.
                The total-to-selective extinction used for host-galaxy extinction.

        If SEDmodel.rv_type == "pop":
            Each SN's RV value is similar to RV_s ~ N(mu_R, sigma_R).
            mu_R: default 3
                The mean of the normal distribution of RV.
            sigma_R: default 0.5
                The mean of the normal distribution of RV.
            mu_R_min: default 1.2
                If infer_dust_properties is True, the input mu_R parameter is ignored
                and instead sampled as mu_R ~ U(mu_R_min, mu_R_max)
                1.2 is the value for pure Rayleigh Scattering
                (A propto wl^-4) as estimated in Draine 2003, 2003ARA&A..41..241D
            mu_R_max: default 6
                If infer_dust_properties is True, the input mu_R parameter is ignored
                and instead sampled as mu_R ~ U(mu_R_min, mu_R_max)
                Very large grains would produce extinction curves with no theoretical
                upper limit on RV, but many extinction curves use data capping out
                around RV = 6 (e.g. Cardelli 1989, Fitzpatrick 1999/2019, etc.).
            sigma_sigma_R: default 2
                If infer_dust_properties is True, the input sigma_R parameter is ignored
                and instead sampled as sigma_R ~ HalfNormal(sigma_sigma_R)

        If SEDmodel.rv_type == "uniform"
            Each SN's RV value RV_s ~ U(uniform_RV_min, uniform_RV_max).
            uniform_RV_min: default 1
                The lower bound of the uniform distribution of RV.
            uniform_RV_max: default 6
                The upper bound of the uniform distribution of RV.

        If vary_redshift is True and infer_dust_properties is True:
            Linear correlations between redshift and mu_R / tauA will be sampled as
            mu_z_grad ~ U(mu_R_min - mu_R, mu_R_max - mu_R) and
            tau_z_grad ~ U(tau_z_min, tau_z_max)
            tau_z_min: default -0.5
                The lower bound of the uniform distribution of tau_z_grad.
            tau_z_max: default 0.5
                The upper bound of the uniform distribution of tau_z_grad.
        """
        N_sn = obs.shape[2]
        if not train_new_model:
            W0, W1, L_Sigma = self.W0, self.W1, self.L_Sigma
        else:
            W0, W1, L_Sigma = self._sample_model_params()
        if infer_dust_properties:
            dust_pop, M0, W0 = self._sample_dust_hyperparams(
                split_variant=split_variant,
                vary_redshift=vary_redshift,
                mass=obs[3, 0],
                M_split=M_split,
                W0=W0,
                **kwargs,
            )
        else:
            dust_pop, M0, W0 = self._get_fixed_dust_hyperparams(
                split_variant=split_variant,
                mass=obs[3, 0],
                M_split=M_split,
                W0=W0,
                **kwargs,
            )
        lam_shift, mag_shift = 0, 0
        if vary_filter_shifts:
            lam_shift = numpyro.sample("lam_shift", dist.Normal(0, self.used_wave_sigmas))
        if vary_offsets:
            mag_shift = numpyro.sample("mag_shift", dist.MultivariateNormal(0, scale_tril=self.used_calib_chcov))
        with numpyro.plate("SNe", N_sn) as sn_index:
            band_indices = obs[4, :, sn_index].astype(int).T
            phot_mask = obs[9, :, sn_index].T.astype(bool)
            if photoz:
                z = self._sample_z(
                    obs[5, 0, sn_index],
                    obs[6, 0, sn_index],
                    sn_index=sn_index,
                    z_icdf=kwargs.get("z_icdf"),
                )
            else:
                z = obs[5, 0, sn_index]
            if vary_filter_shifts or photoz:
                # In either case, observer frame transmissions need re-calculation
                # If we decide to sample E(B-V)_MW one day, that will also require
                # re-calculation.
                weights = self._calculate_band_weights(z, obs[8, 0, sn_index], lam_shift)
            AV, RV = self._sample_split_SN_dust_params(
                dust_pop=dust_pop,
                redshift=z,
                z_obs=obs[5, 0, sn_index],
                **kwargs,
            )
            theta, eps, Ds = self._sample_SN_params(
                N_sn=N_sn,
                sn_obs=obs[..., sn_index],
                L_Sigma=L_Sigma,
                sigma0=dust_pop.sigma0,
                **kwargs,
            )

            if fix_tmax:
                hsiao_interp, J_t, tmax = self.hsiao_interp, self.J_t, None
            else:
                hsiao_interp, J_t, tmax = self._sample_SN_tmax(
                    t_all_sn=obs[0],
                    sn_index=sn_index,
                    z_obs=obs[5, 0, sn_index],
                    z_sampled=z,
                    **kwargs,
                )
            phot_epoch_spectra = self._get_spectra(theta, AV, W0, W1, eps, RV, J_t, hsiao_interp)
            if data_type == "flux":
                data_fn = self.get_flux_batch
            elif data_type == "mag":
                data_fn = self.get_mag_batch
            data = data_fn(
                model_spectra=phot_epoch_spectra,
                M0=M0,
                Ds=Ds,
                z=obs[5, 0],
                ebv=obs[8, 0],
                band_indices=band_indices,
                mask=phot_mask,
                weights=weights,
                lam_shift=lam_shift,
                mag_shift=mag_shift,
                num_batch=N_sn,
            )
            with numpyro.handlers.mask(mask=phot_mask):
                numpyro.sample(
                    f"obs",
                    dist.Normal(data, obs[2, :, sn_index].T),
                    obs=obs[1, :, sn_index].T,
                )

    def _sample_model_params(self) -> tuple[Array, Array, Array]:
        """ Sample W0, W1, and L_Sigma

        Returns
        -------
        W0: numpryo Sample if train_new_model, else jax.Array
        W1: same type W0
        L_Sigma: same type as W0
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
        global_RV: ArrayLike | None = None,
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
        phi_alpha_R, mu_z_grad, tau_z_grad, global_RV_val = [0 for _ in range(4)]
        if global_RV is not None:
            global_RV_val = global_RV

        sigma0 = numpyro.sample(f"sigma0{suffix}", dist.HalfCauchy(0.1))
        tauA = numpyro.sample(f"tauA{suffix}", dist.HalfCauchy())

        if self.RV_type == "global" and global_RV_val == 0:
            global_RV_val = numpyro.sample(f"RV{suffix}", dist.Uniform(uniform_RV_min, uniform_RV_max))
        if self.RV_type == "pop":
            mu_R = numpyro.sample(f"mu_R{suffix}", dist.Uniform(mu_R_min, mu_R_max))
            sigma_R = numpyro.sample(f"sigma_R{suffix}", dist.HalfNormal(sigma_sigma_R))
            phi_alpha_R = norm.cdf((1.2 - mu_R) / sigma_R)
        if vary_redshift:
            mu_z_grad = numpyro.sample(f"mu_grad{suffix}", dist.Uniform(mu_R_min - mu_R, mu_R_max - mu_R))
            tau_z_grad = numpyro.sample(f"tau_z_grad{suffix}", dist.Uniform(tau_z_min, tau_z_max))
        return sigma0, tauA, mu_R, sigma_R, phi_alpha_R, mu_z_grad, tau_z_grad, global_RV_val

    def _get_fixed_model_dust_params(
        self,
        mu_R: ArrayLike | None = None,
        sigma_R: ArrayLike | None = None,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array | None, Array | None, Array | None, int, int, Array]:
        """Return pre-computed model dust parameters."""
        phi_alpha_R, mu_z_grad, tau_z_grad, global_RV = [0 for _ in range(4)]
        if mu_R is not None and sigma_R is not None:
            phi_alpha_R = norm.cdf((1.2 - mu_R) / sigma_R)
        return self.sigma0, self.tauA, mu_R, sigma_R, phi_alpha_R, mu_z_grad, tau_z_grad, self.RV

    def _sample_dust_hyperparams(
        self,
        split_variant: str | None,
        vary_redshift: bool,
        mass: ArrayLike,
        M_split: float,
        W0: ArrayLike,
        **kwargs: Any,
    ) -> tuple[DustPop, Array, Array]:
        """Sample population dust parameters (and mass-split parameters if specified)."""
        HM_flag = mass > M_split
        M0 = self.M0

        suffix = "_HM" if split_variant is not None else ""
        hm_params = self._sample_model_dust_params(
            suffix=suffix,
            vary_redshift=vary_redshift,
            **kwargs,
        )
        hm_dust = DustParams(*hm_params)

        lm_dust = None
        sigma0 = hm_dust.sigma0
        if split_variant is not None:
            lm_params = self._sample_model_dust_params(
                suffix="_LM",
                vary_redshift=vary_redshift,
                **kwargs,
            )
            lm_dust = DustParams(*lm_params)
            sigma0 = HM_flag * hm_dust.sigma0 + (1 - HM_flag) * lm_dust.sigma0

        if split_variant == "split_mag":
            M_step_HM = numpyro.sample("M_step_HM", dist.Uniform(-0.2, 0.2))
            M_step_LM = numpyro.sample("M_step_LM", dist.Uniform(-0.2, 0.2))
            M0 = (
                M0 * jnp.ones_like(mass)
                + HM_flag * M_step_HM
                + (1 - HM_flag) * M_step_LM
            )

        if split_variant == "split_sed":
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

        dust_pop = DustPop(
            HM=hm_dust,
            LM=lm_dust,
            HM_flag=HM_flag,
            sigma0=sigma0,
            split_variant=split_variant,
        )
        return dust_pop, M0, W0

    def _get_fixed_dust_hyperparams(
        self,
        split_variant: str | None,
        mass: ArrayLike,
        M_split: float,
        W0: ArrayLike,
        **kwargs: Any,
    ) -> tuple[DustPop, Array, Array]:
        """Retrieve pre-computed model dust parameters (and sample mass-split steps if specified)."""
        HM_flag = mass > M_split
        M0 = self.M0

        hm_params = self._get_fixed_model_dust_params(**kwargs)
        hm_dust = DustParams(*hm_params)

        lm_dust = None
        sigma0 = hm_dust.sigma0
        if split_variant is not None:
            lm_params = self._get_fixed_model_dust_params(**kwargs)
            lm_dust = DustParams(*lm_params)
            sigma0 = HM_flag * hm_dust.sigma0 + (1 - HM_flag) * lm_dust.sigma0

        if split_variant == "split_mag":
            M_step_HM = numpyro.sample("M_step_HM", dist.Uniform(-0.2, 0.2))
            M_step_LM = numpyro.sample("M_step_LM", dist.Uniform(-0.2, 0.2))
            M0 = (
                M0 * jnp.ones_like(mass)
                + HM_flag * M_step_HM
                + (1 - HM_flag) * M_step_LM
            )

        if split_variant == "split_sed":
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

        dust_pop = DustPop(
            HM=hm_dust,
            LM=lm_dust,
            HM_flag=HM_flag,
            sigma0=sigma0,
            split_variant=split_variant,
        )
        return dust_pop, M0, W0

    def _sample_split_model_dust_params(
        self,
        split_variant: str | None,
        infer_dust_properties: bool,
        vary_redshift: bool,
        mass: ArrayLike,
        M_split: float,
        W0: ArrayLike,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Array, Array, Array]:
        """Legacy method retained for backward compatibility."""
        if infer_dust_properties:
            dust_pop, M0, W0 = self._sample_dust_hyperparams(
                split_variant=split_variant,
                vary_redshift=vary_redshift,
                mass=mass,
                M_split=M_split,
                W0=W0,
                **kwargs,
            )
        else:
            dust_pop, M0, W0 = self._get_fixed_dust_hyperparams(
                split_variant=split_variant,
                mass=mass,
                M_split=M_split,
                W0=W0,
                **kwargs,
            )
        split_kwargs = {"HM": dict(dust_pop.HM._asdict(), **kwargs)}
        if dust_pop.LM is not None:
            split_kwargs["LM"] = dict(dust_pop.LM._asdict(), **kwargs)
        split_kwargs["HM"]["sigma0"] = dust_pop.sigma0
        return split_kwargs, dust_pop.HM_flag, M0, W0

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
        dust_params: DustParams | None = None,
        tauA: float | ArrayLike | None = None,
        AV_dist: Callable[[float], Array] = dist.Exponential,
        fix_AV: float | None = None,
        global_RV: ArrayLike | None = None,
        mu_R: ArrayLike | None = None,
        sigma_R: ArrayLike | None = None,
        phi_alpha_R: ArrayLike | None = None,
        redshift: float | ArrayLike = 0,
        mu_z_grad: float = 0,
        tau_z_grad: float = 0,
        uniform_RV_min: float = 1,
        uniform_RV_max: float = 6,
        suffix: str = "",
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        """Sample AV and RV for each SN.

        Parameters
        ----------
        dust_params: DustParams, optional
            NamedTuple containing population dust parameters. If provided, overrides
            individual dust arguments.
        tauA:
            Scale factor for sampling AV ~ Exponential(1/tauA).
        AV_dist:
            The stochastic function used to draw AV as fn(1/tauA).
        fix_AV: default None
            If not None, use float(fix_AV) as the AV value for all SNe.
        global_RV: default None
            If self.RV_type == "global", this is the RV value for all SNe.
        mu_R: default None
            If self.RV_type == "pop", this is used to calculate RV.
        sigma_R: default None
            If self.RV_type == "pop", this is used to calculate RV.
        phi_alpha_R: default None
            If self.RV_type == "pop", this is used to calculate RV.
        redshift: default 0
            Needed if mu_z_grad or tau_z_grad are not 0.
        mu_z_grad: default 0
            global_RV and mu_R are increased by redshift * mu_z_grad.
        tau_z_grad: default 0
            tauA is increased by redshift * tau_z_grad.
        uniform_RV_min: default 1
            If self.RV_type == "uniform", RV ~ U(uniform_RV_min, uniform_RV_max).
        uniform_RV_max: default 6
            If self.RV_type == "uniform", RV ~ U(uniform_RV_min, uniform_RV_max).
        suffix: default ""
            This string is appended to the parameter naming scheme used by numpyro.

        Returns
        -------
            AV:
            RV:
        """
        if dust_params is not None:
            tauA = dust_params.tauA
            global_RV = dust_params.global_RV
            mu_R = dust_params.mu_R
            sigma_R = dust_params.sigma_R
            phi_alpha_R = dust_params.phi_alpha_R
            mu_z_grad = dust_params.mu_z_grad
            tau_z_grad = dust_params.tau_z_grad

        suffix = f"_{suffix}".replace("__", "_").rstrip("_")
        if fix_AV is not None:
            AV = jnp.array([float(fix_AV)])
        else:
            AV = numpyro.sample(f"AV{suffix}", AV_dist(1 / (tauA + redshift * tau_z_grad)))

        if self.RV_type == "global":
            RV = global_RV + redshift * mu_z_grad
        if self.RV_type == "pop":
            RV_tform = numpyro.sample(f"RV_tform{suffix}", dist.Uniform(0, 1))
            RV = numpyro.deterministic(
                f"RV{suffix}",
                mu_R + redshift * mu_z_grad + sigma_R * ndtri(phi_alpha_R + RV_tform * (1 - phi_alpha_R)),
            )
        elif self.RV_type == "uniform":
            RV = numpyro.sample(f"RV{suffix}", dist.Uniform(uniform_RV_min, uniform_RV_max))
        return AV, RV

    def _sample_split_SN_dust_params(
        self,
        dust_pop: DustPop | None = None,
        redshift: ArrayLike = 0,
        z_obs: ArrayLike | None = None,
        split_variant: str | None = None,
        HM_flag: ArrayLike | None = None,
        **kwargs: Any,
    ) -> tuple[Array, Array]:
        """Sample AV and RV for each SN given population dust parameters."""
        if z_obs is None:
            z_obs = redshift

        if dust_pop is not None:
            split_variant = dust_pop.split_variant
            HM_flag = dust_pop.HM_flag
            AV, RV = self._sample_SN_dust_params(
                dust_params=dust_pop.HM,
                redshift=redshift,
                suffix="HM" * (split_variant is not None),
                **kwargs,
            )
            if split_variant is not None and dust_pop.LM is not None:
                AV_LM, RV_LM = self._sample_SN_dust_params(
                    dust_params=dust_pop.LM,
                    redshift=z_obs,
                    suffix="LM",
                    **kwargs,
                )
                AV = numpyro.deterministic("AV", HM_flag * AV + (1 - HM_flag) * AV_LM)
                RV = numpyro.deterministic("RV", HM_flag * RV + (1 - HM_flag) * RV_LM)
            return AV, RV

        # Legacy fallback if dust_pop is not provided
        split_kwargs = kwargs
        AV, RV = self._sample_SN_dust_params(
            redshift=redshift,
            suffix="HM" * (split_variant is not None),
            **split_kwargs.get("HM", {}),
        )
        if split_variant is not None:
            AV_LM, RV_LM = self._sample_SN_dust_params(
                redshift=z_obs,
                suffix="LM",
                **split_kwargs.get("LM", {}),
            )
            AV = numpyro.deterministic("AV", HM_flag * AV + (1 - HM_flag) * AV_LM)
            RV = numpyro.deterministic("RV", HM_flag * RV + (1 - HM_flag) * RV_LM)
        return AV, RV

    def _sample_SN_params(
        self,
        N_sn: int,
        sn_obs: ArrayLike,
        L_Sigma: ArrayLike,
        sigma0: float | ArrayLike,
        fix_theta: float | None = None,
        muhat_err: float | None = None,
        fix_dist_limit: float = 0.08,
        fix_dist_Ds_err: float = 5,
        **kwargs: Any,
    ) -> tuple[Array, Array, Array]:
        """ Sample theta, eps, Ds for each SN.

        Parameters
        ----------
        N_sn:
            Total number of SN in self.data.
            This information is required because eps is delivered as a matrix of shape
            (N_sn, N_l_knots, N_tau_knots) where the N_sn broadcasting is handled by
            calling this function within a numpyro plate.
        sn_obs: ArrayLike
            Slice of self.data of shape (10, N_max_epochs).
            The first dimension spans
                phase, flux, flux error, host-galaxy mass, band indices, host-galaxy z,
                host-galaxy z error, cosmological distance modulus, MW E(B-V), masking
            N_max_epochs is the greatest number of observations for a single SN across
            all SN in self.data.
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
        redshift, redshift_error, muhat = sn_obs[5:8, 0]

        if fix_theta is not None:
            theta = jnp.array([float(fix_theta)])
        else:
            theta = numpyro.sample(f"theta", dist.Normal(0, 1.0))

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
        eps_full = jnp.zeros(
            (N_sn, self.l_knots.shape[0], self.tau_knots.shape[0])
        )
        eps = eps_full.at[:, 1:-1, :].set(eps)

        # x * x seems more performant than x ** 2 or jnp.power(x, 2)
        # Should this be just 5?
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


    def fit_model_globalRV_noeps(self, obs, weights, fix_tmax=False, fix_theta=False, theta_val=0, fix_AV=False, AV_val=0):
        """
        Numpyro model used for fitting latent SN properties with single global RV. Will fit for time of maximum as well
        as theta, epsilon, AV and distance modulus.
        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry
        fix_tmax: Boolean, optional
            If True, tmax will be fixed to fiducial value and will not be inferred. Defaults to False
        fix_theta: Boolean, optional
            If True, theta will be fixed to value specified by theta_val. Defaults to False.
        theta_val: float or array-like, optional
            Value to fix theta to, if fix_theta=True. Defaults to 0
        fix_AV: Boolean, optional
            If True, AV will be fixed to value specified by theta_AV. Defaults to False.
        AV_val: float or array-like, optional
            Value to fix AV to, if fix_AV=True. Defaults to 0
        Returns
        -------
        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))
            theta = theta * (1 - fix_theta) + theta_val * fix_theta
            AV = numpyro.sample(f'AV', dist.Exponential(1 / self.tauA))
            AV = AV * (1 - fix_AV) + AV_val * fix_AV
            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            tmax = tmax * (1 - fix_tmax)
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = self.get_hsiao_interp(t)
            J_t = self.get_J_t(t)
            eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            band_indices = obs[-6, :, sn_index].astype(int).T
            muhat = obs[-3, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            # Ds = numpyro.sample('Ds', dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))  # Ds_err
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def fit_model_globalRV_vi(self, obs, weights, prior_only=False):
        """
        Numpyro model used for fitting SN properties assuming fixed global properties from a trained model. Will fit for
        tmax as well as theta, epsilon, Av and distance modulus. This model is slightly modified for ZLTN VI.
        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry
        prior_only: bool, optional
            If True, return after sampling all latents and skip the data-likelihood
            (``get_flux_batch`` and the obs sample). Used by ``_prior_pot`` to
            compute the prior log-density without the cost or memory footprint
            of running the model's flux computation.
        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        with numpyro.plate('SNe', sample_size) as sn_index:
            AV = numpyro.sample(f'AV', My_Exponential(1 / self.tauA))
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))
            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = self.get_hsiao_interp(t)
            J_t = self.get_J_t(t)
            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            # eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            band_indices = obs[-6, :, sn_index].astype(int).T
            muhat = obs[-3, 0, sn_index]
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)

            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))  # Ds_err
            if prior_only:
                return
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def fit_model_photoz_noeps(self, obs, weights, z_icdf=None):
        """
        Photo-z model without epsilon, for Stage-1 LM of the VI fit. Same latent set as
        fit_model_photoz (AV, theta, tmax, redshift, Ds) minus the eps residuals, with the
        redshift sampled cosmology-independently and the phase time-dilated at the sampled z.
        Mirrors fit_model_globalRV_noeps.
        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry
        z_icdf: array-like, optional
            Single per-SN host photo-z quantile row, passed through the vmap for the quantile prior
        """
        sample_size = obs.shape[-1]

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample('theta', dist.Normal(0, 1.0))
            AV = numpyro.sample('AV', dist.Exponential(1 / self.tauA))
            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            band_indices = obs[-6, :, sn_index].astype(int).T
            zhat = obs[-5, 0, sn_index]
            zhat_err = obs[-4, 0, sn_index]
            if self.z_icdf_grid is not None:  # per-SN host photo-z PDF via ICDF-reparam
                u = numpyro.sample('u', dist.Uniform(self.z_u_grid[0], self.z_u_grid[-1]))
                if z_icdf is not None:  # single per-SN row passed in (VI vmaps over SNe)
                    z = numpyro.deterministic('z', jnp.interp(u, self.z_u_grid, z_icdf))
                else:  # MCMC: one plate over all SNe, index the shared table
                    z = numpyro.deterministic('z', jax.vmap(jnp.interp, in_axes=(0, None, 0))(
                        u, self.z_u_grid, self.z_icdf_grid[sn_index]))
            else:  # Gaussian catalog prior
                ztform = numpyro.sample('ztform', dist.Normal(0, 1))
                z = numpyro.deterministic('z', zhat + zhat_err * ztform)
            # Rest-frame phase from the observer frame at the sampled z (time dilation)
            t = obs[0, ...] * (1 + zhat) / (1 + z) - tmax[None, sn_index]
            hsiao_interp = self.get_hsiao_interp(t)
            J_t = self.get_J_t(t)
            eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            muhat = obs[-3, 0, sn_index]
            weights = self._calculate_band_weights(z, self.ebv_mw, lam_shifts=0)
            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample('obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def fit_model_photoz_vi(self, obs, weights, z_icdf=None, prior_only=False):
        """
        Photo-z model modified for ZLTN VI: AV sampled first with real support so the guide's
        first-positive dimension handles it, and no fix_* pins.
        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry
        z_icdf: array-like, optional
            Single per-SN host photo-z quantile row, passed through the vmap for the quantile prior
        prior_only: bool, optional
            If True, return after sampling all latents and skip the data-likelihood (the band-weight
            recompute, get_flux_batch and the obs sample). Used by _prior_pot in the LM Stage-2 solve.
        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        with numpyro.plate('SNe', sample_size) as sn_index:
            AV = numpyro.sample(f'AV', My_Exponential(1 / self.tauA))
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))
            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            zhat = obs[-5, 0, sn_index]
            zhat_err = obs[-4, 0, sn_index]
            if self.z_icdf_grid is not None:  # per-SN host photo-z PDF via ICDF-reparam
                u = numpyro.sample('u', dist.Uniform(self.z_u_grid[0], self.z_u_grid[-1]))
                if z_icdf is not None:  # single per-SN row passed in (VI vmaps over SNe)
                    z = numpyro.deterministic('z', jnp.interp(u, self.z_u_grid, z_icdf))
                else:  # MCMC: one plate over all SNe, index the shared table
                    z = numpyro.deterministic('z', jax.vmap(jnp.interp, in_axes=(0, None, 0))(
                        u, self.z_u_grid, self.z_icdf_grid[sn_index]))
            else:  # Gaussian catalog prior
                ztform = numpyro.sample('ztform', dist.Normal(0, 1))
                z = numpyro.deterministic('z', zhat + zhat_err * ztform)
            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            muhat = obs[-3, 0, sn_index]
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            if prior_only:
                return
            band_indices = obs[-6, :, sn_index].astype(int).T
            # Rest-frame phase from the observer frame at the sampled z (time dilation)
            t = obs[0, ...] * (1 + zhat) / (1 + z) - tmax[None, sn_index]
            hsiao_interp = self.get_hsiao_interp(t)
            J_t = self.get_J_t(t)
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            weights = self._calculate_band_weights(z, self.ebv_mw, lam_shifts=0)
            mask = obs[-1, :, sn_index].T.astype(bool)
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def initial_guess(self, args: dict, reference_model: str | Path = "T21_model") -> dict[str, Array]:
        """
        Sets initialisation for training chains, using some global parameter values
        from previous models. W0 and W1 matrices are interpolated to match wavelength
        knots of new model, and set to zero beyond the time range that the reference
        model is defined for. Note that unlike Stan, in numpyro we cannot set each
        chain's initialisation separately.

        Parameters
        ----------
        args:
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        reference_model:
            Previously-trained model to be used to set initialisation, defaults to T21.

        Returns
        -------
        param_init:
            Dictionary containing initial values to be used
        """
        # Set hyperparameter initialisations
        built_in_models = [f.name for f in self.__root_dir__.glob("model_files/*_model")]
        if Path(reference_model).exists():
            print(f"Using custom model at {reference_model} to initialise chains")
            with open(reference_model, "r") as file:
                params = yaml.load(file)
        elif reference_model in built_in_models:
            print(f"Loading built-in model {reference_model} to initialise chains")
            with open(
                self.__root_dir__ / "model_files" / reference_model / "BAYESN.YAML",
                "r",
            ) as file:
                params = yaml.load(file)
        else:
            raise ValueError(
                "Invalid initialisation method, please choose either 'median' or 'sample', or choose "
                "either one of the built-in models or a custom model to base the hyperparmeter "
                "initialisation on"
            )
        W0_init = params["W0"]
        l_knots = params["L_KNOTS"]
        tau_knots = params["TAU_KNOTS"]
        W1_init = params["W1"]
        RV_init, tauA_init = params["RV"], params["TAUA"]

        # Interpolate to match new wavelength knots
        W0_init = interp1d(
            l_knots, W0_init, kind="cubic", axis=0, fill_value=0, bounds_error=False
        )(self.l_knots)
        W1_init = interp1d(
            l_knots, W1_init, kind="cubic", axis=0, fill_value=0, bounds_error=False
        )(self.l_knots)

        # Interpolate to match new time knots
        W0_init = interp1d(
            tau_knots, W0_init, kind="linear", axis=1, fill_value=0, bounds_error=False
        )(self.tau_knots)
        W1_init = interp1d(
            tau_knots, W1_init, kind="linear", axis=1, fill_value=0, bounds_error=False
        )(self.tau_knots)

        W0_init = W0_init.flatten(order="F")
        W1_init = W1_init.flatten(order="F")

        sigma0_init = 0.1
        sigmaepsilon_init = 0.1 * jnp.ones(self.N_knots_sig)
        L_Omega_init = jnp.eye(self.N_knots_sig)

        N_sn = self.data.shape[-1]

        # Prepare initial guesses
        param_init = {}
        tauA_ = tauA_init + np.random.normal(0, 0.01)
        while tauA_ < 0:
            tauA_ = tauA_init + np.random.normal(0, 0.01)
        sigma0_ = sigma0_init + np.random.normal(0, 0.01)
        param_init["W0"] = jnp.array(
            W0_init + np.random.normal(0, 0.01, W0_init.shape[0])
        )
        param_init["W1"] = jnp.array(
            W1_init + np.random.normal(0, 0.01, W1_init.shape[0])
        )
        if args["rv_type"] == "pop":
            param_init["mu_R"] = jnp.array(3.0)
            param_init["sigma_R"] = jnp.array(0.5)
            param_init["RV_tform"] = jnp.array(
                np.random.uniform(0, 1, self.data.shape[-1])
            )
        else:
            param_init["RV"] = jnp.array(3.0)
        param_init["tauA_tform"] = jnp.arctan(tauA_ / 1.0)
        param_init["sigma0_tform"] = jnp.arctan(sigma0_ / 0.1)
        param_init["sigma0"] = jnp.array(sigma0_)
        param_init["theta"] = jnp.array(np.random.normal(0, 1, N_sn))
        param_init["AV"] = jnp.array(np.random.exponential(tauA_, N_sn))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon_init), L_Omega_init)

        param_init["epsilon_tform"] = jnp.matmul(
            np.linalg.inv(L_Sigma), np.random.normal(0, 1, (self.N_knots_sig, N_sn))
        )
        param_init["epsilon"] = np.random.normal(0, 1, (N_sn, self.N_knots_sig))
        param_init["sigmaepsilon_tform"] = jnp.arctan(
            sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape) / 1.0
        )
        param_init["sigmaepsilon"] = sigmaepsilon_init + np.random.normal(
            0, 0.01, sigmaepsilon_init.shape
        )
        param_init["L_Omega"] = jnp.array(L_Omega_init)

        param_init["Ds_tform"] = jnp.array(np.random.normal(np.zeros_like(self.data[-3, 0, :]), 1))
        param_init["Ds"] = jnp.array(np.random.normal(self.data[-3, 0, :], sigma0_))

        param_init["lam_shift"] = jnp.zeros(self.band_weights.shape[-1])
        param_init["mag_shift"] = jnp.zeros(self.band_weights.shape[-1] - 1) + 0.005

        return param_init

    def run(self, args: dict, cmd_args: Any) -> None:
        """
        Main method to run BayeSN. The input yaml file allows for customisation of the
        sampling configuration and model via keyword arguments.

        Parameters
        ----------
        args:
            Arguments from input yaml file before command line overrides,
            defines model wavelength range and data set to load.
        cmd_args:
            dictionary of command line arguments, which overrides yaml file if specified
        """
        args = self.parse_args(args, cmd_args)
        if args.get("version_photometry") is not None:
            self._depr_process_dataset_version_photometry(args)
        else:
            self._depr_process_dataset_data_table(args)
        # self.process_dataset(args)

        # Set up initialisation for HMC chains
        # -------------------------
        if args["initialisation"] == "T21":
            init_strategy = init_to_value(
                values=self.initial_guess(args, reference_model="T21")
            )
        elif args["initialisation"] == "median":
            init_strategy = init_to_median()
        elif args["initialisation"] == "sample":
            init_strategy = init_to_sample()
        else:
            init_strategy = init_to_value(
                values=self.initial_guess(args, reference_model=args["initialisation"])
            )
        mode = args["mode"]
        self.RV_type = args["rv_type"]
        fitting_mode = mode.startswith("fit")
        if (args['mode'].lower() == 'fitting'
            and args['fit_method'] == 'vi'
            and args['laplace_method'] == 'lm'
        ):
            from numpyro.infer.util import initialize_model
            noeps_model = self.fit_model_photoz_noeps if args['photoz'] else self.fit_model_globalRV_noeps
            vi_model = self.fit_model_photoz_vi if args['photoz'] else self.fit_model_globalRV_vi
            self._lm_model_info = initialize_model(
                PRNGKey(0), noeps_model,
                init_strategy=init_strategy, dynamic_args=True,
                model_args=(self.data[..., 0:1], self.band_weights[0:1, ...]),
            )
            self._vi_model_info = initialize_model(
                PRNGKey(0), vi_model,
                init_strategy=init_strategy, dynamic_args=True,
                model_args=(self.data[..., 0:1], self.band_weights[0:1, ...]),
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
        print(f"self.data shape: {self.data.shape} dtype: {self.data.dtype} "
            f"size: {self.data.nbytes / 1024**2:.1f} MiB")
        print(f"self.band_weights shape: {self.band_weights.shape} dtype: {self.band_weights.dtype} "
            f"size: {self.band_weights.nbytes / 1024**2:.1f} MiB")
        print(f"Current mode: {args['mode']}")
        print("Running...")

        weights = self.band_weights
        if (
            fitting_mode and args["fit_method"] == "mcmc"
        ):  # Use vmap to vectorise over individual fitting jobs

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
                    progress_bar=True,
                )
                if args["photoz"] and self.z_icdf_grid is not None:
                    args["z_icdf"] = z_icdf
                mcmc.run(rng_key, data[..., None], weights[None, ...],  **args)
                return {
                    **mcmc.get_samples(group_by_chain=True),
                    **mcmc.get_extra_fields(group_by_chain=True),
                }

            start = timeit.default_timer()
            vmap = jax.vmap(fit_vmap_mcmc, in_axes=(2, 0, 3))
            n_sne = self.data.shape[-1]
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
            end = timeit.default_timer()
        elif fitting_mode and args["fit_method"] == "vi":

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
                    noeps_model = self.fit_model_photoz_noeps
                    vi_model = self.fit_model_photoz_vi
                    z_loc = "u" if self.z_icdf_grid is not None else "ztform"
                    sample_locs = ["AV", "theta", "tmax", z_loc, "eps_tform", "Ds"]
                    # per-SN host photo-z quantiles threaded through the vmap (empty for the Gaussian case)
                    z_kwargs = {"z_icdf": z_icdf} if self.z_icdf_grid is not None else {}
                    # z-latent starts at unconstrained 0 (Normal mean / Uniform prior midpoint)
                    extra_template = {z_loc: jnp.array([0.0])}
                else:
                    noeps_model = self.fit_model_globalRV_noeps
                    vi_model = self.fit_model_globalRV_vi
                    sample_locs = ["AV", "theta", "tmax", "eps_tform", "Ds"]
                    z_kwargs = {}
                    extra_template = {}

                warm_scale_tril = None
                if args['laplace_method'] == 'lm':
                    model_args = (data[..., None], weights[None, ...])
                    # Stage 1: Gauss-Newton LM MAP for (AV, theta, tmax, [redshift], Ds)
                    # under the Exponential prior.
                    mi = self._lm_model_info
                    pot_fn_noeps = mi.potential_fn(data[..., None], weights[None, ...], **z_kwargs)
                    post_fn_noeps = mi.postprocess_fn(data[..., None], weights[None, ...], **z_kwargs)
                    predict_fn_noeps = lambda z: _predict(noeps_model, model_args, {}, z)
                    prior_pot_fn_noeps = lambda z: _prior_pot(noeps_model, model_args, {}, z)
                    # Per-SN init: prior medians for AV/theta/tmax, this SN's muhat for Ds.
                    z_template_s1 = {
                        'AV': jnp.array([jnp.log(self.tauA * jnp.log(2.0))]),
                        'Ds': data[-3, 0:1],
                        'theta': jnp.array([0.0]),
                        'tmax': jnp.array([0.0]),
                        **extra_template,
                    }
                    noeps_median, _, z_unc_noeps = run_lm_laplace_gn(
                        predict_fn_noeps, prior_pot_fn_noeps, z_template_s1,
                        maxiter=args['lm_maxiter'],
                        lam_init=args['lm_lam_init'],
                        use_linesearch=args['lm_use_linesearch'],
                    )
                    # Stage 2: Gauss-Newton LM on the full VI model, warm-started from the Stage 1 MAP
                    vi_mi = self._vi_model_info
                    post_fn_vi = vi_mi.postprocess_fn(*model_args, **z_kwargs)
                    predict_fn = lambda z: _predict(vi_model, vi_args, z_kwargs, z)
                    prior_pot_fn = lambda z: _prior_pot(vi_model, vi_args, z_kwargs, z)
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
                    laplace_guide = AutoLaplaceApproximation(noeps_model, init_loc_fn=init_strategy)
                    svi = SVI(noeps_model, laplace_guide, optimizer, loss=Trace_ELBO(5))
                    svi_result = svi.run(PRNGKey(123), 15000, data[..., None], weights[None, ...], progress_bar=False, **z_kwargs)
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
                    vi_model,
                    init_loc_fn=init_to_value(values=new_init_dict),
                    init_scale_tril=warm_scale_tril
                )
                if args['zltn_lr_final'] == args['zltn_lr']:
                    step_size = args['zltn_lr']
                else:
                    decay_base = (args['zltn_lr_final'] / args['zltn_lr']) ** (1.0 / args['num_zltn_iter'])
                    step_size = lambda t: args['zltn_lr'] * decay_base ** t
                svi = SVI(vi_model, zltn_guide, Adam(step_size), Trace_ELBO(args["zltn_particles"]))
                svi_result = svi.run(PRNGKey(123), args["num_zltn_iter"], data[..., None], weights[None, ...], progress_bar=False, **z_kwargs)
                params, losses = svi_result.params, svi_result.losses
                predictive = Predictive(
                    zltn_guide, params=params, num_samples=4 * args["num_samples"]
                )
                samples = predictive(PRNGKey(123), data=None)
                if args["photoz"]: # surface z (a deterministic, so not in the guide samples)
                    if self.z_icdf_grid is not None:
                        samples['z'] = jnp.interp(samples['u'], self.z_u_grid, z_icdf)
                    else:
                        samples['z'] = data[-5, 0] + data[-4, 0] * samples['ztform']
                samples['eps'] = jnp.matmul(self.L_Sigma[None, ...], samples['eps_tform'].transpose(0, 2, 1))
                # samples['losses'] = losses
                return {**samples}

            start = timeit.default_timer()
            batched_map = jax.vmap(fit_vmap_vi, in_axes=(2, 0, 0))
            n_sne = self.data.shape[-1]
            # per-SN host photo-z quantiles threaded through the vmap (dummy zeros otherwise)
            if args['photoz'] and self.z_icdf_grid is not None:
                z_icdf_all = np.asarray(self.z_icdf_grid)
            else:
                z_icdf_all = np.zeros((n_sne, 1))
            batch_size = args['batch_size'] if args['batch_size'] is not None else n_sne
            n_batches = (n_sne + batch_size - 1) // batch_size

            chunks = []
            for b in tqdm(range(n_batches), desc='VI batches', disable=n_batches == 1):
                lo, hi = b * batch_size, min((b + 1) * batch_size, n_sne)
                n_real = hi - lo
                if n_real == batch_size:
                    batch_data = self.data[..., lo:hi]
                    batch_weights = self.band_weights[lo:hi]
                    batch_zicdf = z_icdf_all[lo:hi]
                else:
                    # Pad final batch by replicating SN 0; padded outputs discarded.
                    batch_data = np.empty(
                        (*self.data.shape[:-1], batch_size), dtype=self.data.dtype)
                    batch_weights = np.empty(
                        (batch_size, *self.band_weights.shape[1:]), dtype=self.band_weights.dtype)
                    batch_zicdf = np.empty((batch_size, z_icdf_all.shape[1]), dtype=z_icdf_all.dtype)
                    batch_data[..., :n_real] = self.data[..., lo:hi]
                    batch_data[..., n_real:] = self.data[..., 0:1]
                    batch_weights[:n_real] = self.band_weights[lo:hi]
                    batch_weights[n_real:] = self.band_weights[0:1]
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
                if len(val.shape) == 1:  # In case fitting only one object
                    expand_dim = True
                if expand_dim:
                    val = val[None, ...]
                if len(val.shape) == 3:
                    samples[key] = val.transpose(1, 2, 0)
                else:
                    samples[key] = val.transpose()
                samples[key] = samples[key].reshape(
                    4, args["num_samples"], *samples[key].shape[1:]
                )
            end = timeit.default_timer()
        else:
            mcmc = MCMC(
                nuts_kernel,
                num_samples=args["num_samples"],
                num_warmup=args["num_warmup"],
                num_chains=args["num_chains"],
                chain_method=args["chain_method"],
                progress_bar=True,
            )
            rng = PRNGKey(0)
            start = timeit.default_timer()

            mcmc.run(
                rng, self.data, weights, **args, extra_fields=("potential_energy",),
            )
            end = timeit.default_timer()
            mcmc.print_summary()
            samples = mcmc.get_samples(group_by_chain=True)
        print(f"Total inference runtime: {end - start:.2f} seconds")
        self.postprocess(samples, args)

    def fit_from_file(
        self,
        path: str,
        filt_map: dict = {},
        peakmjd_key: str = "SEARCH_PEAKMJD",
        print_summary: bool = True,
        file_prefix: str | None = None,
        drop_bands: list = [],
        mag: bool = False,
        photoz: bool = False,
        z_prior_err: None | float = None,
        z_pdf: None = None,
        z_quantiles: None = None,
        chain_method: str = "parallel",
        ext_rel: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, tuple]:
        """
        Method to fit light curve contained in SNANA-format text file using BayeSN model

        Parameters
        ----------
        path:
            Path to SNANA-format text file containing data to be fit
        filt_map:
            Dictionary providing mapping between filter names in file and BayeSN filters. Defaults to empty dictionary
        peakmjd_key:
            Key to be used for peak MJD in SNANA text file meta. Defaults to "SEARCH_PEAKMJD"
        print_summary:
            Specifies whether to print fit summary
        file_prefix:
            Prefix of name for output files containing summary table and MCMC samples. Default to None, in which case
            output files will not be saved and only returned for use in script.
        drop_bands:
            List of bands to be ignored during fitting. Defaults to empty list
        fix_tmax:
            If True, tmax will not be inferred and fiducial value in file meta will be fixed. Defaults to False.
        fix_theta:
            If not None, float(fix_theta) will be used as the theta value for all SNe.
        fix_AV:
            If not None, float(fix_AV) will be used as the AV value for all SNe.
        RV:
            Value to fix RV at during fitting. Defaults to False, meaning that default model RV treatment will be used.
        mu_R:
            Value of mean of RV distribution to be used during fitting. Defaults to False, meaning that default model
            RV treatment will be used. If specified, sigma_R must also be specified.
        sigma_R:
            Value of standard deviation of RV distribution. Defaults to False, meaning that default model RV treatment
            will be used.
        mag:
            Specifies whether data is mag or flux. If True, data is assumed to be mag and is automatically converted to
            flux before fitting.

        Returns
        -------
        samples:
            Dictionary containing parameter names as keys and MCMC samples as values
        sn_props:
            Tuple containing SN redshift and MW E(B-V), which can be useful to have in memory when making plots

        """
        meta, lcdata = sncosmo.read_snana_ascii(path, default_tablename="OBS")
        lcdata = lcdata["OBS"].to_pandas()

        t = lcdata.MJD.values
        flux = lcdata.FLUXCAL.values
        flux_err = lcdata.FLUXCALERR.values
        filters = lcdata.FLT.values
        peak_mjd = meta[peakmjd_key]
        z = meta["REDSHIFT_HELIO"]
        ebv_mw = meta["MWEBV"]
        zpt = meta.get('ZP_FLUXCAL', self.ZPT)  # header ZP overrides the configured default
        if zpt != self.ZPT:
            print(f'Using ZP_FLUXCAL={zpt} from data header')
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
        ebv_mw: float = 0,
        peak_mjd: float | None = None,
        filt_map: dict = {},
        print_summary: bool = True,
        file_prefix: str | None = None,
        drop_bands: list = [],
        mag: bool = False,
        photoz: bool = False,
        z_prior_err: None | float = None,
        z_pdf: None = None,
        z_quantiles: None = None,
        verbose: bool = False,
        num_samples: int = 250,
        num_warmup: int = 250,
        num_chains: int = 4,
        chain_method: str = "parallel",
        zpt: None | Number = None,
        **kwargs: Any,
    ) -> tuple[dict, tuple]:
        """
        Method to fit light curve data loaded into memory with BayeSN model

        Parameters
        ----------
        t:
            Set of MJDs/rest-frame phases for light curve data to be fit. If you pass
            MJD and also a peak_mjd, values will automatically be converted to
            rest-frame phases
        flux:
            Set of fluxes/mags for light curve data to be fit. Despite the name, you
            can use mags and if mag=True data will be automatically converted into flux
            for fitting.
        flux_err:
            Set of flux/mag errors for light curve data to be fit. Despite the name,
            you can use mags and if mag=True data will be automatically converted into
            flux for fitting.
        filters:
            Set of filters that flux/flux_err are measurements for, telling BayeSN
            which filters to use when fitting data. Must be of same length as
            flux/flux_err i.e. specify the filter for each data point individually.
        z:
            Heliocentric redshift of SN to be used when fitting
        ebv_mw:
            Milky Way E(B-V) value of SN. Defaults to 0.
        peak_mjd:
            Fiducial value for maximum MJD of SN, used to convert phases to rest-frame.
            Note that this value only needs to be rough as BayeSN will fit for the time
            of maximum. However, if you set fix_tmax=True then this will be fixed as
            the time of maximum. Defaults to None, meaning that the code will assume
            phases are already rest-frame rather than MJD and will not be converted.
        filt_map:
            Dictionary providing mapping between filter names in file and BayeSN
            filters. Defaults to empty dictionary
        print_summary:
            Specifies whether to print fit summary
        file_prefix:
            Prefix of name for output files containing summary table and MCMC samples.
            Default to None, in which case output files will not be saved and only
            returned for use in script.
        drop_bands:
            List of bands to be ignored during fitting. Defaults to empty list.
        mag:
            Specifies whether data is mag or flux.
            If True, data is assumed to be mag and is automatically converted to flux
            before fitting. This is a boolean version of the data_type str argument used
            elsewhere.
        verbose:
            Prints updates to stdout if True. Default False.
        **kwargs:
            See SEDmodel._model for a list of recognized keyword arguments.

        Returns
        -------
        samples:
            Dictionary containing parameter names as keys and MCMC samples as values
        sn_props:
            Tuple containing SN redshift and MW E(B-V), which can be useful to have in
            memory when making plots
        """
        if ext_rel is not None and self.ext_rel.name != ext_rel:
            self.ext_rel = DustExtRel(name=ext_rel)
        if isinstance(drop_bands, str):
            drop_bands = [drop_bands]
        t, flux, flux_err, filters = (
            np.array(t),
            np.array(flux),
            np.array(flux_err),
            np.array(filters),
        )
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
            else:
                z_lo, z_hi = z - 3 * z_prior_err, z + 3 * z_prior_err
            p1, p2 = t * (1 + z) / (1 + z_lo), t * (1 + z) / (1 + z_hi)
            keep = np.maximum(p1, p2) + 10 > float(self.hsiao_t[0])
        else:
            keep = (t > self.tau_knots.min()) & (t < self.tau_knots.max())
        flux, flux_err, filters, t = flux[keep], flux_err[keep], filters[keep], t[keep]

        if verbose and any(~keep):
            print(
                f"Cutting {len(t) - sum(t_mask)} data due to rest-frame phases "
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
        data = jnp.zeros((10, n_data, 1))
        data = data.at[0, :, 0].set(t)
        data = data.at[1, :, 0].set(flux)
        data = data.at[2, :, 0].set(flux_err)
        data = data.at[4, :, 0].set(band_indices)
        data = data.at[5, :, 0].set(np.full_like(t, z))
        if photoz:
            data = data.at[6, :, 0].set(np.full_like(t, z_prior_err))
        data = data.at[7, :, 0].set(np.full_like(t, self.cosmo.distmod(z).value))
        data = data.at[8, :, 0].set(np.full_like(t, ebv_mw))
        data = data.at[9, :, 0].set(np.ones_like(t))
        self.ebv_mw = data[8,0,:]

        self.band_weights = weights = self._calculate_band_weights(
            data[5, 0, :],
            self.ebv_mw,
            lam_shifts=0
        )

        kwargs["mode"] = "fitting"
        kwargs = self.parse_args(kwargs)
        self.RV_type = kwargs["rv_type"] = self._get_rv_type(kwargs)
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
            data,
            weights,
            **kwargs,
            extra_fields=("potential_energy",),
        )
        if print_summary:
            mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        if peak_mjd is not None:
            samples["peak_MJD"] = peak_mjd + samples["tmax"] * (1 + (samples["z"] if photoz else z))
        if not photoz:
            # muhat-shrinkage ties Ds to distmod(z); skip for cosmology-independent photo-z
            muhat = self.cosmo.distmod(z).value
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            samples["mu"] = np.random.normal(
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

        if kwargs["fix_tmax"]:
            samples["tmax"] = jnp.zeros_like(samples["tmax"])
        if kwargs["fix_theta"] is not None:
            samples["theta"] = jnp.ones((num_chains, num_samples, 1))*kwargs["fix_theta"]
        if kwargs["fix_AV"] is not None:
            samples["AV"] = jnp.ones((num_chains, num_samples, 1))*kwargs["fix_AV"]

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
        l_knot_1: float | int = 6200.0,
        tau_knot_0: float | int = 0.0,
        tau_knot_1: float | int = 10.0
    ) -> None:
        """
        Function to postprocess BayeSN output. Applies transformations to some
        parameters e.g. ensuring consistency for W1 and theta, as flipping the sign in
        front of W1 and theta will lead to an identical result. Saves output chains
        and calculates a fit summary.

        Parameters
        ----------
        samples:
            Output of MCMC, dictionary containing posterior samples for each parameter
            with parameter names as keys.
        args:
            Combination of arguments from input yaml file and command line overrides,
            defines model wavelength range and data set to load.
        """
        start = time.time()
        if "W1" in samples:  # If training
            with open(args["outputdir"] / "initial_chains.pkl", "wb") as file:
                pickle.dump(samples, file)
            # Sign flipping-----------------
            J_R = spline_coeffs([float(l_knot_1)], self.l_knots, invKD(self.l_knots))
            J_0 = spline_coeffs([float(tau_knot_0)], self.tau_knots, invKD(self.tau_knots))
            J_10 = spline_coeffs([float(tau_knot_1)], self.tau_knots, invKD(self.tau_knots))
            W1 = np.reshape(
                samples["W1"],
                (
                    samples["W1"].shape[0],
                    samples["W1"].shape[1],
                    self.l_knots.shape[0],
                    self.tau_knots.shape[0],
                ),
                order="F",
            )
            N_chains = W1.shape[0]
            sign = np.zeros(N_chains)
            for chain in range(N_chains):
                chain_W1 = np.mean(W1[chain, ...], axis=0)
                padded_chain_W1 = np.ones((chain_W1.shape[0]+1, chain_W1.shape[1]+1))
                padded_chain_W1[1:,1:] = chain_W1
                chain_sign = np.sign(
                    np.squeeze(np.matmul(J_R, np.matmul(padded_chain_W1, J_10.T)))
                    - np.squeeze(np.matmul(J_R, np.matmul(padded_chain_W1, J_0.T)))
                )
                sign[chain] = chain_sign
            samples["W1"] = samples["W1"] * sign[:, None, None]
            samples["theta"] = samples["theta"] * sign[:, None, None]
            # Modify W1 and theta----------------
            theta_std = np.std(samples["theta"], axis=2)
            samples["theta"] = samples["theta"] / theta_std[..., None]
            samples["W1"] = samples["W1"] * theta_std[..., None]

            # Save best fit global params to files for easy inspection and reading in------
            W0 = np.mean(samples["W0"], axis=[0, 1]).reshape(
                (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
            )
            W1 = np.mean(samples["W1"], axis=[0, 1]).reshape(
                (self.l_knots.shape[0], self.tau_knots.shape[0]), order="F"
            )

            L_Sigma = np.matmul(
                np.diag(np.mean(samples["sigmaepsilon"], axis=[0, 1])),
                np.mean(samples["L_Omega"], axis=[0, 1]),
            )
            # L_Sigma_dust = np.matmul(
            #     np.diag(np.mean(samples["sigmaepsilon_dust"], axis=[0, 1])),
            #     np.mean(samples["L_Omega_dust"], axis=[0, 1]),
            # )
            sigma0 = self.sigma0
            if "sigma0" in samples:
                sigma0 = np.mean(samples["sigma0"])
            tauA = self.tauA
            if "tauA" in samples:
                tauA = np.mean(samples["tauA"])

            yaml_data = {
                "M0": float(self.M0),
                "SIGMA0": float(sigma0),
                "TAUA": float(tauA),
                "TAU_KNOTS": self.tau_knots.tolist(),
                "L_KNOTS": self.l_knots.tolist(),
                "W0": W0.tolist(),
                "W1": W1.tolist(),
                "L_SIGMA_EPSILON": L_Sigma.tolist(),
                # "L_SIGMA_EPSILON_DUST": L_Sigma_dust.tolist(),
            }

            if args["rv_type"] == "global":
                yaml_data["RV"] = float(self.RV)
            elif args["rv_type"] == "uniform":
                yaml_data["RV"] = float(np.mean(samples.get("RV", float(self.RV))))
            elif args["rv_type"] == "pop":
                yaml_data["MUR"] = float(np.mean(samples.get("mu_R", float(self.RV))))
                yaml_data["SIGMAR"] = float(np.mean(samples["sigma_R"]))

            with open(args["outputdir"] / "bayesn.yaml", "w") as file:
                yaml.dump(yaml_data, file)

        z_HEL = self.data[5, 0, :]
        muhat = self.data[7, 0, :]

        if args["mode"].startswith("fit"):
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            if args["photoz"]:
                # Cosmology-independent: report the fitted light-curve distance Ds directly,
                # without the muhat (catalog-z distmod) shrinkage that would inject a fiducial cosmology
                samples["mu"] = samples["Ds"]
                samples["delM"] = np.zeros_like(samples["Ds"])
            else:
                samples["mu"] = np.random.normal(
                    loc=(
                        samples["Ds"] * np.power(muhat_err, 2)
                        + muhat * np.power(self.sigma0, 2)
                    ) / np.power(Ds_err, 2),
                    scale=np.sqrt(
                        (np.power(self.sigma0, 2) * np.power(muhat_err, 2))
                        / np.power(Ds_err, 2)
                    ),
                )
                samples["delM"] = samples["Ds"] - samples["mu"]
            if "tmax" in samples:  # Convert tmax samples into peak_MJD samples
                # Time dilation at the fitted z for photo-z, else the fixed catalog z
                z_dilation = samples["z"] if args["photoz"] else z_HEL[None, None, :]
                samples["peak_MJD"] = self.peak_mjds[None, None, :] + samples["tmax"] * (1 + z_dilation)

            # Compute FITPROB (must be before LCPLOT generation which corrupts self.band_weights)
            fitprob, fitchi2, ndof = self.get_fitprob(samples, batch_size=args.get("batch_size"))

            # Create lcplot file
            t = np.arange(self.tau_knots[0], self.tau_knots[-1], 2)
            if args["num_lcplot"] is None:
                num_lcplot = self.data.shape[-1]
            else:
                num_lcplot = args["num_lcplot"]

            if num_lcplot > 0:
                bands_by_cid = self.lcplot_data.groupby("CID")["FLT"].unique().to_dict()
                bands = [list(bands_by_cid.get(sn, [])) for sn in self.sn_list]
                f = self.get_flux_from_chains(
                    t,
                    bands,
                    samples,
                    self.data[5, 0, :],
                    self.data[8, 0, :],
                    num_samples=None,
                    num_sne=num_lcplot,
                    mag=False,
                    mean=not args["save_fit_errors"],
                )
                f, ferr = f.mean(axis=1), f.std(axis=1)

                self.lcplot_data["DATA_FLAG"] = 1
                z_hel = self.data[5, 0, :]
                fit_dfs = []
                for i, sn in enumerate(self.lcplot_data.CID.unique()):
                    fit_df = pd.DataFrame()
                    fit_df["MJD"] = (self.peak_mjds[i] + t * (1 + z_hel[i])).repeat(
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
                "SURVEY": self.survey,
                "IDSURVEY": int(self.survey_id),
                "NEVT_TOT": self.data.shape[-1],
                "NEVT_LC_CUTS": self.data.shape[-1],
                "NEVT_LCFIT_CUTS": int(self.data.shape[-1] - drop_count),
                "CPU_MINUTES": round(cpu_time / 60, 2),
            }
            with open(f"{args['outfile_prefix']}.YAML", "w") as file:
                yaml.dump(out_dict, file)

        if not (args["mode"].startswith("fit") and args["snana"]):
            # Save convergence data for each parameter to csv file
            summary = arviz.summary(samples)
            summary.to_csv(args["outputdir"] / "fit_summary.csv")

            with open(args["outputdir"] / "chains.pkl", "wb") as file:
                pickle.dump(samples, file)

            dump_args = copy.deepcopy(args)
            with open(args["outputdir"] / "input.yaml", "w") as file:
                if args.get("AV_dist") == dist.Exponential:
                    dump_args["AV_dist"] = "dist.Exponential"
                elif args.get("AV_dist") == zltn.My_Exponential:
                    dump_args["AV_dist"] = "zltn.My_Exponential"
                yaml.dump(dump_args, file)

    #################
    ### Utilities ###
    #################
    def get_J_t(self, t: ArrayLike, extrap: None | str | int = 1) -> tuple[Array, Array]:
        """ Convenience function for generating the matrices needed for fast
        interpolation.

        Parameters
        ----------
        t: ArrayLike shape (N_max_epochs, N_sn)
            Array of phase values used to calculate J_t and hsiao_interp
        extrap:
            integer 1, 2, or 3 that describes how to extrapolate the spline mapping
            to phases before or after the first/last phase knot. The value corresponds
            to the order of the polynomial extrapolation (default linear).

        Returns
        -------
        J_t: jax.Array shape (N_sn, N_tau_knots, N_max_epochs)
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: jax.Array shape (3, N_max_epochs, N_sn)
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation
        """
        keep_shape = t.shape
        return self.J_t_map(t.flatten(), self.tau_knots, self.KD_t, extrap).reshape((*keep_shape, self.tau_knots.shape[0]+1)).transpose(1, 2, 0)

    def get_hsiao_interp(self, t: ArrayLike) -> Array:
        return jnp.array([self.hsiao_offset + jnp.floor(t), self.hsiao_offset + jnp.ceil(t), jnp.remainder(t, 1)])

    def get_fitprob(self, samples, batch_size=None):
        """
        Compute FITPROB for each SN using posterior mean parameters.

        Uses the BayeSN joint test statistic T = chi2_data + chi2_epsilon,
        where chi2_data is the standard flux chi-squared at the posterior mean
        and chi2_epsilon = sum(eps_tform_mean^2) penalizes extreme epsilon.

        Parameters
        ----------
        samples: dict
            Posterior samples dictionary with keys including 'theta', 'AV', 'tmax', 'Ds', 'eps_tform'

        Returns
        -------
        fitprob : array (n_sne,)
            Goodness-of-fit probability for each SN
        fitchi2 : array (n_sne,)
            Joint test statistic T = chi2_data + chi2_epsilon
        ndof : array (n_sne,)
            Effective degrees of freedom: N_obs + alpha*N_eps - N_params
        """
        from scipy.stats import chi2 as chi2_dist

        n_sne = self.data.shape[-1]

        # --- 1. Posterior means ---
        theta_mean = np.array(samples['theta'].mean(axis=(0, 1)))
        AV_mean = np.array(samples['AV'].mean(axis=(0, 1)))
        tmax_mean = np.array(samples['tmax'].mean(axis=(0, 1)))
        Ds_mean = np.array(samples['Ds'].mean(axis=(0, 1)))

        # eps_tform: (n_chains, n_samples, N_knots_sig, n_sne) -> mean -> (N_knots_sig, n_sne)
        eps_tform_mean = np.array(samples['eps_tform'].mean(axis=(0, 1)))
        # Reconstruct eps via L_Sigma transform (mirrors fit_model_globalRV lines 864-869)
        eps = np.matmul(self.L_Sigma, eps_tform_mean)
        eps = eps.T
        eps = eps.reshape(
            (n_sne, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F'
        )
        eps_full = jnp.zeros((n_sne, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full = eps_full.at[:, 1:-1, :].set(eps)

        # RV: per-SN from posterior (popRV) or global scalar (globalRV)
        if 'RV' in samples:
            RV = np.array(samples['RV'].mean(axis=(0, 1)))
        else:
            RV = self.RV

        # --- 2. Rebuild J_t and hsiao_interp at posterior mean tmax ---
        obs_times = self.data[0, ...]
        if self.photoz:  # evaluate the model at the fitted redshift (band weights + time dilation)
            z_mean = np.array(samples['z'].mean(axis=(0, 1)))
            zhat = np.asarray(self.data[-5, 0, :])
            t = obs_times * (1 + zhat[None, :]) / (1 + z_mean[None, :]) - tmax_mean[None, :]
            weights = self._calculate_band_weights(z_mean, self.ebv_mw, lam_shift=0)
        else:
            t = obs_times - tmax_mean[None, :]
            weights = self.band_weights
        hsiao_interp = self.get_hsiao_interp(t)
        keep_shape = t.shape
        J_t = self.get_J_t(t)
        # --- 3. Inputs in (N_obs, n_sne) convention ---
        band_indices = self.data[4, :, :].astype(int)
        mask = self.data[9, :, :].astype(bool)

        # --- 4. Model flux — batched over SN axis to keep peak memory bounded ---
        bs = batch_size if batch_size is not None else n_sne
        rv_is_array = isinstance(RV, (np.ndarray, jnp.ndarray)) and getattr(RV, 'ndim', 0) > 0
        chunks = []
        for lo in range(0, n_sne, bs):
            hi = min(lo + bs, n_sne)
            chunk = self.get_flux_batch(
                self.M0, theta_mean[lo:hi], AV_mean[lo:hi], self.W0, self.W1,
                eps_full[lo:hi], Ds_mean[lo:hi],
                RV[lo:hi] if rv_is_array else RV,
                band_indices[:, lo:hi], mask[:, lo:hi],
                J_t[lo:hi], hsiao_interp[:, :, lo:hi], self.band_weights[lo:hi],
            )
            chunks.append(np.asarray(chunk))
        model_flux = np.concatenate(chunks, axis=-1)

        # --- 5. chi2_data ---
        obs_flux = self.data[1, :, :]
        obs_err = self.data[2, :, :]
        residuals_sq = (obs_flux - model_flux) ** 2
        chi2_per_obs = jnp.where(mask, residuals_sq / obs_err ** 2, 0.0)
        chi2_data = jnp.sum(chi2_per_obs, axis=0)

        # --- 6. chi2_epsilon: sum of squared whitened epsilon ---
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        chi2_epsilon = np.sum(eps_tform_mean ** 2, axis=0)

        # --- 7. Host photo-z prior term -2 ln p_host(z), added to the joint chi2 (matches SNANA) ---
        chi2_z = 0.0
        if self.photoz:
            if self.z_icdf_grid is None:  # Gaussian prior: pull^2 (drop the log-norm const, as SNANA does)
                zhat_err = np.asarray(self.data[-4, 0, :])
                chi2_z = ((z_mean - zhat) / zhat_err) ** 2
            else:  # quantile prior: p_host is dCDF/dz, the finite-difference slope of the quantiles
                zq, pl = np.asarray(self.z_icdf_grid), np.asarray(self.z_u_grid)
                slopes = np.diff(pl) / np.diff(zq, axis=1)  # dP/dz per quantile bin
                k = np.clip(np.sum(zq <= z_mean[:, None], axis=1) - 1, 0, pl.shape[0] - 2)
                p_host = slopes[np.arange(len(z_mean)), k]
                chi2_z = np.where(p_host > 0, -2.0 * np.log(p_host), 1000.0)

        # --- 8. Joint statistic, NDOF, FITPROB ---
        T_joint = np.array(chi2_data) + chi2_epsilon + chi2_z
        # NDOF = N_obs + alpha*N_eps - N_params.
        # Physical params (tmax, theta, AV, Ds, +RV) consume DOF from data.
        # Epsilon knots are prior-constrained, so their effective DOF is
        # between 0 (fully free) and N_knots_sig (fully prior-determined).
        # alpha=0.75 calibrated on Foundation DR1 sample.
        # Photo-z z is net-zero DOF: the host-z prior adds one prior point while the
        # floated z consumes one parameter (as SNANA does), leaving ndof unchanged.
        alpha = 0.75
        n_params = 4 + (1 if 'RV' in samples else 0)
        ndof = (np.sum(np.array(mask), axis=0) + alpha * N_knots_sig - n_params).astype(int)

        fitprob = np.where(
            (ndof > 0) & (T_joint > 0),
            chi2_dist.sf(T_joint, ndof),
            1.0
        )
        return fitprob, T_joint, ndof

    def sample_del_M(self, N: int) -> Array:
        """
        Samples grey offset del_M from model prior

        Parameters
        ----------
        N :
            Number of objects to sample for

        Returns
        -------
        del_M :
            Sampled del_M values
        """
        del_M = np.random.normal(0, self.sigma0, N)
        return del_M

    def sample_AV(self, N: int) -> Array:
        """
        Samples AV from model prior

        Parameters
        ----------
        N :
            Number of objects to sample for

        Returns
        -------
        AV :
            Sampled AV values
        """
        AV = np.random.exponential(self.tauA, N)
        return AV

    def sample_theta(self, N: int) -> Array:
        """
        Samples theta from model prior

        Parameters
        ----------
        N :
            Number of objects to sample for

        Returns
        -------
        theta :
            Sampled theta values
        """
        theta = np.random.normal(0, 1, N)
        return theta

    def sample_epsilon(self, N: int) -> Array:
        """
        Samples epsilon from model prior

        Parameters
        ----------
        N :
            Number of objects to sample for

        Returns
        -------
        eps_full :
            Sampled epsilon values
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

    def sample_lam_shift(self, *args: Any) -> Array:
        """
        Samples lam_shift from model prior. N accepted for consistency with other
        sampling functions, but it doesn't do anything.

        Returns
        -------
        lam_shift :
            Sampled lam_shift values
        """
        lam_shift = np.random.normal(0, self.used_wave_sigmas)
        return lam_shift

    def sample_mag_shift(self, *args: Any) -> Array:
        """
        Samples mag_shift from model prior. N accepted for consistency with other
        sampling functions, but it doesn't do anything.

        Returns
        -------
        mag_shift :
            Sampled mag_shift values
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
    ) -> tuple[Array, Array, dict]:
        """
        Simulates spectra for given parameter values in the observer-frame. If parameter values are not set, model
        priors will be sampled.

        Parameters
        ----------
        t:
            Set of t values to simulate spectra at
        N:
            Number of separate objects to simulate spectra for
        dl:
            Wavelength spacing for simulated spectra in rest-frame. Default is 10 AA
        z:
            Redshift to simulate spectra at, affecting observer-frame wavelengths and reducing spectra by factor of
            (1+z). Defaults to 0. If passing an ArrayLike object, there must be a corresponding value for each of the N
            simulated objects. If a float is passed, the same redshift will be used for all objects.
        mu:
            Distance modulus to simulate spectra at. Defaults to 0. If passing an ArrayLike object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. If set to "z", distance moduli corresponding to the redshift values passed in the default
            model cosmology will be used.
        ebv_mw:
            Milky Way E(B-V) values for simulated spectra. Defaults to 0. If passing an ArrayLike object, there must be
            a corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects.
        RV:
            RV values for host extinction curves for simulated spectra. If passing an ArrayLike object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the global RV value for the BayeSN model loaded when
            initialising SEDmodel will be used.
        logM:
            Currently unused, will be implemented when split models are included
        del_M:
            Grey offset del_M value to be used for each SN. If passing an ArrayLike object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        AV:
            Host extinction RV value to be used for each SN. If passing an ArrayLike object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        theta:
            Theta value to be used for each SN. If passing an ArrayLike object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        eps:
            Epsilon values to be used for each SN. If passing a 2d array, this must be of shape (l_knots, tau_knots)
            and will be used for each SN generated. If passing a 3d array, this must be of shape (N, l_knots, tau_knots)
            and provide an epsilon value for each generated SN. You can also pass 0, in which case an array of zeros of
            shape (N, l_knots, tau_knots) will be used and epsilon is effectively turned off. Defaults to None, in which
            case the prior distribution will be sampled for each object.

        Returns
        -------
        l_o:
            Array of observer-frame wavelength values
        spectra:
            Array of simulated spectra
        param_dict:
            Dictionary of corresponding parameter values for each simulated object
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

        self.model_wave = l_r
        KD_l = invKD(self.l_knots)
        self.J_l_T = device_put(spline_coeffs(self.model_wave, self.l_knots, KD_l))
        self._load_hsiao_template()

        t = jnp.array(t)
        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = self.get_hsiao_interp(t)
        J_t = self.get_J_t(t)
        spectra = self._get_spectra(
            theta, AV, self.W0, self.W1, eps, RV, J_t, hsiao_interp
        )
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
        Simulates light curves from the BayeSN model in either mag or flux space
        and saves them to SNANA-format text files if requested

        Parameters
        ----------
        t :
            Set of t values to simulate spectra at.
            If len(t) == len(bands), will assume the t values correspond to the bands.
            Otherwise, will simulate photometry at each value of t for each band.
        N :
            Number of separate objects to simulate spectra for.
        bands :
            List of bands in which to simulate photometry.
            If len(t) == len(bands), will assume the t values correspond to the bands.
            Otherwise, will simulate photometry at each value of t for each band.
        yerr :
            Uncertainties for each data point, simulated light curves will be
            randomised assuming a Gaussian uncertainty around the true values.
            If a Scalar, meaning that the same value will be used for each data point.
            If a 1D ArrayLike of length equal to each light curve, these values will
            be used for each simulated light curve.
            If a 2D ArrayLike of shape (N, light curve length) allowing you to specify
            each individual error.
            With the default of 0, exact model photometry will be returned.
        err_type :
            Specifies which type of error you are passing, either "mag" or "flux".
            The errors will be coerced to match the data type specified by "mag".
        z :
            Redshift to simulate spectra at, affecting observer-frame wavelengths and
            reducing spectra by factor of (1+z).
            If a float is passed, the same redshift will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        zerr :
            Error on z. Only needed when saving to SNANA-format light curve files.
        mu :
            Distance modulus to simulate spectra at.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
            If set to "z", distance moduli corresponding to the redshift values passed
            in the default model cosmology will be used. Technically these are
            heliocentric redshifts rather than Hubble diagram redshifts so it's not
            perfect, but can be useful sometimes.
        ebv_mw :
            Milky Way E(B-V) values for simulated spectra.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        RV :
            RV values for host extinction curves for simulated spectra.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
            Defaults to None, in which case the global RV value for the BayeSN model
            loaded when initialising SEDmodel will be used.
        logM :
            Currently unused, will be implemented when split models are included
        tmax :
            Time of maximum in rest-frame days, useful for plotting light curve fits
            with free tmax.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        del_M :
            Grey offset del_M value to be used for each SN.
            If None, the prior distribution will be sampled for each object.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        AV :
            Host extinction RV value to be used for each SN.
            If None, the prior distribution will be sampled for each object.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        theta :
            Theta value to be used for each SN.
            If None, the prior distribution will be sampled for each object.
            If a float is passed, the same value will be used for all objects.
            If a 1D ArrayLike of length equal to the N simulated objects is passed,
            these values will be used for each light curve.
        eps :
            Epsilon values to be used for each SN.
            If None, the prior distribution will be sampled for each object.
            If passing a 2D array, this must be of shape (N_l_knots, N_tau_knots) and
            will be used for each SN generated.
            If passing a 3d array, this must be of shape (N_sn, N_l_knots, N_tau_knots)
            providing epsilon values for each generated SN.
            If 0, an array of zeros of shape (N_sn, N_l_knots, N_tau_knots) will be used
            and epsilon is effectively turned off.
        mag :
            Determines whether returned values are mags (True) or fluxes (False).
        write_to_files :
            Determines whether to save simulated light curves to SNANA-format light
            curve files.
        output_dir :
            Path to output directory to save simulated SNANA-format files,
            only required if write_to_files == True.

        Returns
        -------
        data :
            Array containing simulated flux or mag values.
        yerr :
            Aray containing corresponding errors for each data point.
        param_dict :
            Dictionary of corresponding parameter values for each simulated object
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
                elif var.shape[0] != N:
                    raise ValueError(
                        f"If not providing a scalar {name} value, array must be of "
                        f"same length as the number of objects to simulate, N={N}."
                    )
            return var

        param_dict = {
            "del_M": (del_M, self.sample_del_M, N),
            "AV": (AV, self.sample_AV, N),
            "theta": (theta, self.sample_theta, N),
            "z": (z, None, N),
            "ebv_mw": (ebv_mw, None, N),
            "RV": (RV, lambda x: self.RV, N),
            "lam_shift": (lam_shift, self.sample_lam_shift, len(self.used_wave_sigmas)),
            "mag_shift": (mag_shift, self.sample_mag_shift, len(self.used_calib_cov)+1),
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
                    "For epsilon, please pass an ArrayLike object of shape (N, N_l_knots, N_tau_knots). The only scalar "
                    "value accepted is 0, which will effectively remove the effect of epsilon"
                )
        elif (
            len(eps.shape) != 3
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
            band_indices = np.array([self.band_dict[band] for band in bands])
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
        data = fn(
            model_spectra=phot_epoch_spectra,
            M0=self.M0,
            band_indices=band_indices,
            mask=mask,
            weights=jnp.repeat(band_weights, N, axis=0),
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
                    z[i],
                    ebv_mw[i],
                )
                sn_t = sn_t * (1 + sn_z)
                sn_tmax = 0
                sn_flt = [self.inv_band_dict[f] for f in band_indices[:, i]]
                sn_file = write_snana_lcfile(
                    output_dir,
                    sn_name,
                    sn_t,
                    sn_flt,
                    sn_mag,
                    sn_mag_err,
                    sn_tmax,
                    sn_z,
                    sn_z,
                    zerr,
                    sn_ebv_mw,
                )
                sn_names.append(sn_name)
                sn_files.append(sn_file)
        elif write_to_files:
            raise ValueError("If writing to SNANA files, please generate mags")
        return data, yerr, param_dict
    def list_bandpasses(self, group_by: str = "bp_name", return_dict: bool = False):
        """ Convenience method for listing recognised bandpasses.
        Parameters
        ----------
        group_by: str
            Either "bp_name" (default), or "site"
        return_dict: bool
            If True, return the created dictionary.

        Returns
        -------
        ret_dict: dict
            If return_dict is True, the method returns the created dictionary.
        """
        full_names = self.filter_dict["filters"].keys()
        paths = [val["path"].strip(
            str(self.__root_dir__ / "bayesn-filters/filters/")
            ) for val in self.filter_dict["filters"].values()]
        ret_dict = {}
        match group_by:
            case "bp_name":
                for bp in set(name.split("_")[0] for name in full_names):
                    ret_dict[bp] = [name for name in full_names if name.startswith(bp)]
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
        return {val: key for key, val in self.band_dict.items()}
