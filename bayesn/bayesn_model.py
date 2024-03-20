"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model.
"""

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_sample, init_to_value, Predictive
import numpyro.distributions as dist
from numpyro.optim import Adam
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import h5py
import sncosmo
from .spline_utils import invKD_irr, spline_coeffs_irr
from .bayesn_io import write_snana_lcfile
import pickle
import pandas as pd
import jax
from jax import device_put
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import ndtri, ndtr
from jax.random import PRNGKey, split
from astropy.cosmology import FlatLambdaCDM
import astropy.table as at
import astropy.constants as const
from astropy.io import ascii
import matplotlib as mpl
from matplotlib import rc
import arviz
import extinction
import timeit
from astropy.io import fits
from ruamel.yaml import YAML
import time
from tqdm import tqdm
from astropy.table import QTable

yaml = YAML(typ='safe')
yaml.default_flow_style = False

jax.config.update('jax_enable_x64', True)  # Enables 64 computation

np.seterr(divide='ignore', invalid='ignore')  # Disable divide by zero warnings

# jax.config.update('jax_platform_name', 'cpu')  # Forces CPU


class SEDmodel(object):
    """
    BayeSN-SED Model

    Class which imports a BayeSN model, and allows one to fit or simulate
    Type Ia supernovae based on this model.

    Parameters
    ----------
    num_devices: int, optional
            If running on a CPU, numpyro will by default see it as a single device - this argument will set the number
            of available cores for numpyro to use e.g. set to 4, you can train 4 chains on 4 cores in parallel. Defaults
            to 4.
    load_model : str, optional
        Can be either a pre-defined BayeSN model name (see table below), or
        a path to directory containing a set of .txt files from which a
        valid model can be constructed. Currently implemented default models
        are listed below - default is T21.

        ``M20_model``: Mandel+20 BayeSN model (arXiv:2008.07538).
                        Covers rest wavelength range of 3000-18500A (BVRIYJH). No treatment of host mass effects.
                        Global RV assumed. Trained on low-z Avelino+19 (ApJ, 887, 106) compilation of CfA, CSP and
                        others.
        ``T21_model``: Thorp+21 No-Split BayeSN model (arXiv:2102:05678).
                        Covers rest wavelength range of 3500-9500A (griz). No treatment of host mass effects. Global RV
                        assumed. Trained on Foundation DR1 (Foley+18, Jones+19).
        ``W22_model``: Ward+22 No-Split BayeSN model (arXiv:2209.10558).
                        Covers rest wavelength range of 3000-18500A (BVRIYJH). No treatment of host mass effects. Global
                        RV assumed. Trained on Foundation DR1 (Foley+18, Jones+19) and low-z Avelino+19 (ApJ, 887, 106)
                        compilation of CfA, CSP and others.
    fiducial_cosmology :  dict, optional
        Dictionary containg kwargs ``{H0, Om0}`` for initialising an ``astropy.cosmology.FlatLambdaCDM`` instance.
        Defaults to Riess+16 (ApJ, 826, 56) cosmology:
        ``{H0:73.24, "Om0":0.28}``.
    filter_yaml: str, optional
        Path to yaml file containing details on filters and standards to use. If not specified, will look for a file
        called filters.yaml in directory that BayeSN is called from.

    Methods
    -------

    dust_model:
        Defines numpyro model for inferring dust parameters with population level SN parameters fixed based on
        previously trained model.
    dust_redshift_model:
        Defines numpyro model for inferring dust parameters with population level SN parameters fixed based on
        previously trained model, allowing the means of the RV and AV distribution to linearly evolve with redshift
    dust_model_split_mag:
        Defines numpyro model for inferring dust parameters, splitting the population in two based on host galaxy
        stellar mass as well as allowing an intrinsic magnitude offset between the bins. Population level SN parameters
        are fixed based on previously trained model.
    dust_model_split_sed:
        Defines numpyro model for inferring dust parameters, splitting the population in two based on host galaxy
        stellar mass as well as allowing an intrinsic SED difference between the bins. Population level SN parameters
        are fixed based on previously trained model.
    fit_model_globalRV:
        Defines numpyro model for fitting latent SN parameters including distance, conditioned on fixed population
        level parameters based on previously trained model. Assumes single global RV across population.
    fit_model_popRV:
        Defines numpyro model for fitting latent SN parameters including distance, conditioned on fixed population
        level parameters based on previously trained model. Assumes truncated Gaussian population RV distribution.
    get_flux_batch:
        Get integrated fluxes for BayeSN SED model across a large number of SNe, phases and bands.
    def get_flux_from_chains:
        Get model photometry from BayeSN SED model for posterior samples from model fitting chains.
    get_mag_batch:
        Get magnitudes for BayeSN SED model across a large number of SNe, phases and bands.
    get_spectra:
        Get spectra for BayeSN SED model across a large number of SNe and phases.
    initial_guess:
        Defined method used to initialise chains for model training.
    parse_yaml_input:
        Parse the input yaml file along with any command line arguments to define the job being run.
    postprocess:
        Postprocess the output of the MCMC run if required and save the chains and summaries.
    process_dataset:
        Process a set of data for use by the BayeSN model.
    run:
        Run an inference job using the BayeSN model.
    sample_AV:
        Sample AV from the population distribution based on a pre-trained model.
    sample_del_M:
        Sample delta_M from the population distribution based on a pre-trained model.
    sample_epsilon:
        Sample epsilon from the population distribution based on a pre-trained model.
    sample_theta:
        Sample theta from the population distribution based on a pre-trained model.
    simulate_light_curve:
        Simulate a light curve or set of light curves from the BayeSN SED model.
    simulate_spectrum:
        Simulate a specrum or set of spectra from the BayeSN SED model.
    spline_coeffs_irr_step:
        Vectorized version of spline coefficient calculations in spline_utils.
    train_model_globalRV:
        Defines numpyro model to train the BayeSN SED model assuming a truncated Gaussian RV population distribution.
    train_model_popRV:
        Defines numpyro model to train the BayeSN SED model assuming a single global fixed RV value across the
        population.

    Attributes
    ----------
    cosmo: `astropy.cosmology.FlatLambdaCDM`
        Defines the fiducial cosmology assumed by the model when training
    RV_MW: float
        RV value for calculating Milky Way extinction
    sigma_pec: float
        Peculiar velocity to be used in calculating redshift uncertainties, set to 150 km/s
    l_knots: array-like
        Array of wavelength knots which the model is defined at
    t_knots: array-like
        Array of time knots which the model is defined at
    W0: array-like
        W0 matrix for loaded model
    W1: array-like
        W1 matrix for loaded model
    L_Sigma: array-like
        Covariance matrix describing epsilon distribution for loaded model
    M0: float
        Reference absolute magnitude for scaling Hsiao template
    sigma0: float
        Standard deviation of grey offset parameter for loaded model
    RV: float
        Global host extinction value for loaded model
    tauA: float
        Global tauA value for exponential AV prior for loaded model
    spectrum_bins: int
        Number of wavelength bins used for modelling spectra and calculating photometry. Based on ParSNiP as presented
        in Boone+21
    hsiao_flux: array-like
        Grid of flux value for Hsiao template
    hsiao_t: array-like
        Time values corresponding to Hsiao template grid
    hsiao_l: array-like
        Wavelength values corresponding to Hsiao template grid

    Returns
    -------
    out: `bayesn_model.SEDmodel` instance
    """

    def __init__(self, num_devices=4, load_model='T21_model', filter_yaml=None,
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        # Settings for jax/numpyro
        numpyro.set_host_device_count(num_devices)
        self.start_time = time.time()
        self.end_time = None
        # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print('Current devices:', jax.devices())

        self.__root_dir__ = os.path.dirname(os.path.abspath(__file__))
        print(f'Currently working in {os.getcwd()}')

        # Use built-in filters if filters.yaml is not provided
        if filter_yaml is None:
            filter_yaml = os.path.join(self.__root_dir__, 'bayesn-filters', 'filters.yaml')

        self.cosmo = FlatLambdaCDM(**fiducial_cosmology)
        self.data = None
        self.hsiao_interp = None
        self.RV_MW = device_put(jnp.array(3.1))
        self.sigma_pec = device_put(jnp.array(150 / 3e5))
        self.sn_list = None
        self.filter_yaml = filter_yaml
        built_in_models = next(os.walk(os.path.join(self.__root_dir__, 'model_files')))[1]

        if os.path.exists(load_model):
            print(f'Loading custom model at {load_model}')
            with open(load_model, 'r') as file:
                params = yaml.load(file)
        elif load_model in built_in_models:
            print(f'Loading built-in model {load_model}')
            with open(os.path.join(self.__root_dir__, 'model_files', load_model, 'BAYESN.YAML'), 'r') as file:
                params = yaml.load(file)
        else:
            raise FileNotFoundError(f'Specified model {load_model} does not exist and does not correspond to one '
                                    f'of the built-in model {built_in_models}')

        self.l_knots = jnp.array(params['L_KNOTS'])
        self.tau_knots = jnp.array(params['TAU_KNOTS'])
        self.W0 = jnp.array(params['W0'])
        self.W1 = jnp.array(params['W1'])
        self.L_Sigma = jnp.array(params['L_SIGMA_EPSILON'])
        self.M0 = jnp.array(params['M0'])
        self.sigma0 = jnp.array(params['SIGMA0'])
        self.tauA = jnp.array(params['TAUA'])
        if 'RV' in params.keys():
            self.model_type = 'fixed_RV'
            self.RV = jnp.array(params['RV'])
        elif 'MUR' in params.keys():
            self.model_type = 'pop_RV'
            self.mu_R = jnp.array(params['MUR'])
            self.sigma_R = jnp.array(params['SIGMAR'])

        self.trunc_val = 1.2

        self.used_band_inds = None
        self._setup_band_weights()

        KD_l = invKD_irr(self.l_knots)
        self.J_l_T = device_put(spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        self.KD_t = device_put(invKD_irr(self.tau_knots))
        self._load_hsiao_template()

        self.ZPT = 27.5  # Zero point
        self.J_l_T = device_put(self.J_l_T)
        self.hsiao_flux = device_put(self.hsiao_flux)
        self.J_l_T_hsiao = device_put(self.J_l_T_hsiao)
        self.xk = jnp.array(
            [0.0, 1e4 / 26500., 1e4 / 12200., 1e4 / 6000., 1e4 / 5470., 1e4 / 4670., 1e4 / 4110., 1e4 / 2700.,
             1e4 / 2600.])
        KD_x = invKD_irr(self.xk)
        self.M_fitz_block = device_put(spline_coeffs_irr(1e4 / self.model_wave, self.xk, KD_x))

        self.J_t_map = jax.jit(jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None)))

    def _load_hsiao_template(self):
        """
        Loads the Hsiao template from the internal HDF5 file.

        Stores the template as an attribute of `SEDmodel`.


        Returns
        -------

        """
        with h5py.File(os.path.join(self.__root_dir__, 'data', 'hsiao.h5'), 'r') as file:
            data = file['default']

            hsiao_phase = data['phase'][()].astype('float64')
            hsiao_wave = data['wave'][()].astype('float64')
            hsiao_flux = data['flux'][()].astype('float64')

        KD_l_hsiao = invKD_irr(hsiao_wave)
        self.KD_t_hsiao = device_put(invKD_irr(hsiao_phase))
        self.J_l_T_hsiao = device_put(spline_coeffs_irr(self.model_wave, hsiao_wave, KD_l_hsiao))

        self.hsiao_t = device_put(hsiao_phase)
        self.hsiao_l = device_put(hsiao_wave)
        self.hsiao_flux = device_put(hsiao_flux.T)
        self.hsiao_flux = jnp.matmul(self.J_l_T_hsiao, self.hsiao_flux)

    def _setup_band_weights(self):
        """
        Sets up the interpolation for the band weights used for photometry as well as calculating the zero points for
        each band. This code is partly based off ParSNiP from Boone+21
        """
        # Build the model in log wavelength
        self.min_wave = self.l_knots[0]
        self.max_wave = self.l_knots[-1]
        self.spectrum_bins = 300
        self.band_oversampling = 51
        self.max_redshift = 4

        model_log_wave = np.linspace(np.log10(self.min_wave),
                                     np.log10(self.max_wave),
                                     self.spectrum_bins)

        model_spacing = model_log_wave[1] - model_log_wave[0]

        band_spacing = model_spacing / self.band_oversampling
        band_max_log_wave = (
                np.log10(self.max_wave * (1 + self.max_redshift))
                + band_spacing
        )

        # Oversampling must be odd.
        assert self.band_oversampling % 2 == 1
        pad = (self.band_oversampling - 1) // 2
        band_log_wave = np.arange(np.log10(self.min_wave),
                                  band_max_log_wave, band_spacing)
        band_wave = 10 ** band_log_wave

        if not os.path.exists(self.filter_yaml):
            raise FileNotFoundError(f'Specified filter yaml {self.filter_yaml} does not exist')
        with open(self.filter_yaml, 'r') as file:
            filter_dict = yaml.load(file)

        # Load standard spectra if necessary, AB is just calculated analytically so no standard spectrum is required----
        if 'standards' in filter_dict.keys():
            if 'standards_root' in filter_dict.keys():
                standards_root = filter_dict['standards_root']
            else:
                standards_root = ''
            for key, val in filter_dict['standards'].items():
                path = os.path.join(standards_root, val['path'])
                # Fill environment variables if used e.g. $SNDATA_ROOT
                split_path = os.path.normpath(path).split(os.path.sep)
                root = split_path[0]
                if root[:1] == '$':
                    env = os.getenv(root[1:])
                    if env is None:
                        raise FileNotFoundError(f'The environment variable {root} was not found')
                    path = os.path.join(env, *split_path[1:])
                elif not os.path.isabs(path):  # If relative path, prepend yaml location
                    path = os.path.join(os.path.split(self.filter_yaml)[0], path)
                if '.fits' in path:  # If fits file
                    with fits.open(path) as hdu:
                        standard_df = pd.DataFrame.from_records(hdu[1].data)
                    standard_lam, standard_f = standard_df.WAVELENGTH.values, standard_df.FLUX.values
                else:
                    standard_txt = np.loadtxt(path)
                    standard_lam, standard_f = standard_txt[:, 0], standard_txt[:, 1]
                filter_dict['standards'][key]['lam'] = standard_lam
                filter_dict['standards'][key]['f_lam'] = standard_f
        else:
            print('You have not provided any standard spectra e.g. Vega in filter input yaml, this is fine as long '
                  'as everything is AB, otherwise make sure to add this')

        def ab_standard_flam(l):  # Can just use analytic function for AB spectrum
            f = (const.c.to('AA/s').value / 1e23) * (l ** -2) * 10 ** (-48.6 / 2.5) * 1e23
            return f

        # Load filters------------------------------
        if 'filters_root' in filter_dict.keys():
            filters_root = filter_dict['filters_root']
        else:
            filters_root = ''

        band_weights, zps, offsets = [], [], []
        self.band_dict, self.zp_dict, self.band_lim_dict = {}, {}, {}

        # Prepare NULL band. This is a fake band with a very wide wavelength range used only for padded data points to
        # ensure that these padded data points never fall out of the wavelength coverage of the model. These padded
        # data points do not contribute to the likelihood in any way, this is entirely for computational reasons
        self.band_dict['NULL_BAND'] = 0
        self.zp_dict['NULL_BAND'] = 10  # Arbitrary number
        self.band_lim_dict['NULL_BAND'] = band_wave[0], band_wave[-1]
        band_weights.append(np.ones_like(band_wave))
        zps.append(10)
        offsets.append(0)

        band_ind = 1
        for key, val in filter_dict['filters'].items():
            path = os.path.join(filters_root, val['path'])
            # Fill environment variables if used e.g. $SNDATA_ROOT
            split_path = os.path.normpath(path).split(os.path.sep)
            root = split_path[0]
            if root[:1] == '$':
                env = os.getenv(root[1:])
                if env is None:
                    raise FileNotFoundError(f'The environment variable {root} was not found')
                path = os.path.join(env, *split_path[1:])
            elif not os.path.isabs(path):  # If relative path, prepend yaml location
                path = os.path.join(os.path.split(self.filter_yaml)[0], path)
            band, magsys, offset = key, val['magsys'], val['magzero']
            try:
                R = np.loadtxt(path)
            except:
                raise FileNotFoundError(f'Filter response not found for {key}')

            # Convert wavelength units if required, model is defined in Angstroms
            units = val.get('lam_unit', 'AA')
            if units.lower() == 'nm':  # Convert from nanometres to Angstroms
                R[:, 0] = R[:, 0] * 10
            elif units.lower() == 'micron':  # Convert from microns to Angstroms
                R[:, 0] = R[:, 0] * 1e4

            band_low_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][0], 0]
            band_up_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][-1], 0]

            # Convolve the bands to match the sampling of the spectrum.
            band_conv_transmission = jnp.interp(band_wave, R[:, 0], R[:, 1], left=0, right=0)
            # band_conv_transmission = scipy.interpolate.interp1d(R[:, 0], R[:, 1], kind='cubic',
            #                                                     fill_value=0, bounds_error=False)(band_wave)

            dlamba = jnp.diff(band_wave)
            dlamba = jnp.r_[dlamba, dlamba[-1]]

            num = band_wave * band_conv_transmission * dlamba
            denom = jnp.sum(num)
            band_weight = num / denom

            band_weights.append(band_weight)

            # Get zero points
            lam = R[:, 0]
            if magsys == 'ab':
                zp = ab_standard_flam(lam)
            else:
                standard = filter_dict['standards'][magsys]
                zp = interp1d(standard['lam'], standard['f_lam'], kind='cubic')(lam)

            int1 = simpson(lam * zp * R[:, 1], lam)
            int2 = simpson(lam * R[:, 1], lam)
            zp = 2.5 * np.log10(int1 / int2)
            self.band_dict[band] = band_ind
            self.band_lim_dict[band] = [band_low_lim, band_up_lim]
            self.zp_dict[band] = zp
            zps.append(zp)
            offsets.append(offset)
            band_ind += 1

        self.used_band_inds = np.array(list(self.band_dict.values()))
        self.zps = jnp.array(zps)
        self.offsets = jnp.array(offsets)
        self.inv_band_dict = {val: key for key, val in self.band_dict.items()}

        # Get the locations that should be sampled at redshift 0. We can scale these to
        # get the locations at any redshift.
        band_interpolate_locations = jnp.arange(
            0,
            self.spectrum_bins * self.band_oversampling,
            self.band_oversampling
        )

        # Save the variables that we need to do interpolation.
        self.band_interpolate_locations = device_put(band_interpolate_locations)
        self.band_interpolate_spacing = band_spacing
        self.band_interpolate_weights = jnp.array(band_weights)
        self.model_wave = 10 ** model_log_wave

    def _calculate_band_weights(self, redshifts, ebv):
        """
        Calculates the observer-frame band weights, including the effect of Milky Way extinction, for each SN

        Parameters
        ----------
        redshifts: array-like
            Array of redshifts for each SN
        ebv: array-like
            Array of Milky Way E(B-V) values for each SN

        Returns
        -------

        weights: array-like
            Array containing observer-frame band weights

        """
        # Figure out the locations to sample at for each redshift.
        locs = (
                self.band_interpolate_locations
                + jnp.log10(1 + redshifts)[:, None] / self.band_interpolate_spacing
        )

        flat_locs = locs.flatten()

        # Linear interpolation
        int_locs = flat_locs.astype(jnp.int32)
        remainders = flat_locs - int_locs

        self.band_interpolate_weights = self.band_interpolate_weights[self.used_band_inds, ...]

        start = self.band_interpolate_weights[..., int_locs]
        end = self.band_interpolate_weights[..., int_locs + 1]

        flat_result = remainders * end + (1 - remainders) * start
        weights = flat_result.reshape((-1,) + locs.shape).transpose(1, 2, 0)
        # Normalise so max transmission = 1
        sum = jnp.sum(weights, axis=1)
        weights /= sum[:, None, :]

        # Apply MW extinction
        av = self.RV_MW * ebv
        all_lam = np.array(self.model_wave[None, :] * (1 + redshifts[:, None]))
        all_lam = all_lam.flatten(order='F')
        mw_ext = extinction.fitzpatrick99(all_lam, 1, self.RV_MW)
        mw_ext = mw_ext.reshape((weights.shape[0], weights.shape[1]), order='F')
        mw_ext = mw_ext * av[:, None]
        mw_ext = jnp.power(10, -0.4 * mw_ext)

        weights = weights * mw_ext[..., None]

        # We need an extra term of 1 + z from the filter contraction.
        weights /= (1 + redshifts)[:, None, None]

        return weights

    def get_spectra(self, theta, AV, W0, W1, eps, RV, J_t, hsiao_interp):
        """
        Calculates rest-frame spectra for given parameter values

        Parameters
        ----------
        theta: array-like
            Set of theta values for each SN
        AV: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        RV: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation


        Returns
        -------

        model_spectra: array-like
            Matrix containing model spectra for all SNe at all time-steps

        """
        num_batch = theta.shape[0]
        # W0 = jnp.repeat(W0[None, ...], num_batch, axis=0)
        # W1 = jnp.repeat(W1[None, ...], num_batch, axis=0)

        W = W0 + theta[..., None, None] * W1 + eps

        WJt = jnp.matmul(W, J_t)
        W_grid = jnp.matmul(self.J_l_T, WJt)

        low_hsiao = self.hsiao_flux[:, hsiao_interp[0, ...].astype(int)]
        up_hsiao = self.hsiao_flux[:, hsiao_interp[1, ...].astype(int)]
        H_grid = ((1 - hsiao_interp[2, :]) * low_hsiao + hsiao_interp[2, :] * up_hsiao).transpose(2, 0, 1)

        model_spectra = H_grid * 10 ** (-0.4 * W_grid)

        # Extinction----------------------------------------------------------
        f99_x0 = 4.596
        f99_gamma = 0.99
        f99_c2 = -0.824 + 4.717 / RV
        f99_c1 = 2.030 - 3.007 * f99_c2
        f99_c3 = 3.23
        f99_c4 = 0.41
        f99_c5 = 5.9
        f99_d1 = self.xk[7] ** 2 / ((self.xk[7] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[7]) ** 2)
        f99_d2 = self.xk[8] ** 2 / ((self.xk[8] ** 2 - f99_x0 ** 2) ** 2 + (f99_gamma * self.xk[8]) ** 2)
        yk = jnp.zeros((num_batch, 9))
        yk = yk.at[:, 0].set(-RV)
        yk = yk.at[:, 1].set(0.26469 * RV / 3.1 - RV)
        yk = yk.at[:, 2].set(0.82925 * RV / 3.1 - RV)
        yk = yk.at[:, 3].set(-0.422809 + 1.00270 * RV + 2.13572e-4 * RV ** 2 - RV)
        yk = yk.at[:, 4].set(-5.13540e-2 + 1.00216 * RV - 7.35778e-5 * RV ** 2 - RV)
        yk = yk.at[:, 5].set(0.700127 + 1.00184 * RV - 3.32598e-5 * RV ** 2 - RV)
        yk = yk.at[:, 6].set(
            1.19456 + 1.01707 * RV - 5.46959e-3 * RV ** 2 + 7.97809e-4 * RV ** 3 - 4.45636e-5 * RV ** 4 - RV)
        yk = yk.at[:, 7].set(f99_c1 + f99_c2 * self.xk[7] + f99_c3 * f99_d1)
        yk = yk.at[:, 8].set(f99_c1 + f99_c2 * self.xk[8] + f99_c3 * f99_d2)

        A = AV[..., None] * (1 + (self.M_fitz_block @ yk.T).T / RV[..., None])  # RV[..., None]
        f_A = 10 ** (-0.4 * A)
        model_spectra = model_spectra * f_A[..., None]

        return model_spectra

    def get_flux_batch(self, M0, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame fluxes for given parameter values

        Parameters
        ----------
        M0: float or array-like
            Normalising constant to scale Hsiao template to correct order of magnitude. Typically fixed to -19.5
            although can be inferred separately for different bins in a mass split analysis
        theta: array-like
            Set of theta values for each SN
        AV: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        Ds: array-like
            Set of distance moduli for each SN
        RV: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        band_indices: array-like
            Array containing indices describing which filter each observation is in
        mask: array-like
            Array containing mask describing whether observations should contribute to the posterior
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation
        weights: array_like
            Array containing band weights to use for photometry

        Returns
        -------

        model_flux: array-like
            Matrix containing model fluxes for all SNe at all time-steps

        """
        num_batch = theta.shape[0]
        num_observations = band_indices.shape[0]

        model_spectra = self.get_spectra(theta, AV, W0, W1, eps, RV, J_t, hsiao_interp)

        batch_indices = (
            jnp.arange(num_batch)
            .repeat(num_observations)
        ).astype(int)
        obs_band_weights = (
            weights[batch_indices, :, band_indices.T.flatten()]
            .reshape((num_batch, num_observations, -1))
            .transpose(0, 2, 1)
        )

        model_flux = jnp.sum(model_spectra * obs_band_weights, axis=1).T
        model_flux = model_flux * 10 ** (-0.4 * (M0 + Ds))
        zps = self.zps[band_indices]
        offsets = self.offsets[band_indices]
        zp_flux = 10 ** (zps / 2.5)
        model_flux = (model_flux / zp_flux) * 10 ** (0.4 * (27.5 - offsets))  # Convert to FLUXCAL
        model_flux *= mask
        return model_flux

    def get_mag_batch(self, M0, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame magnitudes for given parameter values

        Parameters
        ----------
        M0: float or array-like
            Normalising constant to scale Hsiao template to correct order of magnitude. Typically fixed to -19.5
            although can be inferred separately for different bins in a mass split analysis
        theta: array-like
            Set of theta values for each SN
        AV: array-like
            Set of host extinction values for each SN
        W0: array-like
            Global W0 matrix
        W1: array-like
            Global W1 matrix
        eps: array-like
            Set of epsilon values for each SN, describing residual colour variation
        Ds: array-like
            Set of distance moduli for each SN
        RV: float
            Global R_V value for host extinction (need to allow this to be variable in future)
        band_indices: array-like
            Array containing indices describing which filter each observation is in
        mask: array-like
            Array containing mask describing whether observations should contribute to the posterior
        J_t: array-like
            Matrix for cubic spline interpolation in time axis for each SN
        hsiao_interp: array-like
            Array containing Hsiao template spectra for each t value, comprising model for previous day, next day and
            t % 1 to allow for linear interpolation
        weights: array_like
            Array containing band weights to use for photometry

        Returns
        -------

        model_mag: array-like
            Matrix containing model magnitudes for all SNe at all time-steps
        """
        model_flux = self.get_flux_batch(M0, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights)
        model_flux = model_flux + (1 - mask) * 0.01  # Masked data points are set to 0, set them to a small value
        # to avoid nans when logging

        model_mag = - 2.5 * jnp.log10(model_flux) + 27.5
        model_mag *= mask  # Re-apply mask

        return model_mag

    @staticmethod
    def spline_coeffs_irr_step(x_now, x, invkd):
        """
        Vectorized version of cubic spline coefficient calculator found in spline_utils

        Parameters
        ----------
        x_now: array-like
            Current x location to calculate spline knots for
        x: array-like
            Numpy array containing the locations of the spline knots.
        invkd: array-like
            Precomputed matrix for generating second derivatives. Can be obtained
            from the output of ``spline_utils.invKD_irr``.

        Returns
        -------

        X: Set of spline coefficients for each x knot

        """
        X = jnp.zeros_like(x)
        up_extrap = x_now > x[-1]
        down_extrap = x_now < x[0]
        interp = 1 - up_extrap - down_extrap

        h = x[-1] - x[-2]
        a = (x[-1] - x_now) / h
        b = 1 - a
        f = (x_now - x[-1]) * h / 6.0

        X = X.at[-2].set(X[-2] + a * up_extrap)
        X = X.at[-1].set(X[-1] + b * up_extrap)
        X = X.at[:].set(X[:] + f * invkd[-2, :] * up_extrap)

        h = x[1] - x[0]
        b = (x_now - x[0]) / h
        a = 1 - b
        f = (x_now - x[0]) * h / 6.0

        X = X.at[0].set(X[0] + a * down_extrap)
        X = X.at[1].set(X[1] + b * down_extrap)
        X = X.at[:].set(X[:] - f * invkd[1, :] * down_extrap)

        q = jnp.argmax(x_now < x) - 1
        h = x[q + 1] - x[q]
        a = (x[q + 1] - x_now) / h
        b = 1 - a
        c = ((a ** 3 - a) / 6) * h ** 2
        d = ((b ** 3 - b) / 6) * h ** 2

        X = X.at[q].set(X[q] + a * interp)
        X = X.at[q + 1].set(X[q + 1] + b * interp)
        X = X.at[:].set(X[:] + c * invkd[q, :] * interp + d * invkd[q + 1, :] * interp)

        return X

    def fit_model_globalRV(self, obs, weights):
        """
        Numpyro model used for fitting latent SN properties with single global RV. Will fit for time of maximum as well
        as theta, epsilon, AV and distance modulus.

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry

        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))
            AV = numpyro.sample(f'AV', dist.Exponential(1 / self.tauA))
            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
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
            # Ds = numpyro.sample('Ds', dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))  # Ds_err
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def fit_model_popRV(self, obs, weights):
        """
        Numpyro model used for fitting latent SN properties with a truncated Gaussian prior on RV. Will fit for time of
        maximum as well as theta, epsilon, AV, RV and distance modulus.

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band-weights to calculate photometry

        """
        sample_size = obs.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        phi_alpha_R = norm.cdf((self.trunc_val - self.mu_R) / self.sigma_R)

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))
            AV = numpyro.sample(f'AV', dist.Exponential(1 / self.tauA))
            RV_tform = numpyro.sample('RV_tform', dist.Uniform(0, 1))
            RV = numpyro.deterministic('Rv_LM',
                                       self.mu_R + self.sigma_R * ndtri(phi_alpha_R + RV_tform * (1 - phi_alpha_R)))

            tmax = numpyro.sample('tmax', dist.Uniform(-10, 10))
            t = obs[0, ...] - tmax[None, sn_index]
            hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
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
            # Ds = numpyro.sample('Ds', dist.ImproperUniform(dist.constraints.greater_than(0), (), event_shape=()))
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))  # Ds_err
            flux = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, Ds, RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)  # _{sn_index}

    def train_model_globalRV(self, obs, weights):
        """
        Numpyro model used for training to learn global parameters, assuming a single global RV

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        """
        sample_size = self.data.shape[-1]
        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)
        W0 = numpyro.sample('W0', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W1 = numpyro.sample('W1', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W0 = jnp.reshape(W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = jnp.reshape(W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')

        # sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(1 * jnp.ones(N_knots_sig)))
        sigmaepsilon_tform = numpyro.sample('sigmaepsilon_tform',
                                            dist.Uniform(0, (jnp.pi / 2.) * jnp.ones(N_knots_sig)))
        sigmaepsilon = numpyro.deterministic('sigmaepsilon', 1. * jnp.tan(sigmaepsilon_tform))
        L_Omega = numpyro.sample('L_Omega', dist.LKJCholesky(N_knots_sig))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)

        # sigma0 = numpyro.sample('sigma0', dist.HalfCauchy(0.1))
        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0 = numpyro.deterministic('sigma0', 0.1 * jnp.tan(sigma0_tform))

        RV = numpyro.sample('RV', dist.Uniform(1, 5))

        # tauA = numpyro.sample('tauA', dist.HalfCauchy())
        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            AV = numpyro.sample(f'AV', dist.Exponential(1 / tauA))

            eps_mu = jnp.zeros(N_knots_sig)
            # eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=L_Sigma))
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            # eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_mag_batch(self.M0, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, self.J_t, self.hsiao_interp,
                                      weights)

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def train_model_popRV(self, obs, weights):
        """
        Numpyro model used for training to learn global parameters with a truncated Gaussian RV distribution

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        """
        sample_size = self.data.shape[-1]
        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)
        W0 = numpyro.sample('W0', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W1 = numpyro.sample('W1', dist.MultivariateNormal(W_mu, jnp.eye(N_knots)))
        W0 = jnp.reshape(W0, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        W1 = jnp.reshape(W1, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')

        # sigmaepsilon = numpyro.sample('sigmaepsilon', dist.HalfNormal(1 * jnp.ones(N_knots_sig)))
        sigmaepsilon_tform = numpyro.sample('sigmaepsilon_tform',
                                            dist.Uniform(0, (jnp.pi / 2.) * jnp.ones(N_knots_sig)))
        sigmaepsilon = numpyro.deterministic('sigmaepsilon', 1. * jnp.tan(sigmaepsilon_tform))
        L_Omega = numpyro.sample('L_Omega', dist.LKJCholesky(N_knots_sig))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon), L_Omega)

        # sigma0 = numpyro.sample('sigma0', dist.HalfCauchy(0.1))
        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0 = numpyro.deterministic('sigma0', 0.1 * jnp.tan(sigma0_tform))

        mu_R = numpyro.sample('mu_R', dist.Uniform(1, 5))
        sigma_R = numpyro.sample('sigma_R', dist.HalfNormal(2))
        phi_alpha_R = norm.cdf((self.trunc_val - mu_R) / sigma_R)

        # tauA = numpyro.sample('tauA', dist.HalfCauchy())
        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            AV = numpyro.sample(f'AV', dist.Exponential(1 / tauA))
            RV_tform = numpyro.sample('RV_tform', dist.Uniform(0, 1))
            RV = numpyro.deterministic('Rv_LM', mu_R + sigma_R * ndtri(phi_alpha_R + RV_tform * (1 - phi_alpha_R)))

            eps_mu = jnp.zeros(N_knots_sig)
            # eps = numpyro.sample('eps', dist.MultivariateNormal(eps_mu, scale_tril=L_Sigma))
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            # eps = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_mag_batch(self.M0, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, self.J_t, self.hsiao_interp,
                                      weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def dust_model(self, obs, weights):
        """
        Numpryo model used to infer dust properties conditioned on fixed SN population parameters from a previously
        trained model.

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        Returns
        -------

        """
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        mu_R = numpyro.sample('mu_R', dist.Uniform(1.2, 6))
        sigma_R = numpyro.sample('sigma_R', dist.HalfNormal(2))
        phi_alpha_R = norm.cdf((1.2 - mu_R) / sigma_R)
        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0 = numpyro.deterministic('sigma0', 0.1 * jnp.tan(sigma0_tform))

        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / tauA))

            Rv_tform = numpyro.sample('Rv_tform', dist.Uniform(0, 1))
            Rv = numpyro.deterministic('Rv', mu_R + sigma_R * ndtri(phi_alpha_R + Rv_tform * (1 - phi_alpha_R)))

            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(self.M0, theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                       weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def dust_redshift_model(self, obs, weights):
        """
        Numpryo model used to infer dust properties conditioned on fixed SN population parameters from a previously
        trained model, allowing the mean of the RV and AV distributions to linearly evolve with redshift.

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        Returns
        -------

        """
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        mu_R_0 = numpyro.sample('mu_R_0', dist.Uniform(1.2, 6))
        sigma_R = numpyro.sample('sigma_R', dist.HalfNormal(2))
        phi_alpha_R = norm.cdf((1.2 - mu_R_0) / sigma_R)

        mu_z_grad = numpyro.sample('mu_grad', dist.Uniform(1.2 - mu_R_0, 6 - mu_R_0))

        sigma0_tform = numpyro.sample('sigma0_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0 = numpyro.deterministic('sigma0', 0.1 * jnp.tan(sigma0_tform))

        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))
        tau_z_grad = numpyro.sample('tau_z_grad', dist.Uniform(-0.5, 0.5))

        with numpyro.plate('SNe', sample_size) as sn_index:
            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))

            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)

            mu_R = mu_R_0 + redshift * mu_z_grad
            tauA = tauA + redshift * tau_z_grad

            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            Av = numpyro.sample(f'AV', dist.Exponential(1 / tauA))
            Rv_tform = numpyro.sample('Rv_tform', dist.Uniform(0, 1))
            Rv = numpyro.deterministic('Rv', mu_R + sigma_R * ndtri(phi_alpha_R + Rv_tform * (1 - phi_alpha_R)))

            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)

            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(self.M0, theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                       weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def dust_model_split_mag(self, obs, weights):
        """
        Numpryo model used to infer dust properties conditioned on fixed SN population parameters from a previously
        trained model, split into different mass bins above and below 10^10 solar masses. This model allows for a
        constant intrinsic magnitude offset between the two mass bins

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        """
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        mu_R_HM = numpyro.sample('mu_R_HM', dist.Uniform(1.2, 6))
        sigma_R_HM = numpyro.sample('sigma_R_HM', dist.HalfNormal(2))
        phi_alpha_R_HM = norm.cdf((1.2 - mu_R_HM) / sigma_R_HM)

        mu_R_LM = numpyro.sample('mu_R_LM', dist.Uniform(1.2, 6))
        sigma_R_LM = numpyro.sample('sigma_R_LM', dist.HalfNormal(2))
        phi_alpha_R_LM = norm.cdf((1.2 - mu_R_LM) / sigma_R_LM)

        tauA_HM_tform = numpyro.sample('tauA_HM_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA_HM = numpyro.deterministic('tauA_HM', jnp.tan(tauA_HM_tform))

        tauA_LM_tform = numpyro.sample('tauA_LM_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA_LM = numpyro.deterministic('tauA_LM', jnp.tan(tauA_LM_tform))

        sigma0_HM_tform = numpyro.sample('sigma0_HM_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0_HM = numpyro.deterministic('sigma0_HM', 0.1 * jnp.tan(sigma0_HM_tform))

        sigma0_LM_tform = numpyro.sample('sigma0_LM_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0_LM = numpyro.deterministic('sigma0_LM', 0.1 * jnp.tan(sigma0_LM_tform))

        M_step_HM = numpyro.sample('M_step_HM', dist.Uniform(-0.2, 0.2))
        M_step_LM = numpyro.sample('M_step_LM', dist.Uniform(-0.2, 0.2))

        mass = obs[-7, 0, :]
        M_split = 10  # Hardcoded for now, should make this customisable
        HM_flag = mass > M_split

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))

            Av_LM = numpyro.sample(f'AV_LM', dist.Exponential(1 / tauA_LM))
            Av_HM = numpyro.sample(f'AV_HM', dist.Exponential(1 / tauA_HM))
            Av = numpyro.deterministic('AV', HM_flag * Av_HM + (1 - HM_flag) * Av_LM)

            Rv_tform_HM = numpyro.sample('Rv_tform_HM', dist.Uniform(0, 1))
            Rv_HM = numpyro.deterministic('Rv_HM', mu_R_HM + sigma_R_HM * ndtri(phi_alpha_R_HM + Rv_tform_HM * (1 - phi_alpha_R_HM)))
            Rv_tform_LM = numpyro.sample('Rv_tform_LM', dist.Uniform(0, 1))
            Rv_LM = numpyro.deterministic('Rv_LM', mu_R_LM + sigma_R_LM * ndtri(
                phi_alpha_R_LM + Rv_tform_LM * (1 - phi_alpha_R_LM)))
            Rv = numpyro.deterministic('Rv', HM_flag * Rv_HM + (1 - HM_flag) * Rv_LM)

            M0 = self.M0 * jnp.ones_like(Rv) + HM_flag * M_step_HM + (1 - HM_flag) * M_step_LM

            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)

            sigma0 = HM_flag * sigma0_HM + (1 - HM_flag) * sigma0_LM

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(M0, theta, Av, self.W0, self.W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                       weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def dust_model_split_sed(self, obs, weights):
        """
        Numpryo model used to infer dust properties conditioned on fixed SN population parameters from a previously
        trained model, split into different mass bins above and below 10^10 solar masses. This model allows for a
        intrinsic difference in baseline SED (independent of light curve stretch) between the two mass bins

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset
        weights: array-like
            Band weights based on filter responses and MW extinction curves for numerical flux integrals

        """
        sample_size = self.data.shape[-1]
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]

        N_knots = self.l_knots.shape[0] * self.tau_knots.shape[0]
        W_mu = jnp.zeros(N_knots)

        delW_HM = numpyro.sample('delW_HM', dist.MultivariateNormal(W_mu, 0.1 * jnp.eye(N_knots)))
        delW_LM = numpyro.sample('delW_LM', dist.MultivariateNormal(W_mu, 0.1 * jnp.eye(N_knots)))

        delW_HM = jnp.reshape(delW_HM, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')
        delW_LM = jnp.reshape(delW_LM, (self.l_knots.shape[0], self.tau_knots.shape[0]), order='F')

        W0_HM = numpyro.deterministic('W0_HM', self.W0 + delW_HM)
        W0_LM = numpyro.deterministic('W0_LM', self.W0 + delW_LM)

        mu_R_HM = numpyro.sample('mu_R_HM', dist.Uniform(1.2, 6))
        sigma_R_HM = numpyro.sample('sigma_R_HM', dist.HalfNormal(2))
        phi_alpha_R_HM = norm.cdf((1.2 - mu_R_HM) / sigma_R_HM)

        mu_R_LM = numpyro.sample('mu_R_LM', dist.Uniform(1.2, 6))
        sigma_R_LM = numpyro.sample('sigma_R_LM', dist.HalfNormal(2))
        phi_alpha_R_LM = norm.cdf((1.2 - mu_R_LM) / sigma_R_LM)

        tauA_HM_tform = numpyro.sample('tauA_HM_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA_HM = numpyro.deterministic('tauA_HM', jnp.tan(tauA_HM_tform))

        tauA_LM_tform = numpyro.sample('tauA_LM_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA_LM = numpyro.deterministic('tauA_LM', jnp.tan(tauA_LM_tform))

        sigma0_HM_tform = numpyro.sample('sigma0_HM_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0_HM = numpyro.deterministic('sigma0_HM', 0.1 * jnp.tan(sigma0_HM_tform))

        sigma0_LM_tform = numpyro.sample('sigma0_LM_tform', dist.Uniform(0, jnp.pi / 2.))
        sigma0_LM = numpyro.deterministic('sigma0_LM', 0.1 * jnp.tan(sigma0_LM_tform))

        mass = obs[-7, 0, :]
        M_split = 10
        HM_flag = mass > M_split

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0., 1.))

            Av_LM = numpyro.sample(f'AV_LM', dist.Exponential(1 / tauA_LM))
            Av_HM = numpyro.sample(f'AV_HM', dist.Exponential(1 / tauA_HM))
            Av = numpyro.deterministic('AV', HM_flag * Av_HM + (1 - HM_flag) * Av_LM)

            Rv_tform_HM = numpyro.sample('Rv_tform_HM', dist.Uniform(0, 1))
            Rv_HM = numpyro.deterministic('Rv_HM', mu_R_HM + sigma_R_HM * ndtri(phi_alpha_R_HM + Rv_tform_HM * (1 - phi_alpha_R_HM)))
            Rv_tform_LM = numpyro.sample('Rv_tform_LM', dist.Uniform(0, 1))
            Rv_LM = numpyro.deterministic('Rv_LM', mu_R_LM + sigma_R_LM * ndtri(
                phi_alpha_R_LM + Rv_tform_LM * (1 - phi_alpha_R_LM)))
            Rv = numpyro.deterministic('Rv', HM_flag * Rv_HM + (1 - HM_flag) * Rv_LM)

            W0 = HM_flag[:, None, None] * W0_HM[None, ...] + (1 - HM_flag)[:, None, None] * W0_LM[None, ...]

            eps_mu = jnp.zeros(N_knots_sig)
            eps_tform = numpyro.sample('eps_tform', dist.MultivariateNormal(eps_mu, jnp.eye(N_knots_sig)))
            eps_tform = eps_tform.T
            eps = numpyro.deterministic('eps', jnp.matmul(self.L_Sigma, eps_tform))
            eps = eps.T
            eps = jnp.reshape(eps, (sample_size, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((sample_size, self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)

            sigma0 = HM_flag * sigma0_HM + (1 - HM_flag) * sigma0_LM

            band_indices = obs[-6, :, sn_index].astype(int).T
            redshift = obs[-5, 0, sn_index]
            redshift_error = obs[-4, 0, sn_index]
            muhat = obs[-3, 0, sn_index]
            ebv = obs[-2, 0, sn_index]

            mask = obs[-1, :, sn_index].T.astype(bool)
            muhat_err = 5 / (redshift * jnp.log(10)) * jnp.sqrt(
                jnp.power(redshift_error, 2) + np.power(self.sigma_pec, 2))
            Ds_err = jnp.sqrt(muhat_err * muhat_err + sigma0 * sigma0)
            Ds = numpyro.sample('Ds', dist.Normal(muhat, Ds_err))
            flux = self.get_flux_batch(self.M0, theta, Av, W0, self.W1, eps, Ds, Rv, band_indices, mask, self.J_t, self.hsiao_interp,
                                      weights)

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def initial_guess(self, args, reference_model='M20_model'):
        """
        Sets initialisation for training chains, using some global parameter values from previous models.
        W0 and W1 matrices are interpolated to match wavelength knots of new model, and set to zero beyond
        the time range that the reference model is defined for. Note that unlike Stan, in numpyro we cannot set each
        chain's initialisation separately.

        Parameters
        ----------
        reference_model: str, optional
            Previously-trained model to be used to set initialisation, defaults to T21.

        Returns
        -------
        param_init: dict
            Dictionary containing initial values to be used

        """
        # Set hyperparameter initialisations
        built_in_models = next(os.walk(os.path.join(self.__root_dir__, 'model_files')))[1]
        if os.path.exists(reference_model):
            print(f'Using custom model at {reference_model} to initialise chains')
            with open(reference_model, 'r') as file:
                params = yaml.load(file)
        elif reference_model in built_in_models:
            print(f'Loading built-in model {reference_model} to initialise chains')
            with open(os.path.join(self.__root_dir__, 'model_files', reference_model, 'BAYESN.YAML'), 'r') as file:
                params = yaml.load(file)
        else:
            raise ValueError("Invalid initialisation method, please choose either 'median' or 'sample', or choose "
                             "either one of the built-in models or a custom model to base the hyperparmeter "
                             "initialisation on")
        W0_init = params['W0']
        l_knots = params['L_KNOTS']
        tau_knots = params['TAU_KNOTS']
        W1_init = params['W1']
        RV_init, tauA_init = params['RV'], params['TAUA']

        # Interpolate to match new wavelength knots
        W0_init = interp1d(l_knots, W0_init, kind='cubic', axis=0, fill_value=0, bounds_error=False)(self.l_knots)
        W1_init = interp1d(l_knots, W1_init, kind='cubic', axis=0, fill_value=0, bounds_error=False)(self.l_knots)

        # Interpolate to match new time knots
        W0_init = interp1d(tau_knots, W0_init, kind='linear', axis=1, fill_value=0, bounds_error=False)(self.tau_knots)
        W1_init = interp1d(tau_knots, W1_init, kind='linear', axis=1, fill_value=0, bounds_error=False)(self.tau_knots)

        W0_init = W0_init.flatten(order='F')
        W1_init = W1_init.flatten(order='F')

        n_eps = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        sigma0_init = 0.1
        sigmaepsilon_init = 0.1 * np.ones(n_eps)
        L_Omega_init = np.eye(n_eps)

        n_sne = self.data.shape[-1]

        # Prepare initial guesses
        param_init = {}
        tauA_ = tauA_init + np.random.normal(0, 0.01)
        while tauA_ < 0:
            tauA_ = tauA_init + np.random.normal(0, 0.01)
        sigma0_ = sigma0_init + np.random.normal(0, 0.01)
        param_init['W0'] = jnp.array(W0_init + np.random.normal(0, 0.01, W0_init.shape[0]))
        param_init['W1'] = jnp.array(W1_init + np.random.normal(0, 0.01, W1_init.shape[0]))
        if 'poprv' in args['mode'].lower():
            param_init['mu_R'] = jnp.array(3.)
            param_init['sigma_R'] = jnp.array(0.5)
            param_init['RV_tform'] = jnp.array(np.random.uniform(0, 1, self.data.shape[-1]))
        else:
            param_init['RV'] = jnp.array(3.)
        param_init['tauA_tform'] = jnp.arctan(tauA_ / 1.)
        param_init['sigma0_tform'] = jnp.arctan(sigma0_ / 0.1)
        param_init['sigma0'] = jnp.array(sigma0_)
        param_init['theta'] = jnp.array(np.random.normal(0, 1, n_sne))
        param_init['AV'] = jnp.array(np.random.exponential(tauA_, n_sne))
        L_Sigma = jnp.matmul(jnp.diag(sigmaepsilon_init), L_Omega_init)

        param_init['epsilon_tform'] = jnp.matmul(np.linalg.inv(L_Sigma), np.random.normal(0, 1, (n_eps, n_sne)))
        param_init['epsilon'] = np.random.normal(0, 1, (n_sne, n_eps))
        param_init['sigmaepsilon_tform'] = jnp.arctan(
            sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape) / 1.)
        param_init['sigmaepsilon'] = sigmaepsilon_init + np.random.normal(0, 0.01, sigmaepsilon_init.shape)
        param_init['L_Omega'] = jnp.array(L_Omega_init)

        param_init['Ds'] = jnp.array(np.random.normal(self.data[-3, 0, :], sigma0_))

        return param_init

    def parse_yaml_input(self, args, cmd_args):
        """
        Method to parse the input yaml file and process data-set

        Parameters
        ----------
        args: dict
            dictionary of arguments to define model based on input yaml file
        cmd_args: dict
            dictionary of command line arguments, which will override yaml file if specified
        -------

        """
        # Command line overrides, if present-----------------------------------
        for arg in vars(cmd_args):
            if arg in ['input', 'filters']:
                continue
            arg_val = getattr(cmd_args, arg)
            if arg_val is not None:
                if arg == 'map':
                    filt_map = np.loadtxt(cmd_args.map, dtype=str)
                    arg_val = {row[0]: row[1] for row in filt_map}
                args[arg] = arg_val

        args.pop('CONFIG', None)
        args.pop('config', None)

        # Set default parameters for some parameters if not specified in input.yaml or command line
        args['num_chains'] = args.get('num_chains', 4)
        args['num_warmup'] = args.get('num_warmup', 500)
        args['num_samples'] = args.get('num_samples', 500)
        args['chain_method'] = args.get('chain_method', 'parallel')
        args['initialisation'] = args.get('initialisation', 'median')
        args['l_knots'] = args.get('l_knots', self.l_knots.tolist())
        args['tau_knots'] = args.get('tau_knots', self.tau_knots.tolist())
        args['map'] = args.get('map', {})
        args['drop_bands'] = args.get('drop_bands', [])
        args['outputdir'] = args.get('outputdir', os.path.join(os.getcwd()))
        args['outfile_prefix'] = args.get('outfile_prefix', 'output')
        args['jobid'] = args.get('jobid', False)
        pdp = args.get('private_data_path', [])
        args['private_data_path'] = [pdp] if isinstance(pdp, str) else pdp
        args['sim_prescale'] = args.get('sim_prescale', 1)
        args['jobsplit'] = args.get('jobsplit')
        if args['jobsplit'] is not None:
            args['snana'] = True
        else:
            args['jobsplit'] = [1, 1]
            args['snana'] = False
        args['jobid'] = args['jobsplit'][0]
        args['njobtot'] = args['jobsplit'][1] * args['sim_prescale']
        args['jobid'] = args['jobsplit'][0]

        if not (args['mode'] == 'fitting' and args['snana']):
            try:
                if not os.path.exists(args['outputdir']):
                    os.mkdir(args['outputdir'])
            except FileNotFoundError:
                raise FileNotFoundError('Requested output directory does not exist and could not be created')

        if 'training' in args['mode'].lower():
            self.l_knots = device_put(np.array(args['l_knots'], dtype=float))
            self._setup_band_weights()
            KD_l = invKD_irr(self.l_knots)
            self.J_l_T = device_put(spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
            self.tau_knots = device_put(np.array(args['tau_knots'], dtype=float))
            self.KD_t = device_put(invKD_irr(self.tau_knots))
        self.process_dataset(args)
        t = self.data[0, ...]
        self.hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        return args

    def run(self, args, cmd_args):
        """
        Main method to run BayeSN. Can be used for either model training, fitting or dust inference with fixed
        population SN parameters, depending on input yaml file.

        Parameters
        ----------
        args: dict
            dictionary of arguments to define model based on input yaml file
        cmd_args: dict
            dictionary of command line arguments, which will override yaml file if specified
        -------

        """
        args = self.parse_yaml_input(args, cmd_args)

        # Set up initialisation for HMC chains
        # -------------------------
        if args['initialisation'] == 'T21':
            init_strategy = init_to_value(values=self.initial_guess(args, reference_model='T21'))

        elif args['initialisation'] == 'median':
            init_strategy = init_to_median()
        elif args['initialisation'] == 'sample':
            init_strategy = init_to_sample()
        else:
            init_strategy = init_to_value(values=self.initial_guess(args, reference_model=args['initialisation']))

        print(f'Current mode: {args["mode"]}')

        if args['mode'].lower() == 'training_globalrv':
            nuts_kernel = NUTS(self.train_model_globalRV, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'training_poprv':
            nuts_kernel = NUTS(self.train_model_popRV, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'dust':
            nuts_kernel = NUTS(self.dust_model, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'dust_split_sed':
            nuts_kernel = NUTS(self.dust_model_split_sed, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'dust_split_mag':
            nuts_kernel = NUTS(self.dust_model_split_mag, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'dust_redshift':
            nuts_kernel = NUTS(self.dust_redshift_model, adapt_step_size=True, target_accept_prob=0.8,
                               init_strategy=init_strategy,
                               dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                               step_size=0.1)
        elif args['mode'].lower() == 'fitting':
            if self.model_type == 'pop_RV':
                nuts_kernel = NUTS(self.fit_model_popRV, adapt_step_size=True, init_strategy=init_strategy,
                                   max_tree_depth=10)
            elif self.model_type == 'fixed_RV':
                nuts_kernel = NUTS(self.fit_model_globalRV, adapt_step_size=True, init_strategy=init_strategy,
                                   max_tree_depth=10)
        else:
            raise ValueError("Invalid mode, must select one of 'training_globalRV', 'training_popRV', 'fitting',"
                             "'dust', 'dust_split_mag', 'dust_split_sed' or 'dust_redshift'")

        mcmc = MCMC(nuts_kernel, num_samples=args['num_samples'], num_warmup=args['num_warmup'],
                    num_chains=args['num_chains'],
                    chain_method=args['chain_method'])
        rng = PRNGKey(0)
        start = timeit.default_timer()
        # self.data, self.band_weights = self.data[..., 0:2], self.band_weights[0:2, ...]
        # print(self.data.shape)
        mcmc.run(rng, self.data, self.band_weights, extra_fields=('potential_energy',))
        end = timeit.default_timer()
        print(f'Total HMC runtime: {end - start} seconds')
        mcmc.print_summary()
        samples = mcmc.get_samples(group_by_chain=True)
        self.postprocess(samples, args)

    def postprocess(self, samples, args):
        """
        Function to postprocess BayeSN output. Applies transformations to some parameters e.g. ensuring consistency for
        W1 and theta, as flipping the sign in front of W1 and theta will lead to an identical result. Saves output
        chains and calculated a fit summary

        Parameters
        ----------
        samples: dict
            Output of MCMC, dictionary containing posterior samples for each parameter with parameter names as keys
        args: dict
            dictionary of arguments to define model based on input yaml file and command line arguments

        Returns
        -------

        """
        if 'W1' in samples.keys():  # If training
            with open(os.path.join(args['outputdir'], 'initial_chains.pkl'), 'wb') as file:
                pickle.dump(samples, file)
            # Sign flipping-----------------
            J_R = spline_coeffs_irr([6200.0], self.l_knots, invKD_irr(self.l_knots))
            J_10 = spline_coeffs_irr([10.0], self.tau_knots, invKD_irr(self.tau_knots))
            J_0 = spline_coeffs_irr([0.0], self.tau_knots, invKD_irr(self.tau_knots))
            W1 = np.reshape(samples['W1'], (
                samples['W1'].shape[0], samples['W1'].shape[1], self.l_knots.shape[0], self.tau_knots.shape[0]),
                            order='F')
            N_chains = W1.shape[0]
            sign = np.zeros(N_chains)
            for chain in range(N_chains):
                chain_W1 = np.mean(W1[chain, ...], axis=0)
                chain_sign = np.sign(
                    np.squeeze(np.matmul(J_R, np.matmul(chain_W1, J_10.T))) - np.squeeze(
                        np.matmul(J_R, np.matmul(chain_W1, J_0.T))))
                sign[chain] = chain_sign
            samples["W1"] = samples["W1"] * sign[:, None, None]
            samples["theta"] = samples["theta"] * sign[:, None, None]
            # Modify W1 and theta----------------
            theta_std = np.std(samples["theta"], axis=2)
            samples['theta'] = samples['theta'] / theta_std[..., None]
            samples['W1'] = samples['W1'] * theta_std[..., None]

            # Save best fit global params to files for easy inspection and reading in------
            W0 = np.mean(samples['W0'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]),
                                                             order='F')
            W1 = np.mean(samples['W1'], axis=[0, 1]).reshape((self.l_knots.shape[0], self.tau_knots.shape[0]),
                                                             order='F')

            L_Sigma = np.matmul(np.diag(np.mean(samples['sigmaepsilon'], axis=[0, 1])),
                                np.mean(samples['L_Omega'], axis=[0, 1]))
            sigma0 = np.mean(samples['sigma0'])

            tauA = np.mean(samples['tauA'])

            yaml_data = {
                'M0': float(self.M0),
                'SIGMA0': float(sigma0),
                'TAUA': float(tauA),
                'TAU_KNOTS': self.tau_knots.tolist(),
                'L_KNOTS': self.l_knots.tolist(),
                'W0': W0.tolist(),
                'W1': W1.tolist(),
                'L_SIGMA_EPSILON': L_Sigma.tolist()
            }

            if 'singlerv' in args['mode'].lower():
                yaml_data['RV'] = float(np.mean(samples['RV']))
            elif 'poprv' in args['mode'].lower():
                yaml_data['MUR'] = float(np.mean(samples['mu_R']))
                yaml_data['SIGMAR'] = float(np.mean(samples['sigma_R']))

            with open(os.path.join(args['outputdir'], 'bayesn.yaml'), 'w') as file:
                yaml.dump(yaml_data, file)

        z_HEL = self.data[-5, 0, :]
        muhat = self.data[-3, 0, :]

        if args['mode'] == 'fitting':
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            samples['mu'] = np.random.normal(
                (samples['Ds'] * np.power(muhat_err, 2) + muhat * np.power(self.sigma0, 2)) /
                np.power(Ds_err, 2),
                np.sqrt((np.power(self.sigma0, 2) * np.power(muhat_err, 2)) / np.power(Ds_err, 2)))
            samples['delM'] = samples['Ds'] - samples['mu']

            # Create FITRES file
            if args['snana']:
                # fetch snana version that includes tag + commit;
                # e.g., v11_05-4-gd033611.
                # Use same git command as in Makefile for C code
                SNANA_DIR = os.environ.get('SNANA_DIR', 'NULL')
                if SNANA_DIR != 'NULL':
                    cmd = f'cd {SNANA_DIR}; git describe --always --tags'
                    ret = subprocess.run([cmd], cwd=os.getcwd(), shell=True, capture_output=True, text=True)
                    snana_version = ret.stdout.replace('\n', '')
                else:
                    snana_version = 'NULL'
                self.fitres_table.meta = {'#\n# SNANA_VERSION:': snana_version,
                                          '# VERSION_PHOTOMETRY:': args['version_photometry'],
                                          '# TABLE NAME:': 'FITRES\n#'}

                n_sn = samples['mu'].shape[-1]
                summary = arviz.summary(samples)
                summary = summary[~summary.index.str.contains('tform')]
                rhat = summary.r_hat.values
                sn_rhat = np.array([rhat[i::n_sn] for i in range(n_sn)])

                self.fitres_table['MU'] = samples['mu'].mean(axis=(0, 1))
                self.fitres_table['MU_ERR'] = samples['mu'].std(axis=(0, 1))
                self.fitres_table['THETA_1'] = samples['theta'].mean(axis=(0, 1))
                self.fitres_table['THETA_1_ERR'] = samples['theta'].std(axis=(0, 1))
                self.fitres_table['AV'] = samples['AV'].mean(axis=(0, 1))
                self.fitres_table['AV_ERR'] = samples['AV'].std(axis=(0, 1))
                self.fitres_table['MEAN_RHAT'] = sn_rhat.mean(axis=1)
                self.fitres_table['MAX_RHAT'] = sn_rhat.max(axis=1)
                self.fitres_table.round(3)

                sncosmo.write_lc(self.fitres_table, f'{args["outfile_prefix"]}.FITRES.TEXT', fmt="snana",
                                 metachar="")

        if args['snana']:
            self.end_time = time.time()
            cpu_time = self.end_time - self.start_time
            # Output yaml
            out_dict = {
                'ABORT_IF_ZERO': 1,
                'SURVEY': self.survey,
                'IDSURVEY': int(self.survey_id),
                'NEVT_TOT': self.data.shape[-1],
                'NEVT_LC_CUTS': self.data.shape[-1],
                'NEVT_LCFIT_CUTS': self.data.shape[-1],
                'CPU_MINUTES': round(cpu_time / 60, 2),
            }
            with open(f'{args["outfile_prefix"]}.YAML', 'w') as file:
                yaml.dump(out_dict, file)

        if not (args['mode'] == 'fitting' and args['snana']):
            # Save convergence data for each parameter to csv file
            summary = arviz.summary(samples)
            summary.to_csv(os.path.join(args['outputdir'], 'fit_summary.csv'))

            with open(os.path.join(args['outputdir'], 'chains.pkl'), 'wb') as file:
                pickle.dump(samples, file)

            with open(os.path.join(args['outputdir'], 'input.yaml'), 'w') as file:
                yaml.dump(args, file)
        return

    def process_dataset(self, args):
        """
        Processes a data set to be used by the numpyro model.

        This will read in SNANA-format files, either in text or FITS format. This will read through all light curves and
        work out the maximum number of data points for a single object - all others will then be padded to match this
        size. This is required because to benefit from the GPU, we need to have a fixed array structure allowing us to
        calculate flux integrals from parameter values across the whole sample in a single tensor operation. A mask is
        applied in the model to ensure that these padded values do not contribute to the likelihood.

        Generated data set is saved to the SEDmodel.data attribute, while the J_t matrices used to interpolate the W0,
        W1 and epsilon matrices are also calculated and saved to the SEDmodel.J_t attribute. Observer-frame band
        weights, including the effect of Milky Way extinction, are also calculated for the data set and saved to the
        SEDmodel.band_weights attribute.

        Parameters
        ----------
        args: dict
            Combination of arguments from input yaml file and command line overrides, defines model wavelength range
            and data set to load

        """
        if 'version_photometry' not in args.keys() and 'data_table' not in args.keys():
            raise ValueError('Please pass either data_dir (for a directory containing all SNANA files such as a '
                             'simulation output) or a combination of data_table and data_root')
        if 'data_table' in args.keys() and 'data_root' not in args.keys():
            raise ValueError('If using data_table, please also pass data_root (which defines the location that the '
                             'paths in data_table are defined with respect to)')
        survey_dict = {}
        if 'version_photometry' in args.keys():  # If using all files in directory
            data_dir = args['version_photometry']
            if args['snana']:  # Assuming you're using SNANA running on Perlmutter or a similar cluster
                # Look in standard public repositories for real data/simulations
                dir_list = ['SNDATA_ROOT/lcmerge', 'SNDATA_ROOT/SIM']
                sim_list = np.loadtxt(os.path.join(os.environ.get('SNDATA_ROOT'), 'SIM', 'PATH_SNDATA_SIM.LIST'), dtype=str)
                dir_list = dir_list + list([sim_dir[1:] for sim_dir in sim_list])
                pdp = [path[1:] if path[0] == '$' else path for path in args['private_data_path']]
                dir_list = dir_list + pdp  # Add any private data directories
                found_in = []
                for dir in dir_list:
                    root_split = dir.split('/')
                    root, remainder = root_split[0], ''.join(root_split[1:])
                    if not os.path.isabs(dir):
                        root = os.environ.get(root, 'NULL')
                    if os.path.exists(os.path.join(root, remainder, data_dir)):
                        found_in.append(os.path.join(root, remainder, data_dir))
                if len(found_in) == 0:
                    raise ValueError(f'Requested photometry {data_dir} was not found in any of the usual public '
                                     f'locations, maybe you need to specify an additional private data location')
                elif len(found_in) > 1:
                    raise ValueError(f'Requested photometry {data_dir} was found in multiple locations, please remove '
                                     f'duplicates and ensure the one you want to use remains')
                data_dir = found_in[0]
                # Load up SNANA survey definitions file
                survey_def_path = os.path.join(os.environ.get('SNDATA_ROOT'), 'SURVEY.DEF')
                with open(survey_def_path) as fp:
                    for line in fp:
                        if line[:line.find(':')] == 'SURVEY':
                            split = line.split()
                            survey_dict[split[1]] = split[2]
            sample_name = os.path.split(data_dir)[-1]
            list_file = os.path.join(data_dir, f'{os.path.split(data_dir)[-1]}.LIST')
            sn_list = np.atleast_1d(np.loadtxt(list_file, dtype='str'))
            file_format = sn_list[0].split('.')[1]
            map_dict = args['map']
            n_obs = []
            all_lcs = []
            t_ranges = []
            sne = []
            # For FITRES table
            idsurvey, sn_type, field, cutflag_snana, z_hels, z_hel_errs, z_hds, z_hd_errs = [], [], [], [], [], [], [], []
            snrmax1s, snrmax2s, snrmax3s = [], [], []
            vpecs, vpec_errs, mwebvs, host_logmasses, host_logmass_errs = [], [], [], [], []
            # --------
            used_bands, used_band_dict = ['NULL_BAND'], {0: 0}
            print('Reading light curves...')
            if file_format.lower() == 'fits':  # If FITS format
                ntot = 0
                # Check if sim or real data
                # if not os.path.exists
                head_file = os.path.join(data_dir, f'{sn_list[0]}')
                if not os.path.exists(head_file):
                    head_file = os.path.join(data_dir, f'{sn_list[0]}.gz')  # Look for .fits.gz if .fits not found
                phot_file = head_file.replace("HEAD", "PHOT")
                sne_file = sncosmo.read_snana_fits(head_file, phot_file)
                # If real data, ignore sim_prescale
                if 'SIM_REDSHIFT_HELIO' not in sne_file[0].meta.keys():
                    args['njobtot'] = args['jobsplit'][0]
                for sn_file in tqdm(sn_list):
                    head_file = os.path.join(data_dir, f'{sn_file}')
                    if not os.path.exists(head_file):
                        head_file = os.path.join(data_dir, f'{sn_file}.gz')  # Look for .fits.gz if .fits not found
                    with fits.open(head_file) as hdu:
                        self.survey = hdu[0].header.get('SURVEY', 'NULL')
                    self.survey_id = survey_dict.get(self.survey, 0)
                    phot_file = head_file.replace("HEAD", "PHOT")
                    sne_file = sncosmo.read_snana_fits(head_file, phot_file)
                    for sn_ind in range(len(sne_file)):
                        ntot += 1
                        if (ntot - args['jobid']) % args['njobtot'] != 0:
                            continue
                        sn = sne_file[sn_ind]
                        meta, data = sn.meta, sn.to_pandas()
                        data['BAND'] = data.BAND.str.decode("utf-8")
                        data['BAND'] = data.BAND.str.strip()
                        peak_mjd = meta['PEAKMJD']
                        zhel = meta['REDSHIFT_HELIO']
                        zcmb = meta['REDSHIFT_FINAL']
                        zhel_err = 5e-4  # Need to handle this better if not defined
                        zcmb_err = 5e-4  # Need to handle this better if not defined
                        data['t'] = (data.MJD - peak_mjd) / (1 + zhel)
                        # If filter not in map_dict, assume one-to-one mapping------
                        for f in data.BAND.unique():
                            if f not in map_dict.keys():
                                map_dict[f] = f
                        data['FLT'] = data.BAND.apply(lambda x: map_dict[x])
                        # Remove bands outside of filter coverage-------------------
                        for f in data.FLT.unique():
                            if zhel > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or zhel < (
                                    self.band_lim_dict[f][1] / self.l_knots[-1] - 1):
                                data = data[~data.FLT.isin([f])]
                        # Record all used bands-------------------------------------
                        for f in data.FLT.unique():
                            if f not in used_bands:
                                used_bands.append(f)
                                try:
                                    used_band_dict[self.band_dict[f]] = len(used_bands) - 1
                                except KeyError:
                                    raise KeyError(
                                        f'Filter {f} not present in BayeSN, check your filter mapping')
                        # ----------------------------------------------------------
                        data['band_indices'] = data.FLT.apply(lambda x: used_band_dict[self.band_dict[x]])
                        data['zp'] = data.FLT.apply(lambda x: self.zp_dict[x])
                        data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                        data['MAGERR'] = (2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL']
                        data['flux'] = data['FLUXCAL']  # np.power(10, -0.4 * (data['MAG'] - data['zp'])) * self.scale
                        data['flux_err'] = data['FLUXCALERR']  # (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
                        data['redshift'] = zhel
                        data['redshift_error'] = meta.get('REDSHIFT_CMB_ERR', 5e-4)  # Made up default if not specified
                        data['MWEBV'] = meta['MWEBV']
                        data['mass'] = meta.get('HOSTGAL_LOGMASS', -9.)
                        data['dist_mod'] = self.cosmo.distmod(zcmb)
                        data['mask'] = 1
                        lc = data[
                            ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'mass', 'band_indices', 'redshift',
                             'redshift_error', 'dist_mod', 'MWEBV', 'mask']]
                        lc = lc.dropna(subset=['flux', 'flux_err'])
                        lc = lc[(lc['t'] > -10) & (lc['t'] < 40)]
                        if lc.empty:  # Skip empty light curves, maybe they don't have any data in [-10, 40] days
                            continue
                        t_ranges.append((lc['t'].min(), lc['t'].max()))
                        n_obs.append(lc.shape[0])
                        all_lcs.append(lc)
                        # Set up FITRES table data
                        # (currently just uses second table, should improve for cases where there are multiple lc files)
                        sn_name = meta['SNID']
                        if isinstance(sn_name, bytes):
                            sn_name = sn_name.decode('utf-8')
                        sne.append(sn_name)
                        sn_type.append(meta.get('TYPE', 0))
                        field.append(meta.get('FIELD', 'VOID'))
                        z_hels.append(zhel)
                        z_hel_errs.append(meta.get('REDSHIFT_HELIO_ERR', zhel_err))
                        z_hds.append(meta['REDSHIFT_FINAL'])
                        z_hd_errs.append(meta.get('REDSHIFT_FINAL_ERR', zcmb_err))
                        vpecs.append(meta.get('VPEC', 0.))
                        vpec_errs.append(meta.get('VPEC_ERR', 0.))
                        mwebvs.append(meta.get('MWEBV', -9.))
                        host_logmasses.append(meta.get('HOSTGAL_LOGMASS', -9.))
                        host_logmass_errs.append(meta.get('HOSTGAL_LOGMASS_ERR', -9.))
                        snrmax1 = np.max(lc.flux / lc.flux_err)
                        lc_snr2 = lc[lc.band_indices != lc[(lc.flux / lc.flux_err) == snrmax1].band_indices.values[0]]
                        if lc_snr2.shape[0] == 0:
                            snrmax2 = -99
                            snrmax3 = -99
                        else:
                            snrmax2 = np.max(lc_snr2.flux / lc_snr2.flux_err)
                            lc_snr3 = lc_snr2[lc_snr2.band_indices !=
                                              lc_snr2[(lc_snr2.flux / lc_snr2.flux_err) == snrmax2].band_indices.values[
                                                  0]]
                            if lc_snr3.shape[0] == 0:
                                snrmax3 = -99
                            else:
                                snrmax3 = np.max(lc_snr3.flux / lc_snr3.flux_err)
                        snrmax1s.append(snrmax1)
                        snrmax2s.append(snrmax2)
                        snrmax3s.append(snrmax3)
            else:  # If not FITS, assume text format
                # Check if sim or real data
                meta, lcdata = sncosmo.read_snana_ascii(os.path.join(data_dir, sn_list[0]), default_tablename='OBS')
                # If real data, ignore sim_prescale
                if 'SIM_REDSHIFT_HELIO' not in meta.keys():
                    args['njobtot'] = args['jobsplit'][0]
                for sn_ind, sn_file in tqdm(enumerate(sn_list), total=len(sn_list)):
                    if (sn_ind + 1 - args['jobid']) % args['njobtot'] != 0:
                        continue
                    meta, lcdata = sncosmo.read_snana_ascii(os.path.join(data_dir, sn_file), default_tablename='OBS')
                    data = lcdata['OBS'].to_pandas()
                    peak_mjd = meta['PEAKMJD']
                    sn_name = meta['SNID']
                    if isinstance(sn_name, bytes):
                        sn_name = sn_name.decode('utf-8')
                    sne.append(sn_name)
                    zhel = meta['REDSHIFT_HELIO']
                    zcmb = meta['REDSHIFT_FINAL']
                    zhel_err = 5e-4  # Placeholder in case value is not defined in meta, need to handle this better
                    zcmb_err = 5e-4  # Placeholder in case value is not defined in meta, need to handle this better
                    data['t'] = (data.MJD - peak_mjd) / (1 + zhel)
                    # If filter not in map_dict, assume one-to-one mapping------
                    map_dict = args['map']
                    for f in data.BAND.unique():
                        if f not in map_dict.keys():
                            map_dict[f] = f
                    data['FLT'] = data.BAND.apply(lambda x: map_dict[x])

                    # Remove bands outside of filter coverage-------------------
                    for f in data.FLT.unique():
                        if zhel > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or zhel < (
                                self.band_lim_dict[f][1] / self.l_knots[-1] - 1):
                            data = data[~data.FLT.isin([f])]
                    # Record all used bands-------------------------------------
                    for f in data.FLT.unique():
                        if f not in used_bands:
                            used_bands.append(f)
                            try:
                                used_band_dict[self.band_dict[f]] = len(used_bands) - 1
                            except KeyError:
                                raise KeyError(
                                    f'Filter {f} not present in BayeSN, check your filter mapping')
                    # ----------------------------------------------------------
                    data['band_indices'] = data.FLT.apply(lambda x: used_band_dict[self.band_dict[x]])
                    data['zp'] = data.FLT.apply(lambda x: self.zp_dict[x])
                    data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                    data['MAGERR'] = np.abs((2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL'])
                    data['flux'] = data['FLUXCAL']  # np.power(10, -0.4 * (data['MAG'] - data['zp']))
                    data['flux_err'] = data['FLUXCALERR']  # (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
                    data['redshift'] = zhel
                    data['redshift_error'] = meta.get('REDSHIFT_CMB_ERR', 5e-4)  # Made up default if not specified
                    data['MWEBV'] = meta['MWEBV']
                    data['mass'] = meta.get('HOSTGAL_LOGMASS', -9.)
                    data['dist_mod'] = self.cosmo.distmod(zcmb)
                    data['mask'] = 1
                    lc = data[
                        ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'mass', 'band_indices', 'redshift', 'redshift_error',
                         'dist_mod', 'MWEBV', 'mask']]
                    lc = lc.dropna(subset=['flux', 'flux_err'])
                    lc = lc[(lc['t'] > self.tau_knots.min()) & (lc['t'] < self.tau_knots.max())]
                    t_ranges.append((lc['t'].min(), lc['t'].max()))
                    n_obs.append(lc.shape[0])
                    all_lcs.append(lc)
                    # Set up FITRES table data
                    # (currently just uses second table, should improve for cases where there are multiple lc files)
                    sn_type.append(meta.get('TYPE', 0))
                    field.append(meta.get('FIELD', 'VOID'))
                    z_hels.append(zhel)
                    z_hel_errs.append(meta.get('REDSHIFT_HELIO_ERR', zhel_err))
                    z_hds.append(meta['REDSHIFT_FINAL'])
                    z_hd_errs.append(meta.get('REDSHIFT_FINAL_ERR', zcmb_err))
                    vpecs.append(meta.get('VPEC', 0.))
                    vpec_errs.append(meta.get('VPEC_ERR', 0.))
                    mwebvs.append(meta.get('MWEBV', -9.))
                    host_logmasses.append(meta.get('HOSTGAL_LOGMASS', -9.))
                    host_logmass_errs.append(meta.get('HOSTGAL_LOGMASS_ERR', -9.))
                    snrmax1 = np.max(lc.flux / lc.flux_err)
                    lc_snr2 = lc[lc.band_indices != lc[(lc.flux / lc.flux_err) == snrmax1].band_indices.values[0]]
                    if lc_snr2.shape[0] == 0:
                        snrmax2 = -99
                        snrmax3 = -99
                    else:
                        snrmax2 = np.max(lc_snr2.flux / lc_snr2.flux_err)
                        lc_snr3 = lc_snr2[lc_snr2.band_indices !=
                                      lc_snr2[(lc_snr2.flux / lc_snr2.flux_err) == snrmax2].band_indices.values[0]]
                        if lc_snr3.shape[0] == 0:
                            snrmax3 = -99
                        else:
                            snrmax3 = np.max(lc_snr3.flux / lc_snr3.flux_err)
                    snrmax1s.append(snrmax1)
                    snrmax2s.append(snrmax2)
                    snrmax3s.append(snrmax3)
                self.survey = meta.get('SURVEY', 'NULL')
                self.survey_id = survey_dict.get(self.survey, 0)
            N_sn = len(all_lcs)
            N_obs = np.max(n_obs)
            N_col = lc.shape[1]
            all_data = np.zeros((N_sn, N_obs, N_col))
            print('Saving light curves to standard grid...')
            for i in tqdm(range(len(all_lcs))):
                lc = all_lcs[i]
                all_data[i, :lc.shape[0], :] = lc.values
                all_data[i, lc.shape[0]:, 2] = 1 / jnp.sqrt(2 * np.pi)
            all_data = all_data.T
            t = all_data[0, ...]
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
            flux_data = all_data[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11], ...]
            mag_data = all_data[[0, 3, 4, 5, 6, 7, 8, 9, 10, 11], ...]
            # Mask out negative fluxes, only for mag data--------------------------
            for i in range(len(all_lcs)):
                mag_data[:2, (flux_data[1, ...] <= 0)] = 0  # Mask out photometry
                mag_data[4, (flux_data[1, ...] <= 0)] = 0  # Mask out band
                mag_data[-1, (flux_data[1, ...] <= 0)] = 0  # Set mask row
                mag_data[2, (flux_data[1, ...] <= 0)] = 1 / jnp.sqrt(2 * np.pi)
            # ---------------------------------------------------------------------
            if 'training' in args['mode'].lower():
                self.data = device_put(mag_data)
            else:
                self.data = device_put(flux_data)
            self.sn_list = sne
            self.J_t = device_put(J_t)
            self.used_band_inds = jnp.array([self.band_dict[f] for f in used_bands])
            self.zps = self.zps[self.used_band_inds]
            self.offsets = self.offsets[self.used_band_inds]
            self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])
            # Prep FITRES table
            varlist = ["SN:"] * len(sne)
            idsurvey = [self.survey_id] * len(sne)
            snrmax1s, snrmax2s, snrmax3s = np.array(snrmax1s), np.array(snrmax2s), np.array(snrmax3s)
            table = QTable([varlist, sne, idsurvey, sn_type, field, z_hels, z_hel_errs, z_hds, z_hd_errs,
                            vpecs, vpec_errs, mwebvs, host_logmasses, host_logmass_errs, snrmax1s, snrmax2s, snrmax3s],
                           names=['VARNAMES:', 'CID', 'IDSURVEY', 'TYPE', 'FIELD', 'zHEL', 'zHELERR',
                                  'zHD', 'zHDERR', 'VPEC', 'VPECERR', 'MWEBV', 'HOST_LOGMASS', 'HOST_LOGMASS_ERR',
                                  'SNRMAX1', 'SNRMAX2', 'SNRMAX3'])
            self.fitres_table = table
        else:
            table_path = os.path.join(args['data_root'], args['data_table'])
            sn_list = pd.read_csv(table_path, comment='#', delim_whitespace=True)
            n_obs = []

            all_lcs = []
            t_ranges = []
            # For FITRES table
            idsurvey, sn_type, field, cutflag_snana, z_hels, z_hel_errs, z_hds, z_hd_errs = [], [], [], [], [], [], [], []
            snrmax1s, snrmax2s, snrmax3s = [], [], []
            vpecs, vpec_errs, mwebvs, host_logmasses, host_logmass_errs = [], [], [], [], []
            # --------
            used_bands, used_band_dict = ['NULL_BAND'], {0: 0}
            print('Reading light curves...')
            for i in tqdm(range(sn_list.shape[0])):
                row = sn_list.iloc[i]
                sn_files = row.files.split(',')
                sn_lc = None
                sn = row.SNID
                data_root = args['data_root']
                for file in sn_files:
                    meta, lcdata = sncosmo.read_snana_ascii(os.path.join(data_root, file), default_tablename='OBS')
                    data = lcdata['OBS'].to_pandas()
                    if 'SEARCH_PEAKMJD' in sn_list.columns:
                        peak_mjd = row.SEARCH_PEAKMJD
                    else:
                        peak_mjd = meta['SEARCH_PEAKMJD']
                    if 'BAND' in data.columns:  # This column can have different names which can be confusing, let's
                                                # just rename it so it's always the same
                        data = data.rename(columns={'BAND': 'FLT'})
                    data = data[~data.FLT.isin(args['drop_bands'])]  # Skip certain bands
                    zhel = meta['REDSHIFT_HELIO']
                    data['t'] = (data.MJD - peak_mjd) / (1 + zhel)
                    # If filter not in map_dict, assume one-to-one mapping------
                    map_dict = args['map']
                    for f in data.FLT.unique():
                        if f not in map_dict.keys():
                            map_dict[f] = f
                    data['FLT'] = data.FLT.apply(lambda x: map_dict[x])
                    # Remove bands outside of filter coverage-------------------
                    for f in data.FLT.unique():
                        if zhel > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or zhel < (
                                self.band_lim_dict[f][1] / self.l_knots[-1] - 1):
                            data = data[~data.FLT.isin([f])]
                    # Record all used bands-------------------------------------
                    for f in data.FLT.unique():
                        if f not in used_bands:
                            used_bands.append(f)
                            try:
                                used_band_dict[self.band_dict[f]] = len(used_bands) - 1
                            except KeyError:
                                raise KeyError(
                                    f'Filter {f} not present in BayeSN, check your filter mapping')
                    # ----------------------------------------------------------
                    data['band_indices'] = data.FLT.apply(lambda x: used_band_dict[self.band_dict[x]])
                    data['zp'] = data.FLT.apply(lambda x: self.zp_dict[x])
                    data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                    data['MAGERR'] = (2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL']
                    data['flux'] = data['FLUXCAL']
                    data['flux_err'] = data['FLUXCALERR']
                    data['redshift'] = zhel
                    data['redshift_error'] = row.REDSHIFT_CMB_ERR
                    data['MWEBV'] = meta['MWEBV']
                    data['mass'] = meta['HOSTGAL_LOGMASS']
                    data['dist_mod'] = self.cosmo.distmod(row.REDSHIFT_CMB)
                    data['mask'] = 1
                    lc = data[
                        ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'mass', 'band_indices', 'redshift', 'redshift_error',
                         'dist_mod', 'MWEBV', 'mask']]
                    lc = lc.dropna(subset=['flux', 'flux_err'])
                    lc = lc[(lc['t'] > self.tau_knots.min()) & (lc['t'] < self.tau_knots.max())]
                    if sn_lc is None:
                        sn_lc = lc.copy()
                    else:
                        sn_lc = pd.concat([sn_lc, lc])
                t_ranges.append((lc['t'].min(), lc['t'].max()))
                n_obs.append(lc.shape[0])
                all_lcs.append(sn_lc)
                # Set up FITRES table data
                # (currently just uses second table, should improve for cases where there are multiple lc files)
                idsurvey.append(meta.get('IDSURVEY', 'NULL'))
                sn_type.append(meta.get('TYPE', 'NULL'))
                field.append(meta.get('FIELD', 'NULL'))
                cutflag_snana.append(meta.get('CUTFLAG_SNANA', 'NULL'))
                z_hels.append(zhel)
                z_hel_errs.append(meta.get('REDSHIFT_HELIO_ERR', row.REDSHIFT_CMB_ERR))
                z_hds.append(row.REDSHIFT_CMB)
                z_hd_errs.append(row.REDSHIFT_CMB_ERR)
                vpecs.append(meta.get('VPEC', 'NULL'))
                vpec_errs.append(meta.get('VPEC_ERR', 'NULL'))
                mwebvs.append(meta.get('MWEBV', 'NULL'))
                host_logmasses.append(meta.get('HOSTGAL_LOGMASS', 'NULL'))
                host_logmass_errs.append(meta.get('HOSTGAL_LOGMASS_ERR', 'NULL'))
                snrmax1 = np.max(lc.flux / lc.flux_err)
                lc_snr2 = lc[lc.band_indices != lc[(lc.flux / lc.flux_err) == snrmax1].band_indices.values[0]]
                snrmax2 = np.max(lc_snr2.flux / lc_snr2.flux_err)
                lc_snr3 = lc_snr2[lc_snr2.band_indices != lc_snr2[(lc_snr2.flux / lc_snr2.flux_err) == snrmax2].band_indices.values[0]]
                snrmax3 = np.max(lc_snr3.flux / lc_snr3.flux_err)
                snrmax1s.append(snrmax1)
                snrmax2s.append(snrmax2)
                snrmax3s.append(snrmax3)
            N_sn = sn_list.shape[0]
            N_obs = np.max(n_obs)
            N_col = lc.shape[1]
            all_data = np.zeros((N_sn, N_obs, N_col))
            print('Saving light curves to standard grid...')
            for i in tqdm(range(len(all_lcs))):
                lc = all_lcs[i]
                all_data[i, :lc.shape[0], :] = lc.values
                all_data[i, lc.shape[0]:, 2] = 1 / jnp.sqrt(2 * np.pi)
                # all_data[i, lc.shape[0]:, 3] = 10  # Arbitrarily set all masked points to H-band
            all_data = all_data.T
            t = all_data[0, ...]
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
            flux_data = all_data[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11], ...]
            mag_data = all_data[[0, 3, 4, 5, 6, 7, 8, 9, 10, 11], ...]
            # Mask out negative fluxes, only for mag data--------------------------
            for i in range(len(all_lcs)):
                mag_data[:2, (flux_data[1, ...] <= 0)] = 0  # Mask out photometry
                mag_data[4, (flux_data[1, ...] <= 0)] = 0  # Mask out band
                mag_data[-1, (flux_data[1, ...] <= 0)] = 0  # Set mask row
                mag_data[2, (flux_data[1, ...] <= 0)] = 1 / jnp.sqrt(2 * np.pi)
            # ---------------------------------------------------------------------
            sne = sn_list['SNID'].values
            self.sn_list = sne
            if 'training' in args['mode'].lower():
                self.data = device_put(mag_data)
            else:
                self.data = device_put(flux_data)
            self.J_t = device_put(J_t)
            self.used_band_inds = jnp.array([self.band_dict[f] for f in used_bands])
            self.zps = self.zps[self.used_band_inds]
            self.offsets = self.offsets[self.used_band_inds]
            self.band_weights = self._calculate_band_weights(self.data[-5, 0, :], self.data[-2, 0, :])

            # Prep FITRES table
            varlist = ["SN:"] * len(sne)
            snrmax1s, snrmax2s, snrmax3s = np.array(snrmax1s), np.array(snrmax2s), np.array(snrmax3s)
            snrmax1s, snrmax2s, snrmax3s = np.around(snrmax1s, 2), np.around(snrmax2s, 2), np.around(snrmax3s, 2)
            table = QTable([varlist, sne, idsurvey, sn_type, field, z_hels, z_hel_errs, z_hds, z_hd_errs,
                            vpecs, vpec_errs, mwebvs, host_logmasses, host_logmass_errs, snrmax1s, snrmax2s, snrmax3s],
                           names=['VARLIST:', 'CID', 'IDSURVEY', 'TYPE', 'FIELD', 'zHEL', 'zHELERR', 'zHD',
                                  'zHDERR', 'VPEC', 'VPECERR', 'MWEBV', 'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 'SNRMAX1',
                                  'SNRMAX2', 'SNRMAX3'])
            self.fitres_table = table

    def simulate_spectrum(self, t, N, dl=10, z=0, mu=0, ebv_mw=0, RV=None, logM=None, del_M=None, AV=None, theta=None,
                          eps=None):
        """
        Simulates spectra for given parameter values in the observer-frame. If parameter values are not set, model
        priors will be sampled.

        Parameters
        ----------
        t: array-like
            Set of t values to simulate spectra at
        N: int
            Number of separate objects to simulate spectra for
        dl: float, optional
            Wavelength spacing for simulated spectra in rest-frame. Default is 10 AA
        z: float or array-like, optional
            Redshift to simulate spectra at, affecting observer-frame wavelengths and reducing spectra by factor of
            (1+z). Defaults to 0. If passing an array-like object, there must be a corresponding value for each of the N
            simulated objects. If a float is passed, the same redshift will be used for all objects.
        mu: float, array-like or str, optional
            Distance modulus to simulate spectra at. Defaults to 0. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. If set to 'z', distance moduli corresponding to the redshift values passed in the default
            model cosmology will be used.
        ebv_mw: float or array-like, optional
            Milky Way E(B-V) values for simulated spectra. Defaults to 0. If passing an array-like object, there must be
            a corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects.
        RV: float or array-like, optional
            RV values for host extinction curves for simulated spectra. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the global RV value for the BayeSN model loaded when
            initialising SEDmodel will be used.
        logM: float or array-like, optional
            Currently unused, will be implemented when split models are included
        del_M: float or array-like, optional
            Grey offset del_M value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        AV: float or array-like, optional
            Host extinction RV value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        theta: float or array-like, optional
            Theta value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        eps: array-like or int, optional
            Epsilon values to be used for each SN. If passing a 2d array, this must be of shape (l_knots, tau_knots)
            and will be used for each SN generated. If passing a 3d array, this must be of shape (N, l_knots, tau_knots)
            and provide an epsilon value for each generated SN. You can also pass 0, in which case an array of zeros of
            shape (N, l_knots, tau_knots) will be used and epsilon is effectively turned off. Defaults to None, in which
            case the prior distribution will be sampled for each object.

        Returns
        -------

        l_o: array-like
            Array of observer-frame wavelength values
        spectra: array-like
            Array of simulated spectra
        param_dict: dict
            Dictionary of corresponding parameter values for each simulated object

        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.array(del_M)
            if len(del_M.shape) == 0:
                del_M = del_M.repeat(N)
            elif del_M.shape[0] != N:
                raise ValueError('If not providing a scalar del_M value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.array(AV)
            if len(AV.shape) == 0:
                AV = AV.repeat(N)
            elif AV.shape[0] != N:
                raise ValueError('If not providing a scalar AV value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.array(theta)
            if len(theta.shape) == 0:
                theta = theta.repeat(N)
            elif theta.shape[0] != N:
                raise ValueError('If not providing a scalar theta value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if eps is None:
            eps = self.sample_epsilon(N)
        else:
            eps = np.array(eps)
            if len(eps.shape) == 0:
                if eps == 0:
                    eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
                else:
                    raise ValueError(
                        'For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots). The only scalar '
                        'value accepted is 0, which will effectively remove the effect of epsilon')
            elif len(eps.shape) == 2 and eps.shape[0] == self.l_knots.shape[0] and eps.shape[1] == self.tau_knots.shape[
                0]:
                eps = eps[None, ...].repeat(N, axis=0)
            elif len(eps.shape) != 3 or eps.shape[0] != N or eps.shape[1] != self.l_knots.shape[0] or eps.shape[2] != \
                    self.tau_knots.shape[0]:
                raise ValueError('For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots)')
        ebv_mw = np.array(ebv_mw)
        if len(ebv_mw.shape) == 0:
            ebv_mw = ebv_mw.repeat(N)
        elif ebv_mw.shape[0] != N:
            raise ValueError(
                'For ebv_mw, either pass a single scalar value or an array of values for each of the N simulated objects')
        if RV is None:
            RV = self.RV
        RV = np.array(RV)
        if len(RV.shape) == 0:
            RV = RV.repeat(N)
        elif RV.shape[0] != N:
            raise ValueError(
                'For RV, either pass a single scalar value or an array of values for each of the N simulated objects')
        z = np.array(z)
        if len(z.shape) == 0:
            z = z.repeat(N)
        elif z.shape[0] != N:
            raise ValueError(
                'For z, either pass a single scalar value or an array of values for each of the N simulated objects')
        mu = np.array(mu)
        if len(mu.shape) == 0:
            mu = mu.repeat(N)
        elif mu.shape[0] != N:
            raise ValueError(
                'For mu, either pass a single scalar value or an array of values for each of the N simulated objects')
        param_dict = {
            'del_M': del_M,
            'AV': AV,
            'theta': theta,
            'eps': eps,
            'z': z,
            'mu': mu,
            'ebv_mw': ebv_mw,
            'RV': RV
        }
        l_r = np.arange(min(self.l_knots), max(self.l_knots) + dl, dl, dtype=float)
        l_r = l_r[l_r <= max(self.l_knots)]
        l_o = l_r[None, ...].repeat(N, axis=0) * (1 + z[:, None])

        self.model_wave = l_r
        KD_l = invKD_irr(self.l_knots)
        self.J_l_T = device_put(spline_coeffs_irr(self.model_wave, self.l_knots, KD_l))
        KD_x = invKD_irr(self.xk)
        self.M_fitz_block = device_put(spline_coeffs_irr(1e4 / self.model_wave, self.xk, KD_x))
        self._load_hsiao_template()

        t = jnp.array(t)
        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        keep_shape = t.shape
        t = t.flatten(order='F')
        map = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,2,0)
        spectra = self.get_spectra(theta, AV, self.W0, self.W1, eps, RV, J_t, hsiao_interp)

        # Host extinction
        host_ext = np.zeros((N, l_r.shape[0], 1))
        for i in range(N):
            host_ext[i, :, 0] = extinction.fitzpatrick99(l_r, AV[i], RV[i])

        # MW extinction
        mw_ext = np.zeros((N, l_o.shape[1], 1))
        for i in range(N):
            mw_ext[i, :, 0] = extinction.fitzpatrick99(l_o[i, ...], 3.1 * ebv_mw[i], 3.1)

        return l_o, spectra, param_dict

    def simulate_light_curve(self, t, N, bands, yerr=0, err_type='mag', z=0, zerr=1e-4, mu=0, ebv_mw=0, RV=None,
                             logM=None, del_M=None, AV=None, theta=None, eps=None, mag=True, write_to_files=False,
                             output_dir=None):
        """
        Simulates light curves from the BayeSN model in either mag or flux space. and saves them to SNANA-format text
        files if requested

        Parameters
        ----------
        t: array-like
            Set of t values to simulate spectra at. If len(t) == len(bands), will assume that the t values
            correspond to the bands. Otherwise, will simulate photometry at each value of t for each band.
        N: int
            Number of separate objects to simulate spectra for
        bands: array-like
            List of bands in which to simulate photometry. If len(t) == len(bands), will assume that the t values
            correspond to the bands. Otherwise, will simulate photometry at each value of t for each band.
        yerr: float or array-like, optional
            Uncertainties for each data point, simulated light curves will be randomised assuming a Gaussian uncertainty
            around the true values. Can be either a float, meaning that the same value will be used for each data point,
            a 1d array of length equal to each light curve, meaning that these values will be used for each simulated
            light curve, or a 2d array of shape (N, light curve length) allowing you to specify each individual error.
            Defaults to 0, meaning that exact model photometry will be returned.
        err_type: str
            Specifies which type of error you are passing, either 'mag' or 'flux'. Defaults to 'mag', meaning that this
            is in mag units. If you want to simulate fluxes and pass a mag error, it will be converted to a flux error.
        z: float or array-like, optional
            Redshift to simulate spectra at, affecting observer-frame wavelengths and reducing spectra by factor of
            (1+z). Defaults to 0. If passing an array-like object, there must be a corresponding value for each of the N
            simulated objects. If a float is passed, the same redshift will be used for all objects.
        zerr: float, optional
            Error on spectroscopic redshifts, only needed when saving to SNANA-format light curve files
        mu: float, array-like or str, optional
            Distance modulus to simulate spectra at. Defaults to 0. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. If set to 'z', distance moduli corresponding to the redshift values passed in the default
            model cosmology will be used. Technically these are heliocentric redshifts rather than Hubble diagram
            redshifts so won't be perfect, but can be useful sometimes.
        ebv_mw: float or array-like, optional
            Milky Way E(B-V) values for simulated spectra. Defaults to 0. If passing an array-like object, there must be
            a corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects.
        RV: float or array-like, optional
            RV values for host extinction curves for simulated spectra. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the global RV value for the BayeSN model loaded when
            initialising SEDmodel will be used.
        logM: float or array-like, optional
            Currently unused, will be implemented when split models are included
        del_M: float or array-like, optional
            Grey offset del_M value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        AV: float or array-like, optional
            Host extinction RV value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        theta: float or array-like, optional
            Theta value to be used for each SN. If passing an array-like object, there must be a
            corresponding value for each of the N simulated objects. If a float is passed, the same value will be used
            for all objects. Defaults to None, in which case the prior distribution will be sampled for each object.
        eps: array-like or int, optional
            Epsilon values to be used for each SN. If passing a 2d array, this must be of shape (l_knots, tau_knots)
            and will be used for each SN generated. If passing a 3d array, this must be of shape (N, l_knots, tau_knots)
            and provide an epsilon value for each generated SN. You can also pass 0, in which case an array of zeros of
            shape (N, l_knots, tau_knots) will be used and epsilon is effectively turned off. Defaults to None, in which
            case the prior distribution will be sampled for each object.
        mag: Bool, optional
            Determines whether returned values are mags or fluxes
        write_to_files: Bool, optional
            Determines whether to save simulated light curves to SNANA-format light curve files, defaults to False
        output_dir: str, optional
            Path to output directory to save simulated SNANA-format files, onl required if write_to_files=True

        Returns
        -------
        data: array-like
            Array containing simulated flux or mag values
        yerr: array-like
            Aray containing corresponding errors for each data point
        param_dict: dict
            Dictionary of corresponding parameter values for each simulated object

        """
        if del_M is None:
            del_M = self.sample_del_M(N)
        else:
            del_M = np.array(del_M)
            if len(del_M.shape) == 0:
                del_M = del_M.repeat(N)
            elif del_M.shape[0] != N:
                raise ValueError('If not providing a scalar del_M value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if AV is None:
            AV = self.sample_AV(N)
        else:
            AV = np.array(AV)
            if len(AV.shape) == 0:
                AV = AV.repeat(N)
            elif AV.shape[0] != N:
                raise ValueError('If not providing a scalar AV value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if theta is None:
            theta = self.sample_theta(N)
        else:
            theta = np.array(theta)
            if len(theta.shape) == 0:
                theta = theta.repeat(N)
            elif theta.shape[0] != N:
                raise ValueError('If not providing a scalar theta value, array must be of same length as the number of '
                                 'objects to simulate, N')
        if eps is None:
            eps = self.sample_epsilon(N)
        elif len(np.array(eps).shape) == 0:
            eps = np.array(eps)
            if eps == 0:
                eps = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
            else:
                raise ValueError(
                    'For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots). The only scalar '
                    'value accepted is 0, which will effectively remove the effect of epsilon')
        elif len(eps.shape) != 3 or eps.shape[0] != N or eps.shape[1] != self.l_knots.shape[0] or eps.shape[2] != \
                self.tau_knots.shape[0]:
            raise ValueError('For epsilon, please pass an array-like object of shape (N, l_knots, tau_knots)')
        ebv_mw = np.array(ebv_mw)
        if len(ebv_mw.shape) == 0:
            ebv_mw = ebv_mw.repeat(N)
        elif ebv_mw.shape[0] != N:
            raise ValueError(
                'For ebv_mw, either pass a single scalar value or an array of values for each of the N simulated objects')
        if RV is None:
            RV = self.RV
        RV = np.array(RV)
        if len(RV.shape) == 0:
            RV = RV.repeat(N)
        elif RV.shape[0] != N:
            raise ValueError(
                'For RV, either pass a single scalar value or an array of values for each of the N simulated objects')
        z = np.array(z)
        if len(z.shape) == 0:
            z = z.repeat(N)
        elif z.shape[0] != N:
            raise ValueError(
                'For z, either pass a single scalar value or an array of values for each of the N simulated objects')
        if type(mu) == str and mu == 'z':
            mu = self.cosmo.distmod(z).value
        else:
            mu = np.array(mu)
            if len(mu.shape) == 0:
                mu = mu.repeat(N)
            elif mu.shape[0] != N:
                raise ValueError(
                    'For mu, either pass a single scalar value or an array of values for each of the N simulated objects')
        param_dict = {
            'del_M': del_M,
            'AV': AV,
            'theta': theta,
            'eps': eps,
            'z': z,
            'mu': mu,
            'ebv_mw': ebv_mw,
            'RV': RV
        }

        if t.shape[0] == np.array(bands).shape[0]:
            band_indices = np.array([self.band_dict[band] for band in bands])
            band_indices = band_indices[:, None].repeat(N, axis=1).astype(int)
        else:
            t = jnp.array(t)
            num_per_band = t.shape[0]
            num_bands = len(bands)
            band_indices = np.zeros(num_bands * num_per_band)
            t = t[:, None].repeat(num_bands, axis=1).flatten(order='F')
            for i, band in enumerate(bands):
                if band not in self.band_dict.keys():
                    raise ValueError(f'{band} is present in filters yaml file')
                band_indices[i * num_per_band: (i + 1) * num_per_band] = self.band_dict[band]
            band_indices = band_indices[:, None].repeat(N, axis=1).astype(int)
        mask = np.ones_like(band_indices)
        band_weights = self._calculate_band_weights(z, ebv_mw)

        t = jnp.repeat(t[..., None], N, axis=1)
        hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        keep_shape = t.shape
        t = t.flatten(order='F')
        map = jax.vmap(self.spline_coeffs_irr_step, in_axes=(0, None, None))
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,
                                                                                                                     2,
                                                                                                                     0)
        t = t.reshape(keep_shape, order='F')
        if mag:
            data = self.get_mag_batch(self.M0, theta, AV, self.W0, self.W1, eps, mu + del_M, RV, band_indices, mask, J_t,
                                      hsiao_interp, band_weights)
        else:
            data = self.get_flux_batch(self.M0, theta, AV, self.W0, self.W1, eps, mu + del_M, RV, band_indices, mask, J_t,
                                       hsiao_interp, band_weights)
        # Apply error if specified
        yerr = jnp.array(yerr)
        if err_type == 'mag' and not mag:
            yerr = yerr * (np.log(10) / 2.5) * data
        if len(yerr.shape) == 0:  # Single error for all data points
            yerr = np.ones_like(data) * yerr
        elif len(yerr.shape) == 1:
            assert data.shape[0] == yerr.shape[0], f'If passing a 1d array, shape of yerr must match number of ' \
                                                   f'simulated data points per objects, {data.shape[0]}'
            yerr = np.repeat(yerr[..., None], N, axis=1)
        else:
            assert data.shape == yerr.shape, f'If passing a 2d array, shape of yerr must match generated data shape' \
                                             f' of {data.shape}'
        data = np.random.normal(data, yerr)

        if write_to_files and mag:
            if output_dir is None:
                raise ValueError('If writing to SNANA files, please provide an output directory')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            sn_names, sn_files = [], []
            for i in range(N):
                sn_name = f'{i}'
                sn_t, sn_mag, sn_mag_err, sn_z, sn_ebv_mw = t[:, i], data[:, i], yerr[:, i], z[i], ebv_mw[i]
                sn_t = sn_t * (1 + sn_z)
                sn_tmax = 0
                sn_flt = [self.inv_band_dict[f] for f in band_indices[:, i]]
                sn_file = write_snana_lcfile(output_dir, sn_name, sn_t, sn_flt, sn_mag, sn_mag_err, sn_tmax, sn_z, sn_z,
                                             zerr, sn_ebv_mw)
                sn_names.append(sn_name)
                sn_files.append(sn_file)
        elif write_to_files:
            raise ValueError('If writing to SNANA files, please generate mags')
        return data, yerr, param_dict

    def sample_del_M(self, N):
        """
        Samples grey offset del_M from model prior

        Parameters
        ----------
        N: int
            Number of objects to sample for

        Returns
        -------
        del_M: array-like
            Sampled del_M values

        """
        del_M = np.random.normal(0, self.sigma0, N)
        return del_M

    def sample_AV(self, N):
        """
        Samples AV from model prior

        Parameters
        ----------
        N: int
            Number of objects to sample for

        Returns
        -------
        AV: array-like
            Sampled AV values

        """
        AV = np.random.exponential(self.tauA, N)
        return AV

    def sample_theta(self, N):
        """
        Samples theta from model prior

        Parameters
        ----------
        N: int
            Number of objects to sample for

        Returns
        -------
        theta: array-like
            Sampled theta values

        """
        theta = np.random.normal(0, 1, N)
        return theta

    def sample_epsilon(self, N):
        """
        Samples epsilon from model prior

        Parameters
        ----------
        N: int
            Number of objects to sample for

        Returns
        -------
        eps_full: array-like
            Sampled epsilon values
        """
        N_knots_sig = (self.l_knots.shape[0] - 2) * self.tau_knots.shape[0]
        eps_mu = jnp.zeros(N_knots_sig)
        eps_tform = np.random.multivariate_normal(eps_mu, np.eye(N_knots_sig), N)
        eps_tform = eps_tform.T
        eps = np.matmul(self.L_Sigma, eps_tform)
        eps = np.reshape(eps, (N, self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
        eps_full = np.zeros((N, self.l_knots.shape[0], self.tau_knots.shape[0]))
        eps_full[:, 1:-1, :] = eps
        return eps_full

    def get_flux_from_chains(self, t, bands, chain_path, zs, ebv_mws, mag=True, num_samples=None):
        """
        Returns model photometry for posterior samples from BayeSN fits, which can be used to make light curve fit
        plots.

        Parameters
        ----------
        t: array-like
            Array of phases to evaluate model photometry at
        bands: array-like
            List of bandpasses to evaluate model photometry in. Photometry will be
        chain_path: str
            Path to file containing BayeSN fitting posterior samples you wish to obtain photometry for
        zs: array-like
            Array of heliocentric redshifts corresponding to the SNe you are obtaining model fit light curves for.
        ebv_mws: array-like
            Array containing Milky Way extincion values corresponding to the SNe you are obtaining model fit light
            curves for.
        mag: Bool, optional
            Boolean to specify whether you want magnitude or flux data. If True, magnitudes will be returned. If False,
            flux densities (f_lambda) will be returned. Default to True i.e. mag data.
        num_samples: int, optional
            An optional keyword argument to specify the number of posterior samples you wish to obtain photometry for.
            Might be useful in testing if you are looking at lots of SNe, as otherwise this function will take a while
            to generate e.g. photometry for 1000 posterior samples across 1000 SNe. Default to None, meaning that
            photometry will be calculated for all posterior samples in chains provided.

        Returns
        -------

        flux_grid: jax.numpy.array
            Array of shape (number of SNe, number of posterior samples, number of bands, number of phases to evaluate),
            containing photometry across all SNe, all posterior samples, all bands and at all phases requested.

        """
        with open(chain_path, 'rb') as file:
            chains = pickle.load(file)

        N_sne = chains['theta'].shape[2]
        if num_samples is None:
            num_samples = chains['theta'].shape[0] * chains['theta'].shape[1]

        flux_grid = jnp.zeros((N_sne, num_samples, len(bands), len(t)))

        print('Getting best fit light curves from chains...')
        for i in tqdm(np.arange(N_sne)):
            theta = chains['theta'][..., i].flatten(order='F')
            AV = chains['AV'][..., i].flatten(order='F')
            if 'RV' in chains.keys():
                RV = chains['RV'][..., i].flatten(order='F')
            else:
                RV = None
            mu = chains['mu'][..., i].flatten(order='F')
            eps = chains['eps'][..., i]
            eps = eps.reshape((eps.shape[0] * eps.shape[1], eps.shape[2]), order='F')
            eps = eps.reshape((eps.shape[0], self.l_knots.shape[0] - 2, self.tau_knots.shape[0]), order='F')
            eps_full = jnp.zeros((eps.shape[0], self.l_knots.shape[0], self.tau_knots.shape[0]))
            eps = eps_full.at[:, 1:-1, :].set(eps)
            del_M = chains['delM'][..., i].flatten(order='F')

            theta, AV, mu, eps, del_M = theta[:num_samples], AV[:num_samples], mu[:num_samples], \
                                        eps[:num_samples, ...], del_M[:num_samples, ...]

            lc, lc_err, params = self.simulate_light_curve(t, num_samples, bands, theta=theta, AV=AV, mu=mu,
                                                           del_M=del_M, eps=eps, RV=RV, z=zs[i], write_to_files=False,
                                                           ebv_mw=ebv_mws[i], yerr=0, mag=mag)
            lc = lc.T
            lc = lc.reshape(num_samples, len(bands), len(t))
            flux_grid.at[i, ...].set(lc)

        return flux_grid
