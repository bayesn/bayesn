"""
BayeSN SED Model. Defines a class which allows you to fit or simulate from the
BayeSN Optical+NIR SED model.
"""

import os
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
import matplotlib as mpl
from matplotlib import rc
import arviz
import extinction
import timeit
from astropy.io import fits
import ruamel.yaml as yaml
import time
from tqdm import tqdm

# Make plots look pretty
rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 22})

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
                        Global RV assumed. Trained on low-z AVelino+19 (ApJ, 887, 106) compilation of CfA, CSP and
                        others.
        ``T21_model``: Thorp+21 No-Split BayeSN model (arXiv:2102:05678).
                        Covers rest wavelength range of 3500-9500A (griz). No treatment of host mass effects. Global RV
                        assumed. Trained on Foundation DR1 (Foley+18, Jones+19).
    fiducial_cosmology :  dict, optional
        Dictionary containg kwargs ``{H0, Om0}`` for initialising an ``astropy.cosmology.FlatLambdaCDM`` instance.
        Defaults to Riess+16 (ApJ, 826, 56) cosmology:
        ``{H0:73.24, "Om0":0.28}``.
    filter_yaml: str, optional
        Path to yaml file containing details on filters and standards to use. If not specified, will look for a file
        called filters.yaml in directory that BayeSN is called from.

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

    def __init__(self, num_devices=4, load_model='T21_model', filter_yaml='filters.yaml',
                 fiducial_cosmology={"H0": 73.24, "Om0": 0.28}):
        # Settings for jax/numpyro
        numpyro.set_host_device_count(num_devices)
        # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print('Current devices:', jax.devices())

        self.__root_dir__ = os.path.dirname(os.path.abspath(__file__))
        print(f'Currently working in {os.getcwd()}')

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
                params = yaml.load(file, Loader=yaml.Loader)
        elif load_model in built_in_models:
            print(f'Loading built-in model {load_model}')
            with open(os.path.join(self.__root_dir__, 'model_files', load_model, 'BAYESN.YAML'), 'r') as file:
                params = yaml.load(file, Loader=yaml.Loader)
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
        if 'TRUNCRV' in params.keys():
            self.truncate_RV = True
            self.trunc_val = jnp.array(params['TRUNCRV'])
        else:
            self.truncate_RV = False

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
            filter_dict = yaml.load(file, Loader=yaml.Loader)

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

            band_low_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][0], 0]
            band_up_lim = R[np.where(R[:, 1] > 0.01 * R[:, 1].max())[0][-1], 0]

            # Convolve the bands to match the sampling of the spectrum.
            # band_conv_transmission = jnp.interp(band_wave, R[:, 0], R[:, 1])
            band_conv_transmission = scipy.interpolate.interp1d(R[:, 0], R[:, 1], kind='cubic',
                                                                fill_value=0, bounds_error=False)(band_wave)

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
        W0 = jnp.repeat(W0[None, ...], num_batch, axis=0)
        W1 = jnp.repeat(W1[None, ...], num_batch, axis=0)

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

    def get_flux_batch(self, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame fluxes for given parameter values

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
        model_flux = model_flux * 10 ** (-0.4 * (self.M0 + Ds))
        zps = self.zps[band_indices]
        offsets = self.offsets[band_indices]
        zp_flux = 10 ** (zps / 2.5)
        model_flux = (model_flux / zp_flux) * 10 ** (0.4 * (27.5 - offsets))  # Convert to FLUXCAL
        model_flux *= mask
        return model_flux

    def get_mag_batch(self, theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights):
        """
        Calculates observer-frame magnitudes for given parameter values

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
        model_flux = self.get_flux_batch(theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, J_t, hsiao_interp, weights)
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

    def fit_model_fixRV(self, obs, weights):
        """
        Numpyro model used for fitting latent SN properties with RV fixed. Will fit for time of maximum as well as
        theta, epsilon, AV and distance modulus.

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
            flux = self.get_flux_batch(theta, AV, self.W0, self.W1, eps, Ds, self.RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)

    def fit_model_popRV_notrunc(self, obs, weights):
        """
        Numpyro model used for fitting latent SN properties with a Gaussian prior on RV. Will fit for time of maximum as
        well as theta, epsilon, AV, RV and distance modulus.

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
            RV = numpyro.sample('RV', dist.Normal(self.mu_R, self.sigma_R))
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
            flux = self.get_flux_batch(theta, AV, self.W0, self.W1, eps, Ds, RV, band_indices, mask,
                                       J_t, hsiao_interp, weights)
            # print(obs.shape)
            # plt.close('all')
            # plt.scatter(obs[0, :, 0], obs[1, :, 0])
            # plt.show()
            # raise ValueError('Nope')
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T),
                               obs=obs[1, :, sn_index].T)  # _{sn_index}

    def fit_model_popRV_trunc(self, obs, weights):
        """
        Numpyro model used for fitting latent SN properties with a Gaussian prior on RV. Will fit for time of maximum as
        well as theta, epsilon, AV, RV and distance modulus.

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
            flux = self.get_flux_batch(theta, AV, self.W0, self.W1, eps, Ds, RV, band_indices, mask,
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
            flux = self.get_mag_batch(theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, self.J_t, self.hsiao_interp,
                                      weights)

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def train_model_popRV_notrunc(self, obs, weights):
        """
        Numpyro model used for training to learn global parameters with a Gaussian RV distribution

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset

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

        # tauA = numpyro.sample('tauA', dist.HalfCauchy())
        tauA_tform = numpyro.sample('tauA_tform', dist.Uniform(0, jnp.pi / 2.))
        tauA = numpyro.deterministic('tauA', jnp.tan(tauA_tform))

        with numpyro.plate('SNe', sample_size) as sn_index:
            theta = numpyro.sample(f'theta', dist.Normal(0, 1.0))  # _{sn_index}
            AV = numpyro.sample(f'AV', dist.Exponential(1 / tauA))
            RV_tform = numpyro.sample('RV_tform', dist.Normal(0, 1))
            RV = numpyro.deterministic('RV', mu_R + RV_tform * sigma_R)

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
            flux = self.get_mag_batch(theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, self.J_t, self.hsiao_interp,
                                      weights)
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(f'obs', dist.Normal(flux, obs[2, :, sn_index].T), obs=obs[1, :, sn_index].T)

    def train_model_popRV_trunc(self, obs, weights):
        """
        Numpyro model used for training to learn global parameters with a truncated Gaussian RV distribution

        Parameters
        ----------
        obs: array-like
            Data to fit, from output of process_dataset

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
            flux = self.get_mag_batch(theta, AV, W0, W1, eps, Ds, RV, band_indices, mask, self.J_t, self.hsiao_interp,
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
                params = yaml.load(file, Loader=reference_model.Loader)
        elif reference_model in built_in_models:
            print(f'Loading built-in model {reference_model} to initialise chains')
            with open(os.path.join(self.__root_dir__, 'model_files', reference_model, 'BAYESN.YAML'), 'r') as file:
                params = yaml.load(file, Loader=yaml.Loader)
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
            if self.truncate_RV:
                param_init['RV_tform'] = jnp.array(np.random.uniform(0, 1, self.data.shape[-1]))
            else:
                param_init['RV_tform'] = jnp.array(np.random.normal(0, 1, self.data.shape[-1]))
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
        args: dictionary
            dictionary of arguments to define model based on input yaml file
        cmd_args: dictionary
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
        args['outputdir'] = args.get('outputdir', os.path.join(os.getcwd(), args['name']))
        args['yamloutputfile'] = args.get('yamloutputfile', 'output.yaml')

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
            # Check if truncating
            self.truncate_RV = False
            self.trunc_val = args.get('trunc_RV')
            if self.trunc_val is not None:
                self.truncate_RV = True
        self.process_dataset(args)
        t = self.data[0, ...]
        self.hsiao_interp = jnp.array([19 + jnp.floor(t), 19 + jnp.ceil(t), jnp.remainder(t, 1)])
        return args

    def run(self, args, cmd_args):
        """
        Main method to run BayeSN. Can be used for either model training or fitting depending on input yaml file.

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
            if self.truncate_RV:
                nuts_kernel = NUTS(self.train_model_popRV_trunc, adapt_step_size=True, target_accept_prob=0.8,
                                   init_strategy=init_strategy,
                                   dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                                   step_size=0.1)
            else:
                nuts_kernel = NUTS(self.train_model_popRV_notrunc, adapt_step_size=True, target_accept_prob=0.8,
                                   init_strategy=init_strategy,
                                   dense_mass=False, find_heuristic_step_size=False, regularize_mass_matrix=False,
                                   step_size=0.1)
        elif args['mode'].lower() == 'fitting':
            if self.model_type == 'pop_RV':
                if self.truncate_RV:
                    nuts_kernel = NUTS(self.fit_model_popRV_trunc, adapt_step_size=True, init_strategy=init_strategy,
                                       max_tree_depth=10)
                else:
                    nuts_kernel = NUTS(self.fit_model_popRV_notrunc, adapt_step_size=True, init_strategy=init_strategy,
                                       max_tree_depth=10)
            elif self.model_type == 'fixed_RV':
                nuts_kernel = NUTS(self.fit_model_fixRV, adapt_step_size=True, init_strategy=init_strategy,
                                   max_tree_depth=10)
        else:
            raise ValueError("Invalid mode, must select one of 'training_globalRV', 'training_popRV', or 'fitting'")

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
        output = args['name']
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
                if self.truncate_RV:
                    yaml_data['TRUNCRV'] = self.trunc_val

            with open(os.path.join(args['outputdir'], 'bayesn.yaml'), 'w') as file:
                yaml.dump(yaml_data, file)

        z = self.data[-5, 0, :]
        muhat = self.data[-3, 0, :]

        if args['mode'] == 'fitting':
            muhat_err = 5
            Ds_err = jnp.sqrt(muhat_err * muhat_err + self.sigma0 * self.sigma0)
            samples['mu'] = np.random.normal(
                (samples['Ds'] * np.power(muhat_err, 2) + muhat * np.power(self.sigma0, 2)) /
                np.power(Ds_err, 2),
                np.sqrt((np.power(self.sigma0, 2) * np.power(muhat_err, 2)) / np.power(Ds_err, 2)))
            samples['delM'] = samples['Ds'] - samples['mu']

        # Save convergence data for each parameter to csv file
        summary = arviz.summary(samples)
        summary.to_csv(os.path.join(args['outputdir'], 'fit_summary.csv'))

        with open(os.path.join(args['outputdir'], 'chains.pkl'), 'wb') as file:
            pickle.dump(samples, file)

        data = np.array([self.sn_list, z, muhat]).T

        df = pd.DataFrame(data, columns=['sn', 'z', 'muhat'])
        df.to_csv(os.path.join(args['outputdir'], 'sn_props.txt'), index=False)

        with open(os.path.join(args['outputdir'], 'input.yaml'), 'w') as file:
            yaml.dump(args, file)

        # Output yaml
        out_dict = {
            'ABORT_IF_ZERO': 1,
            'NLC': self.data.shape[-1],
            'NOBS': int(self.data[-1, ...].sum())
        }
        with open(os.path.join(args['outputdir'], args['yamloutputfile']), 'w') as file:
            yaml.dump(out_dict, file, default_flow_style=False)
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
        if 'data_dir' not in args.keys() and 'data_table' not in args.keys():
            raise ValueError('Please pass either data_dir (for a directory containing all SNANA files such as a '
                             'simulation output) or a combination of data_table and data_root')
        if 'data_table' in args.keys() and 'data_root' not in args.keys():
            raise ValueError('If using data_table, please also pass data_root (which defines the location that the '
                             'paths in data_table are defined with respect to)')
        if 'data_dir' in args.keys():  # If SNANA simulation
            data_dir = args['data_dir']
            sample_name = os.path.split(data_dir)[-1]
            list_file = os.path.join(data_dir, f'{os.path.split(data_dir)[-1]}.LIST')
            sn_list = np.loadtxt(list_file, dtype='str')
            file_format = sn_list[0].split('.')[1]
            map_dict = args['map']
            n_obs = []
            all_lcs = []
            t_ranges = []
            sne = []
            used_bands, used_band_dict = ['NULL_BAND'], {0: 0}
            print('Reading light curves...')
            if file_format.lower() == 'fits':  # If FITS format
                for sn_file in tqdm(sn_list):
                    head_file = os.path.join(data_dir, f'{sn_file}.gz')
                    phot_file = os.path.join(data_dir, f'{sn_file.replace("HEAD", "PHOT")}.gz')
                    sne = sncosmo.read_snana_fits(head_file, phot_file)
                    for sn_ind in range(len(sne)):
                        sn = sne[sn_ind]
                        meta, data = sn.meta, sn.to_pandas()
                        data['BAND'] = data.BAND.str.decode("utf-8")
                        data['BAND'] = data.BAND.str.strip()
                        peak_mjd = meta['PEAKMJD']
                        sne.append(meta['SNID'])
                        z = meta['SIM_REDSHIFT_CMB']
                        z_err = 5e-3
                        data['t'] = (data.MJD - peak_mjd) / (1 + z)
                        # If filter not in map_dict, assume one-to-one mapping------
                        for f in data.BAND.unique():
                            if f not in map_dict.keys():
                                map_dict[f] = f
                        data['FLT'] = data.BAND.apply(lambda x: map_dict[x])
                        # Remove bands outside of filter coverage-------------------
                        for f in data.FLT.unique():
                            if z > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or z < (
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
                        if 'MAG' not in data.columns or (data['MAG'] == 0).sum() > 0:
                            data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                            data['MAGERR'] = (2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL']
                        data['flux'] = data['FLUXCAL']  # np.power(10, -0.4 * (data['MAG'] - data['zp'])) * self.scale
                        data['flux_err'] = data['FLUXCALERR']  # (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
                        data['redshift'] = z
                        data['redshift_error'] = z_err
                        data['MWEBV'] = meta['MWEBV']
                        data['dist_mod'] = self.cosmo.distmod(z)
                        data['mask'] = 1
                        lc = data[
                            ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'band_indices', 'redshift', 'redshift_error',
                             'dist_mod',
                             'MWEBV',
                             'mask']]
                        lc = lc.dropna(subset=['flux', 'flux_err', 'MAG', 'MAGERR'])
                        lc = lc[(lc['t'] > -10) & (lc['t'] < 40)]
                        t_ranges.append((lc['t'].min(), lc['t'].max()))
                        n_obs.append(lc.shape[0])
                        all_lcs.append(lc)
            else:  # If not FITS, assume text format
                for sn_file in tqdm(sn_list):
                    meta, lcdata = sncosmo.read_snana_ascii(os.path.join(data_dir, sn_file), default_tablename='OBS')
                    data = lcdata['OBS'].to_pandas()
                    peak_mjd = meta['PEAKMJD']
                    sne.append(meta['SNID'])
                    z = meta['SIM_REDSHIFT_CMB']
                    z_err = 5e-3
                    data['t'] = (data.MJD - peak_mjd) / (1 + z)
                    # If filter not in map_dict, assume one-to-one mapping------
                    map_dict = args['map']
                    for f in data.BAND.unique():
                        if f not in map_dict.keys():
                            map_dict[f] = f
                    data['FLT'] = data.BAND.apply(lambda x: map_dict[x])

                    # Remove bands outside of filter coverage-------------------
                    for f in data.FLT.unique():
                        if z > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or z < (
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
                    if 'MAG' not in data.columns or (data['MAG'] == 0).sum() > 0:
                        data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                        data['MAGERR'] = np.abs((2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL'])
                    data['flux'] = data['FLUXCAL']  # np.power(10, -0.4 * (data['MAG'] - data['zp'])) * self.scale
                    data['flux_err'] = data['FLUXCALERR']  # (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
                    data['redshift'] = z
                    data['redshift_error'] = z_err
                    data['MWEBV'] = meta['MWEBV']
                    data['dist_mod'] = self.cosmo.distmod(z)
                    data['mask'] = 1
                    lc = data[
                        ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'band_indices', 'redshift', 'redshift_error',
                         'dist_mod',
                         'MWEBV',
                         'mask']]
                    lc = lc.dropna(subset=['flux', 'flux_err', 'MAG', 'MAGERR'])
                    lc = lc[(lc['t'] > -10) & (lc['t'] < 40)]
                    t_ranges.append((lc['t'].min(), lc['t'].max()))
                    n_obs.append(lc.shape[0])
                    all_lcs.append(lc)
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
            flux_data = all_data[[0, 1, 2, 5, 6, 7, 8, 9, 10], ...]
            mag_data = all_data[[0, 3, 4, 5, 6, 7, 8, 9, 10], ...]
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
            return
        else:
            table_path = os.path.join(args['data_root'], args['data_table'])
            sn_list = pd.read_csv(table_path, comment='#', delim_whitespace=True)
            n_obs = []

            all_lcs = []
            t_ranges = []
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
                    data = data[~data.FLT.isin(args['drop_bands'])]  # Skip certain bands
                    data['t'] = (data.MJD - peak_mjd) / (1 + row.REDSHIFT_CMB)
                    # If filter not in map_dict, assume one-to-one mapping------
                    map_dict = args['map']
                    for f in data.FLT.unique():
                        if f not in map_dict.keys():
                            map_dict[f] = f
                    data['FLT'] = data.FLT.apply(lambda x: map_dict[x])
                    # Remove bands outside of filter coverage-------------------
                    for f in data.FLT.unique():
                        if row.REDSHIFT_CMB > (self.band_lim_dict[f][0] / self.l_knots[0] - 1) or row.REDSHIFT_CMB < (
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
                    if (data['MAG'] == 0).sum() > 0:
                        data['MAG'] = 27.5 - 2.5 * np.log10(data['FLUXCAL'])
                        data['MAGERR'] = (2.5 / np.log(10)) * data['FLUXCALERR'] / data['FLUXCAL']
                    data['flux'] = data['FLUXCAL']  # np.power(10, -0.4 * (data['MAG'] - data['zp'])) * self.scale
                    data['flux_err'] = data['FLUXCALERR']  # (np.log(10) / 2.5) * data['flux'] * data['MAGERR']
                    data['redshift'] = row.REDSHIFT_CMB
                    data['redshift_error'] = row.REDSHIFT_CMB_ERR
                    data['MWEBV'] = meta['MWEBV']
                    data['dist_mod'] = self.cosmo.distmod(row.REDSHIFT_CMB)
                    data['mask'] = 1
                    lc = data[
                        ['t', 'flux', 'flux_err', 'MAG', 'MAGERR', 'band_indices', 'redshift', 'redshift_error',
                         'dist_mod',
                         'MWEBV',
                         'mask']]
                    lc = lc.dropna(subset=['flux', 'flux_err', 'MAG', 'MAGERR'])
                    lc = lc[(lc['t'] > self.tau_knots.min()) & (lc['t'] < self.tau_knots.max())]
                    if sn_lc is None:
                        sn_lc = lc.copy()
                    else:
                        sn_lc = pd.concat([sn_lc, lc])
                t_ranges.append((lc['t'].min(), lc['t'].max()))
                n_obs.append(lc.shape[0])
                all_lcs.append(sn_lc)
            N_sn = sn_list.shape[0]
            N_obs = np.max(n_obs)
            N_col = lc.shape[1]
            all_data = np.zeros((N_sn, N_obs, N_col))
            print('Saving light curves to standard grid...')
            for i in tqdm(range(len(all_lcs))):
                lc = all_lcs[i]
                all_data[i, :lc.shape[0], :] = lc.values
                all_data[i, lc.shape[0]:, 2] = 1 / jnp.sqrt(2 * np.pi)
                all_data[i, lc.shape[0]:, 3] = 10  # Arbitrarily set all masked points to H-band
            all_data = all_data.T
            t = all_data[0, ...]
            keep_shape = t.shape
            t = t.flatten(order='F')
            J_t = self.J_t_map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]),
                                                                     order='F').transpose(1, 2, 0)
            flux_data = all_data[[0, 1, 2, 5, 6, 7, 8, 9, 10], ...]
            mag_data = all_data[[0, 3, 4, 5, 6, 7, 8, 9, 10], ...]
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
        l_r = np.linspace(min(self.l_knots), max(self.l_knots), int((max(self.l_knots) - min(self.l_knots)) / dl) + dl)
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
        J_t = map(t, self.tau_knots, self.KD_t).reshape((*keep_shape, self.tau_knots.shape[0]), order='F').transpose(1,
                                                                                                                     2,
                                                                                                                     0)
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
        if mu == 'z':
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
            data = self.get_mag_batch(theta, AV, self.W0, self.W1, eps, mu + del_M, RV, band_indices, mask, J_t,
                                      hsiao_interp, band_weights)
        else:
            data = self.get_flux_batch(theta, AV, self.W0, self.W1, eps, mu + del_M, RV, band_indices, mask, J_t,
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
