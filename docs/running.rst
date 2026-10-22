.. _running_bayesn:

Running large BayeSN Jobs
==========================================

BayeSN can be interacted with in two ways. The first is designed to be run from the command line, with a yaml file
that configures what the user wants to do; this approach is designed for application to large data samples utilising
GPUs. However, if you want to fit/analyse a small sample within a Jupyter notebook/custom Python script you can also do
that as described in :ref:`fitting`.

BayeSN jobs are run just by running the script ``run_bayesn`` (after installation, this script can be called from any
directory), with the specific job defined by an input yaml file which allow you to specify e.g. whether you want to run
model training or fitting, which data you want to use and where your custom filters, if required, are defined. To run a
job, you can just run the following command, making sure to specify the path to your ``input.yaml`` file.

``run_bayesn PATH\TO\input.yaml``

The ``input.yaml`` therefore underpins this code. Examples and explanations of all the keys are given below.

Specifying input to BayeSN
---------------------------------

The keys which can be specified are described below. Depending on your use case, different keys will be required:

- ``load_model``: The existing BayeSN model to use if you are fitting or only doing dust inference. You can either specify one of the built-in models (one of 'G26_model', 'G25_model', 'W22_model', 'T21_model' or 'M20_model') or use a path to a model you have trained yourself (defined by a bayesn.yaml file).
- ``load_ext_rel``: The existing dust extinction relation to use. You can either specify one of the built-in relations (one of 'CCM89', 'C94', 'O94', 'F99', 'F99_AVGLMC', 'F99_LMC2', 'F99_SNANA', 'C00', 'F04', 'VCG04', 'GCC09', 'M14', 'F19', 'D22', or 'G23') or use a path to an extinction relation you have trained yourself (defined by a bayesn.yaml file).
- ``train_new_model``: Boolean specifying whether ``W0``, ``W1``, and ``L_Sigma`` should be sampled or taken from the loaded model.
- ``infer_dust_properties``: Boolean specifying whether ``sigma0``, ``tauA`` should be sampled or taken from the loaded model. If ``RV='normal'``, also affects ``mu_R`` and ``sigma_R``. If ``vary_redshift=True``, also affects ``tau_z_grad`` and ``mu_z_grad``.
- ``shared_RV``: Boolean specifying whether a single RV should be shared across all SNe or whether RVs should be sampled for each SN. Defaults to True.
- ``RV``: Indicates how RV values should be sampled (if at all). Defaults to 'default'. Options are:

  - 'default': All SNe will share a single RV, even if ``shared_RV=False``. This number comes from either ``SEDmodel.RV`` or ``SEDmodel.mu_R`` depending on how the model was trained.
  - Number: All SNe will use the provided Number as RV, even if ``shared_RV=False``.
  - 'normal': RV(s) will be sampled from a normal distribution centred at ``mu_R`` with scales ``sigma_R``, which may provided as inputs or sampled.
  - 'uniform': RV(s) will be sampled from a uniform distribution between ``uniform_RV_min`` and ``uniform_RV_max``, which will take default values if not provided.

- ``num_chains``: The number of MCMC chains to run. Using HMC, it is recommended to use at least 4 chains to assess model convergence.
- ``num_warmup``: The number of warmup steps for HMC. Typically 500 is sufficient for training and 250 for fitting for convergence, but you may need to increase this.
- ``num_samples``: The number of posterior samples to take. Again, typically in development 500 was used when training and 250 when fitting, but you may want to experiment with this to achieve sufficient convergence.
- ``fit_method``: The inference method to use for light curve fitting, only used if fitting. Options are 'mcmc', for doing HMC using the NUTS algorithm, or 'vi' for using variational inference as presented in Uzsoy+2024, which makes an approximation to the shape of the posterior but is faster than MCMC and is ideal for fitting large samples. Defaults to 'mcmc'.
- ``chain_method``: The method to use for running multiple chains in numpyro. If 'sequential', chains will be run one-after-the-other until all are complete. If 'parallel', the chains will be run in parallel over multiple devices - with 4 chains and a node with 4 GPUs, the chains will be run simultaneously in parallel. If 'vectorized', chains will be run in parallel on a single device which may or may not be quicker than running them sequentially depending on the device you are using, and may result in memory issues unless you are using a large GPU.
- ``filters``: Path to a yaml file describing the filters you want to use, if you require custom filters. For more details, please see :ref:`filters`.
- ``map``: A set of ``key: value`` pairs specifying a mapping between filter names. You will want to use this if the filter names in the data files you are using do not match the names in your `filters.yaml` file (see below). For example, if your data files use the filter names _griz_ and you want ensure that you use DES filters you can set ``map: {g: g_DES, r: r_DES, i: i_DES, z: z_DES}``, and just make sure that the filter names _g_DES_ etc. are in your ``filters.yaml`` file. If the filter names in your data files already match those in ``filters.yaml``, this is not required.
- ``data_table``: The path to a table providing paths to data files and metadata for each SN - see below for the required format. This should be used **only** if working with real data, as if you are using an SNANA simulation we can expect all files to be located in the same folder and be confident about the information contained in the headers - in this case, use the ``version_photometry`` key instead.
- ``data_root``: A root directory which will be pre-pended to all file paths in data_table, as well as the location of data_table itself. Not required if using ``version_photometry``, only for ``data_table``.
- ``version_photometry``: The path to an SNANA input directory. Use this if you are running on NERSC Perlmutter and want to use an SNANA input directory. The code will look for a directory with this name in typical public directories within SNDATA_ROOT, or optionally private directories if you specify them using the argument ``private_data_path`` below. Otherwise, use a combination of ``data_table`` and ``data_root`` instead.
- ``private_data_path``: If you are using NERSC Perlmutter and wish to use a directory located in a private location, add a path to the location where your directory sits and it will be added to the default list of locations ``version_photometry`` searches for.
- ``data_type``: Either 'flux' or 'mag', indicating whether the input photometric data are in flux units or magnitudes. Defaults to 'flux'.
- ``save_fit_errors``: Boolean to specify if you want to save model fit errors in the produced LCPLOT table, only relevant when fitting. This will add a bit of computation time as it will involve calculating model photometry for each step along the chains for all SNe fitted. If False, only the light curve corresponding to the posterior mean of each parameter will be saved in the LCPLOT file; if True, the mean and standard deviation of model photometry across all posterior samples will be saved. Defaults to False.
- ``outputdir``: The directory in which to save the output. If not specified, output files will be saved in current directory. This directory will be created if it does not already exist.
- ``outfile_prefix``: A string prefix to prepend to all output files (e.g. ``output.LCPLOT``, ``output.FITRES.TEXT``). Defaults to 'output'.

Finer Control with Additional Inputs
------------------------------------

For semi-completeness, seldom used inputs are described here. Most users can safely ignore this section.
There are a few keys present in the code that are not documented here. These are specific for usage within SNANA or for fitting with variational inference.

- ``initialisation``: The strategy used to initialise the HMC chains. Must be one of:

  - 'median': Initialise chains to the prior median with some additional scatter between chains applied by numpyro.
  - 'sample': Initialise chains to random samples from the priors.
  - Alternatively, when training the model you can pass either the name of a built-in model or the path to a custom model (as for ``load_model`` above). This will initialise some global parameters to those of that BayeSN model, specifically W0 and W1. This is useful just to stop these complex, multidimensional parameters starting in a very bad region of parameter space. Other parameters are random samples from the priors. If using different wavelength knots to the specified model, the initial values for W0 and W1 matrices will be determined by cubic spline interpolation of the originals if within the wavelength range covered by T21, and initialised to 0 otherwise.
- ``split_variant``: If not None, then must be either 'split_mag' or 'split_sed'. 'split_mag' will divide SNe by host-galaxy stellar mass and sample independent ``M0`` values for the two populations, replicating a 'mass step'. 'split_sed' will divide SNe the same way, but sample independent ``W0`` values which parameterizes magnitude differences as a function of phase and wavelength. Either non-None value will sample independent dust properties in each population. Defaults to None.
- ``M_split``: The log10(stellar mass [solar mass])) dividing SNe into low-mass and high-mass populations if ``split_variant`` is 'split_mag' or 'split_sed'.
- ``vary_redshift``: If True and ``infer_dust_properties=True``, sample ``tau_z_grad`` and ``mu_z_grad`` and formulate any SN's ``tauA`` (``mu_R``) as ``tauA + tau_z_grad*z`` (``mu_R + mu_z_grad*z``).
- ``tau_z_min``: The lower limit of the uniform distribution for ``tau_z_grad``.
- ``tau_z_max``: The upper limit of the uniform distribution for ``tau_z_grad``.
- ``vary_filter_shifts``: If True, infer filter wavelength calibration shifts (in Angstroms) for each bandpass (as introduced in Grayling+2026). Defaults to False.
- ``vary_offsets``: If True, infer zero-point magnitude calibration shifts for each bandpass (as introduced in Grayling+2026). Defaults to False.
- ``photoz``: If True, sample each SN's redshift using photometric redshift priors (Gaussian uncertainties or quantiles). Defaults to False.
- ``error_floor``: A lower limit for magnitude uncertainties (in mag units regardless of ``data_type``). The equivalent flux uncertainty is computed and applied. Defaults to 0.0.
- ``batch_size``: An integer limit on the number of SNe processed simultaneously to manage GPU memory usage during light curve fitting. Defaults to None (all in a single batch).
- ``save_summary``: If True, saves an ArviZ summary table to ``{outfile_prefix}.SUMMARY.TEXT`` during postprocessing and records ``MEANRHAT`` and ``MAXRHAT`` in the fitres table. Defaults to False.
- ``jobsplit``: A 2-tuple ``[jobid, njobtot]`` specifying a job number and total jobs to partition and selectively process a fraction of a dataset across batch jobs. Defaults to None.
- ``zlim``: Upper redshift limit; SNe with heliocentric redshift >= ``zlim`` will not be loaded. Defaults to 99.0.
- ``peakmjd_key``: Key in the SN data files identifying the estimated peak MJD. Defaults to 'PEAKMJD'.
- ``l_knots``: The wavelength knot locations for the 2D cubic spline surfaces that define the BayeSN model. These only need to be specified when ``train_new_model=True``, as they are otherwise specified in the model definition.
- ``tau_knots``: The rest-frame phase knot locations for the 2D cubic spline surfaces that define the BayeSN model. These only need to be specified when ``train_new_model=True``, as they are otherwise specified in the model definition.
- ``fix_tmax``: If False, each SN's time of maximum will be sampled from a uniform distribution between 10 days before and 10 days after the ``peak_mjd`` in the data. If True, treat the ``peak_mjd`` values as the times of maxima.
- ``fix_eps``: If False, each SN's residual term spline knots (epsilon) will be sampled from a multivariate normal distribution centred at 0 with covariance matrix ``L_Sigma`` (which may itself be sampled). If True, treat all epsilon terms as 0 at all knots.
- ``fix_AV``: If None, samples AV for each SN. If a Number, uses that value for all SNe.
- ``fix_theta``: If None, samples theta for each SN. If a Number, uses that value for all SNe.
- ``mu_R``: The centre of the normal distribution for RV(s) when ``RV='normal'`` and ``infer_dust_properties=False``. If 'default', uses either ``SEDmodel.RV`` or ``SEDmodel.mu_R``, depending on how the model was trained.
- ``sigma_R``: The standard deviation of the normal distribution for RV(s) when ``RV='normal'`` and ``infer_dust_properties=False``. If 'default', uses ``SEDmodel.sigma_R`` if the model was trained with ``shared_RV=False`` or 0.5.
- ``uniform_RV_min``: The lower limit of the uniform distribution for RV(s) when ``RV='uniform'``. Defaults to 1.
- ``uniform_RV_max``: The upper limit of the uniform distribution for RV(s) when ``RV='uniform'``. Defaults to 6.
- ``mu_R_min``: The lower limit of the uniform distribution for ``mu_R`` when ``RV='normal'`` and ``infer_dust_properties=True``. Defaults to 1.2.
- ``mu_R_max``: The upper limit of the uniform distribution for ``mu_R`` when ``RV='normal'`` and ``infer_dust_properties=True``. Defaults to 6.
- ``sigma_sigma_R``: The scale factor of the Half-Normal distribution for ``sigma_R`` when ``RV='normal'`` and ``infer_dust_properties=True``. Defaults to 2.
- ``muhat_err``: Part of the uncertainty in the redshift-based luminosity distance modulus used as a prior for the inferred distance modulus. If None, ``muhat_err`` is calculated from the redshift uncertainty. If a number, that value is assumed. In both cases, the total error is the quadrature sum of ``muhat_err`` and a peculiar velocity uncertainty. Defaults to None.
- ``fix_dist_limit``: Replaces the quadrature sum of ``muhat_err`` and a peculiar velocity uncertainty with ``fix_dist_Ds_err`` for SNe with redshifts above ``fix_dist_limit``. Useful when training a model using data covering a large redshift range, where cosmological effects can mimic population drift in ``M0`` or ``W0``. Defaults to 0.08.
- ``fix_dist_Ds_err``: The large uncertainty assigned to SNe at redshifts beyond ``fix_dist_limit``. Defaults to 5.
- ``keep_list``: A list of SN names which are present in the data files that you are using which you wish to keep in the analysis. If None, include everything.
- ``drop_bands``: A list of bands which are present in the data files that you are using which you do not wish to include in the analysis, optional.
- ``lc_cuts``: A dictionary with keys matching SNe metadata names and values of 2-tuples, defining the lower and upper limits beyond which SNe will be cut. As an example, ``lc_cuts={'mwebv': (0, 0.3)}`` will identify and remove all SNe with MW E(B-V) values greater than 0.3 before sampling.
- ``mode``: A legacy term included for backwards compatibility. Used to influence BayeSN behavior by specifying other inputs. Supported options are:

  - 'training_globalRv': Train a model with a single RV value across the population. Sets ``train_new_model=True``, ``infer_dust_properties=True``, ``shared_RV=True``, ``RV='uniform'``, ``data_type='mag'``, ``uniform_RV_min=1``, ``uniform_RV_max=5``, and ``fix_tmax=True``. ('training_singleRV' is also accepted for backwards compatibility).
  - 'training_popRV': Train a model with a truncated Gaussian RV population distribution. Sets ``train_new_model=True``, ``infer_dust_properties=True``, ``shared_RV=False``, ``RV='normal'``, ``data_type='mag'``, ``mu_R_min=1``, ``mu_R_max=5``, ``sigma_sigma_R=2``, and ``fix_tmax=True``.
  - 'fitting': Fit using fixed global parameters previously inferred during model training. Sets ``train_new_model=False``, ``infer_dust_properties=False``, ``RV='default'``, ``data_type='flux'``, and ``muhat_err=5``.
  - 'dust': Infer dust properties while keeping SN population parameters fixed based on previously inferred model, as explored in Grayling+2024. Sets ``train_new_model=False``, ``infer_dust_properties=True``, ``shared_RV=False``, ``RV='normal'``, ``data_type='flux'``, ``mu_R_min=1.2``, ``mu_R_max=6``, ``sigma_sigma_R=2``, and ``fix_tmax=True``.
  - 'dust_split_mag': Infer dust properties separately in different mass bins, allowing for an intrinsic magnitude offset between the bins, as explored in Grayling+2024. Sets ``train_new_model=False``, ``infer_dust_properties=True``, ``split_variant='split_mag'``, ``M_split=10``, ``shared_RV=False``, ``RV='normal'``, ``data_type='flux'``, ``mu_R_min=1.2``, ``mu_R_max=6``, ``sigma_sigma_R=2``, and ``fix_tmax=True``.
  - 'dust_split_sed': Infer dust properties separately in different mass bins, allowing for a difference in baseline intrinsic SED, as explored in Grayling+2024. Sets ``train_new_model=False``, ``infer_dust_properties=True``, ``split_variant='split_sed'``, ``M_split=10``, ``shared_RV=False``, ``RV='normal'``, ``data_type='flux'``, ``mu_R_min=1.2``, ``mu_R_max=6``, ``sigma_sigma_R=2``, and ``fix_tmax=True``.


You can see an example of input.yaml files for training and fitting below.

Example yaml files
------------------------------

Training example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the input.yaml that could be used to train the BayeSN model presented in Thorp+2021.

.. code-block:: yaml

    load_ext_rel: F99
    train_new_model: True
    infer_dust_properties: True
    shared_RV: True
    RV: uniform
    uniform_RV_min: 1
    uniform_RV_max: 5
    initialisation: T21
    num_chains: 4
    num_warmup: 500
    num_samples: 500
    map: {g: g_PS1, r: r_PS1, i: i_PS1, z: z_PS1}
    l_knots: [3500.0, 4900.0, 6200.0, 7700.0, 8700.0, 9500.0]
    tau_knots: [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    data_root: /PATH/TO/DATA/ROOT
    data_table: T21_training_set.txt
    outputdir: /PATH/TO/OUTPUT/DIR
    apply_dovekie_mag_shifts: False
    data_type: mag
    fix_dist_limit: 0.1  # to avoid affecting a SN at z>0.0805
    fix_tmax: True


Fitting example
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the input.yaml that could be used to fit some SNANA simulations using a custom BayeSN model
defined in a bayesn.yaml file, with custom filters defined in a filters.yaml file and using variational inference
for the fitting.

.. code-block:: yaml

    load_model: /PATH/TO/CUSTOM/bayesn.yaml
    num_chains: 4
    num_warmup: 250
    num_samples: 250
    filters: /PATH/TO/custom_filters.yaml
    fit_method: vi
    initialisation: median
    version_photometry: NAME_OF_SNANA_SIMULATION
    outputdir: /PATH/TO/OUTPUT/DIR

Specifying data to use
-------------------------------

As discussed above, if you are using an SNANA input directory with SNDATA_ROOT on NERSC Perlmutter as input you need only pass
the name of the directory to the ``version_photometry`` key in the input file. However, if you want to have more control
over the input data e.g. you may want to use data spanning multiple surveys which means you won't necessarily be able to
point to a single directory, you should use the keys ``data_table`` and ``data_root`` in the input. ``data_table``
should contain file paths to the data for each SN as well as associated metadata for the SN, with the following structure:

.. code-block:: text

    SNID SEARCH_PEAKMJD	REDSHIFT_CMB REDSHIFT_CMB_ERR files
    SN1	57400	0.02	0.0001	survey1/SN1.txt
    SN2	57500	0.03	0.0001  survey1/SN2.txt
    SN3	57600	0.04	0.0001	survey1/SN3.txt,survey2/SN3.txt
    SN4	57700	0.05	0.0001	survey3/SN3_optical.txt,survey3/SN3_NIR.txt
    SN5	57800	0.06	0.0001	survey4/SN4.txt

The table allows for multiple files per object if required, the file names just need to be separated by commas in the
files columns. This approach allows you to read in data from multiple surveys, including cases where the same object has
observations from multiple surveys which are contained in different data files. This is also relevant for cases where
one object may have both optical and NIR data which are contained in different files.

The table should include Hubble diagram (corrected to CMB-frame and for peculiar velocity) redshifts and associated
uncertainties, as well as the time of B-band maximum SEARCH_PEAKMJD. When training a model, these redshifts will be used
to fix cosmological distances, and the time of maximum is kept fixed (it is possible to leave this as a free parameter
when training but this is computationally expensive; the idea is that a model would only be trained on high quality data
with good constraint on the time of maximum).

If you are only interested in fitting, these CMB-frame redshifts will be ignored and instead it is assumed that accurate
heliocentric redshifts are contained within the data files themselves. In this case SEARCH_PEAKMJD need only be a rough
estimate as the model will also infer the time of maximum, using a uniform prior covering 10 rest-frame days either side
of the specified SEARCH_PEAKMJD.

The key ``data_root`` simply specifies the location that the file paths in ``data_table`` are defined with respect to.
For example, with ``data_root: /data/photometry/``, the full file path for the first file in the table above will be
``/data/photometry/survey1/SN1.txt`` and similar for the rest. The same is true for ``data_table`` itself; in the case
described here, if ``data_table`` is ``table.txt`` then the full file path for the data table is
``/data/photometry/table.txt``.

Vectorisation
---------------

BayeSN is designed for vectorised calculations across a sample of SNe, with all calculations handling all SNe in a
single operation. For model training, it is natural that parameters are simultaneously inferred for all SNe in the
sample in addition to the population-level parameters. However, we apply the same approach in light curve fitting as
well. For a sample of 100 SNe, rather than running 100 fitting jobs in serial as you might do with e.g SALT or SNooPy,
in BayeSN you run a single fitting job that fits all 100 SNe in parallel.
