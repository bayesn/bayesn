.. _fitting:

Light curve fits within Python/Jupyter
==========================================

If you have a large data set which you want to use for training, or if you want to fit a large sample of light curves,
you should follow the instructions on :ref:`running_bayesn`. However, often that won't be the case; you may well just have
a small selection of light curves that you'd like to fit and plot. For those cases, BayeSN includes methods
designed for use within your own Jupyter notebooks/Python scrips.

The notebook `example_fits.ipynb` that is included on the Github repo for BayeSN
`here <https://github.com/bayesn/bayesn>`_ provides examples of how to use this functionality. Much of this
notebook is also reproduced here to demonstrate how this works.

Fitting from an SNANA-format text file
------------------------------------------

Let's say you have an SNANA format file containing light curve data and metadata. In this case, you can just use the
method ``fit_from_file`` of the BayeSN ``SEDmodel`` class to fit the light curve, as shown below. You can run this code
as-is, and ``model.example_lc`` will automatically load an example light curve built into BayeSN, SN 2016W from Foundation.
You can use this for testing or swap this out for your own data - just provide the path to the SNANA text file.

.. code-block:: python

    from bayesn import SEDmodel

    model = SEDmodel(load_model='T21_model')

    filt_map = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
    samples, sn_props = model.fit_from_file(model.example_lc, filt_map=filt_map)

This function will return MCMC samples for your object, which you can analyse yourself or use to make plots (as shown
later in this notebook). It will also return a tuple, here called ``sn_props``, containing the redshift and Milky Way E(B-V)
for this object which you can use when plotting.

``filt_map`` is a dictionary which allows you to specify which filters to use e.g. in the SNANA file for this example
object, the filter names are just griz, this map will ensure that Pan-STARRS 1 filters are used when fitting.

``fit_from_file`` is looking for particular keys in an SNANA file. Most of those are fairly standard but depending on your
file you may want to use different keys for peak MJD (this is only a rough guess, BayeSN will fit for time of maximum).
By default, the code will look for the key 'SEARCH_PEAKMJD', but you can change this with the optional kwarg
``peak_mjd_key``.

Fitting more general data formats
----------------------------------------
However, you might not have a nicely formatted SNANA format light curve file. You can also use the more general method
``fit`` and provide phases/fluxes/uncertainties etc. yourself, all stored in arrays, as shown below.

.. code-block:: python

    filt_map = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
    samples, sn_props = model.fit(t, flux, flux_err, filters, z, peak_mjd=peak_mjd, ebv_mw=ebv_mw, filt_map=filt_map, mag=False)

This returns MCMC samples and a tuple of SN properties just as ``model.fit_from_file()`` does. For this function, if you
provide a value for peak_mjd the code will automatically convert this into rest-frame phase. The model does fit for
time of maximum so this doesn't have to be exact, it just needs to be a rough guess. When fitting, the prior on
time-of-maximum is a uniform distribution 10 rest-frame days either side of this initial value. Alternatively, if the
data you have is already in rest-frame phase, just don't provide a value for peak_mjd and the phases will be left as
they are.

If you have mag data, you can just use mag and mag_err instead of flux and flux_err and set the kwarg ``mag=True``,
this way your mag data will automatically be converted into flux space before fitting.

Fitting with fixed parameters
----------------------------------
Both ``model.fit()`` and ``model.fit_from_file()`` allow you to fit light curves with specific parameters fixed, if desired.

If you use the kwarg ``fix_tmax=True``, tmax will not be inferred and will instead be fixed to the peak MJD you
specified/present in the input file, or if you already have rest-frame phases will just be fixed to zero.

If you want to fix either theta or AV, you can do this by setting the optional kwargs ``fit_theta`` and ``fix_AV`` to
the value you want these parameters fixed to e.g. if you use ``fit_theta=1``, theta will be fixed to 1 for your fits.

Testing different dust parameters
---------------------------------------
The built-in BayeSN models generally assume a single fixed host galaxy RV for all SNe, although plenty of BayeSN papers
have explored population distributions of RV as well. You can easily experiment with different values/distributions of
RV during the fitting. For example, you change the single fixed RV value by setting the optional kwarg ``RV`` e.g.
``RV=2.8``. Alternatively, you can set a population distribution of RV values by specifying the mean ``mu_R`` and standard
deviation ``sigma_R`` of the distribution using the kwargs of those names. Note that you must specify both of these,
setting ``mu_R`` but not ``sigma_R`` will raise an error. This works the same for both fit and fit_from_file.

Plotting the fits
--------------------

The notebook `example_fits.ipynb` included in `here <https://github.com/bayesn/bayesn>`_ also shows an example of
plotting light curves fits. This is discussed here in more detail in :ref:`plotting`.
