.. _plotting:

Plotting BayeSN light curves and spectra
==========================================

Currently, BayeSN does not include dedicated plotting tools. However, it does provide tools to allow you easily to
retrieve model fluxes/magnitudes from posterior samples or simulate light curves/spectra from the SED model, ready for
plotting using your preferred approach.

Plotting light curve fits from posterior samples
-----------------------------------------------------------

You can use the ``get_flux_from_chains`` method of the ``bayesn.bayesn_model.SEDmodel`` class to get model fit
photometry from a set of samples inferred when fitting with BayeSN. An example of doing so to get model magnitudes
is given below, and also in the notebook `example_fits.ipynb` included on the Github repo for BayeSN
`here <https://github.com/bayesn/bayesn>`_:

.. code-block:: python

    import numpy as np
    from bayesn.bayesn_model import SEDmodel

    # Get photometry for T21 model
    model = SEDmodel(load_model='T21_model')

    t = np.arange(-10, 40, 1) # Define rest-frame phases to calculate model photometry at
    bands = ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1'] # PS1 griz bands, for example
    z = # Whatever the real heliocentric redshifts of your sample are
    ebv_mw = # Whatever the real Milky Way extinction colour excesses are

    # Get model photometry for all posterior samples
    phot_grid = model.get_flux_from_chains(t, bands, 'PATH/TO/chains.pkl', z, ebv_mw, mag=True)

The output ``phot_grid`` contains model photometry at each phase requested, for each band requested and for all
posterior samples for all SNe in the sample. For the third argument, which specifies the MCMC chains, you can either
pass the chains themselves if they are already loaded into memory, or you can pass a path to the pickle file in which
they are saved and they will be opened automatically.

Flux grid is a multidimensional array with shape (number of SNe, number of posterior samples, number of bands,
number of phases). Let's say, for example, that you had a set of posterior chains from light
curve fits to a sample of 100 SNe Ia. You then set ``t = np.arange(-10, 40, 1)`` for the rest-frame phases you want
to obtain photometry for, and choose to calculate photometry in PS1 __griz__ bands, as in the example above. Finally,
let's say that in this case we have 4 MCMC chains, each with 250 posterior samples. In this
case, ``phot_grid.shape = (100, 1000, 4, 50)``, following the pattern described above.

After you have ``phot_grid``, you can index whichever dimensions you want to make plots. You could just plot
realisations of the posterior for each SN by looping over the first two dimensions. You could also, for example,
calculate a mean and standard deviation across the light curves in each band for all the SNe to get a mean fit and
corresponding uncertainty region, which you could use to make plots showing the BayeSN fits to each SN. You could do
this by calculating:

.. code-block:: python

    mu = phot_grid.mean(axis=1) # Calculate mean over posterior samples for all SNe
    std = phot_grid.std(axis=1) # Calculate standard deviation over posterior samples for all SNe

After you do this, you'll find that ``mu.shape = (100, 4, 50)`` and ``std.shape = (100, 4, 50)`` i.e. you've taken
summary statistics over all the posterior samples and removed that dimension of the array.

This approach should let you make whatever light curve plots you want.

Plotting simulated light curves and spectra from model
-----------------------------------------------------------

The ``bayesn.bayesn_model.SEDmodel`` class has methods ``simulate_light_curve`` and ``simulate_spectrum`` to allow you
to simulate photometry and spectroscopy from the BayeSN SED model, which can in turn be used to make plots however you
like. Interactive examples of both of these functions being used are provided in a Jupyter notebook,
``simulation_examples.ipynb``, present in the Github repo for this code. Alternatively, examples are also provided here.

Both ``simulate_light_curve`` and ``simulate_spectrum`` allow you to create synthetic data from the model for given
parameter values. You can specify values of each parameter, or alternatively if none are provided values will be sampled
from the prior distributions.

Simulating spectra
~~~~~~~~~~~~~~~~~~~~~

An example of simulating spectra from the BayeSN model is given below:

.. code-block:: python

    import numpy as np
    from bayesn.bayesn_model import SEDmodel

    # Simulate spectra from the M20 BayeSN model
    model = SEDmodel(load_model='M20_model', filter_yaml='PATH/TO/filters.yaml')

    ts = np.arange(-10, 40, 5) # Set of phases at which to generate spectra for each object
    N = 20 # Number of SNe to generate spectra for. If you specify parameter values, the number of values passed
           # needs to match N e.g. if you want spectra for 10 objects, specify 10 theta values (unless you want to
           # use the same value for all of them, in which case you can just pass one single value)
    sim = model.simulate_spectrum(ts, N, z=0.3, mu='z', ebv_mw=0.05)
    l, spec, params = sim

Here, ``l`` is the set of wavelengths (the wavelength spacing can be set using the option ``dl`` keyword argument),
while ``spec`` contain the actual spectra for all objects. The shape of ``spec`` is (number of SNe, number of wavelength
elements, number of phases). For spectra with 300 wavelength elements and with 20 simulated SNe, each simulated for 10
phases, ``spec.shape = (20, 300, 10)``. You can then index any dimensions you like to make plots. Finally, params is
just a dictionary which stores all of the true parameter values used to simulate the data.

All the parameters can be set via corresponding keyword arguments e.g. ``theta=``, ``AV=`` etc. otherwise if not set
samples will be drawn from the prior.

Simulating light curves
~~~~~~~~~~~~~~~~~~~~~~~~

An example of simulating light curves from the BayeSN model is given below:

.. code-block:: python

    import numpy as np
    from bayesn.bayesn_model import SEDmodel

    # Simulate light curves from the M20 BayeSN model
    model = SEDmodel(load_model='M20_model', filter_yaml='PATH/TO/filters.yaml')

    N = 100 # Number of SNe to simulate
    t = np.arange(-8, 40, 4) # Set of phases at which to generate photometry for each object
    bands = ['B_CSP', 'V_CSP', 'r_CSP', 'i_CSP', 'Y_RC', 'J_RC1', 'H_RC']
    z = np.random.uniform(0, 0.1, N)
    sim = model.simulate_light_curve(t, N, bands, yerr=0.05, z=z, mu='z', ebv_mw=0, mag=True, write_to_files=False)
    mag, mag_err, params = sim

Here, ``mag`` contains simulated magnitudes and `mag_err` contains their corresponding uncertainties, set here just to
an arbitrary value of 0.05 mag for each observation, while ``params`` is a dictionary which stores the true parameter
values for each simulated SN. Once you have simulated photometry from the model, you can use that to create light curve
plots.

Regarding the phases and bands, you can use this model in two ways. If ``len(bands) == len(t)``, this code will assume
that you have a set of phases and the corresponding filter used for each time of observation, and will generate one
photometric value for each phase only for that given band i.e. with 20 phases, you will get 20 data points per obejct.
This might be the case if you have a set of phases and bands derived from a real survey. Otherwise, if
``len(bands) != len(t)``, the code will work in the same way as ``simulate_spectrum`` and just simulate photometry at
all of the requested phases for all bands i.e. with 4 bands and 20 phases you will get 80 data points per object.

The shape of mag and mag_err is (number of observations, number of SNe). In the example above,
``mag.shape = (84, 100)``. For the argument ``yerr``, which will lead to the output ``mag_err``, you could use an array
of the same length as t if you have realistic uncertainties for each phase that you wish you use, or you could just set
them all to a fixed magnitude error (or fixed SNR in the case of simulating flux values). Alternatively, if you want
exact model photometry just set ``yerr=0``.

In principle, this code could be used to forward simulate an entire observed SN Ia sample in a differentiable,
vectorised way if using realistic cadences and uncertainties. However, in reality this approach lacks some of the
finer details present in a SN survey. BayeSN is currently being implemented within SNANA to allow for more realistic
forward modelling.
