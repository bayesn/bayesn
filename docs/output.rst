Output
=============

The output of BayeSN will vary depending on whether the mode you are using. The output will be saved in
``outputdir/name`` where ``outputdir`` and ``name`` correspond to the keys present in the input file as described in
:ref:`running_bayesn`.

Training output
-------------------

The output of a training job will have the following structure:

- ``bayesn.yaml``: A yaml file containing the inferred global parameter values, which can then be used to fit data.
- ``fit_summary.csv``: A summary of the MCMC output, showing parameter means/medians etc. as well as the Gelman-Rubin statistic and effective sample sizes to assess fit quality.
- ``initial_chains.pkl``: The MCMC chains containing posterior samples, prior to any postprocessing, saved as a pickle file. This is a dictionary, with the keys corresponding to each parameter and the values the posterior samples for that parameter.
- ``chains.pkl``: The same as above, except after postprocessing is applied. Postprocessing is required for a number of reasons. For example in the BayeSN model there exists a mirror degeneracy between theta and W1 whereby flipping the signs on both will lead to an identical output since they are multiplied together. As a result, sometimes different chains can move towards mirrored solutions. Postprocessing corrects for this to ensure that all chains have the same sign for elements of W1/theta values.

Fitting output
---------------

The output of a fitting job will have the following structure:

- ``fit_summary.csv``: A summary of the MCMC output, showing parameter means/medians etc. as well as the Gelman-Rubin statistic and effective sample sizes to assess fit quality.
- ``chains.pkl``: The MCMC chains, as for the training output. Unlike for training, no postprocessing is required therefore only one set of chains needs to be saved.
- ``output.fitres``: A FITRES file of the same structure as those returned by fits done within SNANA e.g. from SALT, summarising the properties of each fit light curve.
- ``output.LCPLOT``: An LCPLOT data table containing both the data that was fit and corresponding model fits (with or without errors depending on whether you set ``save_fit_errors`` in the input yaml. Data is stored in rows with DATA_FLAG=1, while the fit is stored in rows with DATA_FLAG=0.

The plan for large SNANA jobs is that only the last of these outputs will be saved to avoid creating a very large number of output files.

As discussed in :ref:`running_bayesn`, when running BayeSN fits on a sample of SNe all objects are fit in parallel in a single
job, rather than having a separate job for each SN. These output files therefore contain the outputs for all SNe in the
sample.