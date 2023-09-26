Output
=============

The output of BayeSN will vary depending on whether you are training or fitting. The output will be saved in
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