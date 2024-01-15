Overview
===============================

The BayeSN model
-----------------------

BayeSN is a hierarchical Bayesian SED model for SNe Ia. This hierarchical approach is ideal for analysing the
distributions of intrinsic and extrinsic (host galaxy dust) properties of SNe Ia, and ensures that the full time- and
wavelength-evolution of the population is used when constraining these effects. In addition, BayeSN can be used to
infer latent SN properties including a cosmology-independent distance modulus, conditioned on fixed population
parameters. This model can therefore be used for cosmological analyses.

This code will allow you to train BayeSN models (obtain posteriors on global, population-level parameters while
marginalising over latent, individual SN parameters) as well as fit using pre-trained models to infer SN parameters
including distance.

Inference using the BayeSN model is performed using Hamiltonian Monte Carlo (HMC), an MCMC method which uses gradient
information to make efficient steps through the parameter space.

GPU acceleration
------------------

This code is built on numpyro and jax, and has been designed with GPU acceleration in mind;  running on a GPU should
yield a considerable (~100 times) increase in performance over using a CPU. To enable this, all calculations are handled
in parallel as large tensor operations which each handle all separate SNe, phases and filters simultaneously. However,
it is important to note that GPUs will show the most benefit running large scale jobs. If you want to fit samples of
hundreds of SNe, the fit time per object will be considerably shorter than fitting just a handful. With only 1 object,
you are likely better off running on CPU than GPU.

Depending on the GPUs you have access to and the cadence of your light curves, you should be able to fit thousands of
SNe in parallel. In testing using 4 A100 GPUs, fits have successfully run upwards of 15,000 *griz*
light curves with a ~4 day cadence as part of a single job. Eventually, you will start running out of memory and will
need to split the data set up into batches but this should only be a concern when fitting large scale simulations.

To give an indication of performance, in these tests mentioned above typical fit times were ~1-1.2s per object for
sample sizes upwards of hundreds of SNe. This is the time taken for a full MCMC fit across 29 dimensions (including a
stretch-like parameter :math:`\theta_1`, :math:`A_V`, distance modulus, time of maximum and 24 parameters relating to
the residual intrinsic SN colour distribution). This is a considerable speed increase enabled using GPUs, and further
speed improvements can be achieved through the use of variational inference rather than MCMC, which will be incorporated
within this code in future.

Redshift requirement
----------------------

At the present time, BayeSN is designed for use with spectroscopic redshifts, which are kept fixed during inference.
However, fitting with redshift as a free parameter based on a photometric redshift prior is feasible and planned in
future.

..
    BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily
    implement your own filter responses based on a simple yaml file. However, in order to allow for quick start-up, we
    provide as a separate download a large set of filter responses along with an associated filters.yaml file which can
    be used by BayeSN straight away.