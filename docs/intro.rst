Overview
===============================

The BayeSN model
-----------------------

BayeSN is a hierarchical Bayesian SED model for SNe Ia. This hierarchical approach is ideal for analysing the
distributions of intrinsic and extrinsic (host galaxy dust) properties of SNe Ia, and ensures that the full time- and
wavelength-evolution of the population is used when constraining these effects. In addition, BayeSN can be used to
infer latent SN properties including a cosmology-independent distance modulus, conditioned on a trained model (fixed
population parameters). This model can therefore be used for cosmological analyses.

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
hundreds of SNe, the fit time per object will be considerably shorter than fitting just a handful. For very small
samples you are likely better off running on CPU and fitting objects serially rather than doing one big parallel fit
(see :ref:`filters` for details of how to do this).

Depending on the GPUs you have access to and the cadence of your light curves, you should be able to fit thousands of
SNe in parallel. In testing using 4 A100 GPUs, fits have successfully run upwards of 15,000 *griz*
light curves with a ~4 day cadence as part of a single job. Eventually, you will start running out of memory and will
need to split the data set up into batches but this should only be a concern when fitting large scale simulations. This
type of job splitting is trivial when using `submit_batch_jobs.sh` within SNANA to deploy BayeSN jobs.

Typical performance
----------------------

To give an indication of performance, in these tests mentioned above typical fit times were ~1s per object for
sample sizes upwards of hundreds of SNe, using HMC. This is the time taken for a full MCMC fit across 29 dimensions
(including a stretch-like parameter :math:`\theta_1`, :math:`A_V`, distance modulus, time of maximum and 24 parameters
relating to the residual intrinsic SN colour distribution). This is a considerable speed increase enabled using GPUs.

BayeSN also includes support for variational inference (VI), applying a specially developed zero lower-truncated normal
(ZLTN) distribution to constrain AV values to be positive while allowing other parameters to be treated as a
multivariate Gaussian, as presented in Uzsoy+2024 (to be submitted). Using VI, fitting can be further sped up to
~0.2s per object.

Redshift requirement
----------------------

At the present time, BayeSN is designed for use with spectroscopic redshifts, which are kept fixed during inference.
However, fitting with redshift as a free parameter based on a photometric redshift prior is feasible and planned in
future.