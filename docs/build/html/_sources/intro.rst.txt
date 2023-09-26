Overview
===============================

This code will allow you to train BayeSN models (obtain posteriors on global, population-level parameters while
marginalising over latent, individual SN parameters) as well as fit using existing models (obtain posteriors on latent
SN parameters conditioned on fixed global parameters inferrered during training).

This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times)
increase in performance. To enable this, all SNe are treated in parallel with vectorized calculations handling all SNe
simulatenously. As such, it is important to note that GPUs will show the most benefit running large scale jobs. If you
want to fit samples of 100s of SNe, the fit time per object will be considerably shorter than fitting just a handful.
With only 1 object, you are likely better off running on CPU than GPU.

Depending on the GPUs you have access to and the cadence of your light curves, you should be able to fit thousands of
SNe in parallel. In testing using A100 GPUs on NERSC Perlmutter, fits have successfully run upwards of 15,000 *griz*
light curves with a ~4 day cadence as part of a single job. Eventually, you will start running out of memory and will
need to split the data set up into batches but this should only be a concern when fitting large scale simulations.

..
    To give an indication of performance, previous tests looking at *griz* light curves with a ~4 day cadence have shown
    typical fit times of ~1-1.2s per object for sample sizes upwards of hundreds of SNe. This is the time taken for a full
    MCMC fit across 29 dimensions (including a stretch-like parameter :math:`\theta`, :math:`A_V`, distance modulus, time of
    maximum and 24 parameters relating to the residual intrinsic SN colour distribution).

At the present time, BayeSN is designed for use with spectroscopic redshifts although fitting with a free redshift based
on a photometric redshift prior is feasible and planned in future.

BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily implement your own filter responses based on a simple yaml file. However, in order to allow for quick start-up, we provide as a separate download a large set of filter responses along with an associated filters.yaml file which can be used by BayeSN straight away.