.. bayesn documentation master file, created by
   sphinx-quickstart on Thu Aug 24 13:18:09 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BayeSN
==================================

Welcome to the documentation for BayeSN.

About
------------------------------------
This is the documentation for BayeSN, the hierarchical Bayesian optical-NIR SED model for type Ia supernovae as outlined
in Mandel+2022 (MNRAS 510, 3, 3939-3966). This implementation is introduced and briefly described in Grayling+2024
(MNRAS 531, 953, `arXiv link <https://arxiv.org/abs/2401.08755>`_), and is built on numpyro and jax to enable support
for GPU acceleration.

The model can be used to constrain physical population-level parameters of the distribution of SNe Ia using hierarchical
Bayesian inference, or alternatively to infer latent SN parameters including distance (suitable for use in cosmological
analyses) conditioned on fixed population-level parameters.

Support
---------
Our goal with BayeSN is for the package to be clear and easy to use. If you find something unclear, come across any bugs
or think of any functionality that would add a lot of value for you, please raise a Github issue here:
https://github.com/bayesn/bayesn/issues

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   installation
   running
   fitting
   filters
   output
   plotting
   builtin
   modules

..
   Indices and tables
   ========================================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

Citing the code/model
========================================

If you utilise the BayeSN model in any way, please cite Mandel+2022 (MNRAS 510, 3, 3939-3966), and if you make use of
this code please cite Grayling+2024 (MNRAS 531, 953, `arXiv link <https://arxiv.org/abs/2401.08755>`_). If you make
use of variational inference to fit light curves, please cite Uzsoy+2024 (MNRAS 535, 2306) which developed and assessed
this code.

In addition, if you use any of the pre-trained models included within BayeSN please cite the corresponding papers which
present those models:

- G26 model: Grayling et al. 2026 (arXiv:2606.19429)
- G25 model: Grayling et al. 2026 (MNRAS 548, stag340)
- W22 model: Ward et al. 2022 (ApJ 956, 2, 111)
- T21 model: Thorp et al. 2021 (MNRAS 508, 3, 4310-4331)
- M20 model: Mandel et al. 2022 (MNRAS 510, 3, 3939-3966)

Other works which have used BayeSN include:

- Thorp & Mandel 2022 (MNRAS 517, 2, 2360-2382)
- Dhawan et al. 2023 (MNRAS 524, 1, 234-244)


