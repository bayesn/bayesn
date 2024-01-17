# BayeSN

Hierarchical Bayesian SED modelling of type Ia supernova light curves

Developed and maintained by: Matt Grayling (@mattgrayling), Stephen Thorp (@stevet40), Gautham Narayan (@gnarayan), and 
Kaisey S. Mandel (@CambridgeAstroStat) on behalf of the BayeSN Team (@bayesn).

## About
BayeSN is a probabilistic optical-NIR SED model for type Ia supernovae, allowing for hierarchical analysis of the
population distribution of physical properties as well as cosmology-independent distance estimation for individual
SNe. This repository contains an implementation of the BayeSN SED model built with numpyro and jax, with support for 
GPU acceleration, as discussed in Grayling+2024 (submitted to MNRAS).

## Installation and usage
BayeSN can be pip-installed via the command `pip install bayesn`. 

Detailed instructions on how to install and run the BayeSN model can be found here: https://bayesn.readthedocs.io/en/latest/index.html

## Support
Our goal with BayeSN is for the package to be clear and easy to use. If you find something unclear, come across any bugs
or think of any functionality that would add a lot of value for you, please raise a Github issue.

## Citing the code/model
If you utilise the BayeSN model in any way, please cite Mandel+2022 (MNRAS 510, 3, 3939-3966), and if you make use of
this code please cite Grayling+2024 (submitted to MNRAS). In addition, if you use any of the pre-trained models included
within BayeSN please cite the corresponding papers which present those models:

- M20 model: Mandel et al. 2022 (MNRAS 510, 3, 3939-3966)
- T21 model: Thorp et al. 2021 (MNRAS 508, 3, 4310-4331)
- W22 model: Ward et al. 2023 (ApJ 956, 2, 111)

## Things to note

### Filters

BayeSN includes a set of built-in filters for convenience, as detailed in the documentation. However, you can also add 
your own  filters through a yaml file to allow for new and updated filter responses to be used without needing a package
update.

### GPU Acceleration

This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times)
increase in performance. However, it is important to note that GPUs will show the most benefit running large scale jobs.
If you want to fit samples of 100s of SNe, the fit time per object will be considerably shorter than fitting just a 
handful. With only 1 object, you are likely better off running on CPU than GPU.

### Redshift requirement

At the present time, BayeSN is designed for use with spectroscopic redshifts, which are kept fixed during inference.
However, fitting with redshift as a free parameter based on a photometric redshift prior is feasible and planned in
future.
