# BayeSN

Hierarchical Bayesian SED modelling of type Ia supernova light curves

Developed and maintained by: Matt Grayling (@mattgrayling), Stephen Thorp (@stevet40), Gautham Narayan (@gnarayan), and 
Kaisey S. Mandel (@CambridgeAstroStat) on behalf of the BayeSN Team (@bayesn)

## About
BayeSN is a probabilistic optical-NIR SED model for type Ia supernovae, allowing for hierarchical analysis of the
population distribution of physical properties as well as cosmology-independent distance estimation for individual
SNe. This repository contains an implementation of the BayeSN SED model built with numpyro and jax, with support for 
GPU acceleration, as discussed in Grayling+2024 (to be submitted).

## Installation and usage
Instructions on how to install and run the BayeSN model can be found here: (link to be inserted when uploaded to
readthedocs)

## Citing the code/model
If you utilise the BayeSN model in any way, please cite Mandel+2022 (MNRAS 510, 3, 3939-3966), and if you make use of
this code please cite Grayling+2024 (to be submitted). In addition, if you use any of the pre-trained models included
within BayeSN please cite the corresponding papers which present those models:

- M20 model: Mandel et al. 2022 (MNRAS 510, 3, 3939-3966)
- T21 model: Thorp et al. 2021 (MNRAS 508, 3, 4310-4331)
- W22 model: Ward et al. 2022 (ApJ 956, 2, 111)

## Things to note

### Filters

BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily 
implement your own filter responses based on a simple yaml file. However, in order to allow for quick start-up, we 
provide as a separate download a large set of filter responses along with an associated filters.yaml file which can be
used by BayeSN straight away.

### GPU Acceleration

This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times)
increase in performance. However, it is important to note that GPUs will show the most benefit running large scale jobs.
If you want to fit samples of 100s of SNe, the fit time per object will be considerably shorter than fitting just a 
handful. With only 1 object, you are likely better off running on CPU than GPU.

### Redshift requirement

At the present time, BayeSN is designed for use with spectroscopic redshifts, which are kept fixed during inference.
However, fitting with redshift as a free parameter based on a photometric redshift prior is feasible and planned in
future.
