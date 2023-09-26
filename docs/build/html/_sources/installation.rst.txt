Installation
======================

BayeSN requires Python 3.10+ and the dependencies outlined below. **Please note** that special care must be
taken when installing the dependencies to allow BayeSN to run on GPUs - please see below.

Dependencies
------------

BayeSN depends on the following packages

* jax (see note below)
* numpy
* matplotlib
* pandas
* arviz
* astropy
* sncosmo
* extinction
* h5py
* scipy
* ruamel.yaml
* tqdm
* numpyro

If you want to use GPUs, you must take care to install the correct version of jax following instructions below.

Requirements for GPU
~~~~~~~~~~~~~~~~~~~~~
* cudatoolkit > 11.8
* cudnn > 8.6
* jax version which matches cudatoolkit/cudnn version, instructions below

To use GPUs, you need to install a version of jax specific for GPUs - the default pip install is CPU only. In addition,
the jax version will need to match the version of cudatoolkit and cudnn you have installed. Full installation
instructions for jax GPU can be found here: https://github.com/google/jax#installation.

Install using pip (recommended)
--------------------------------

BayeSN is available on PyPI, and be installed by running

``pip install bayesn``

**However, take care if you want to run using GPUs**. In this case, you must install a version of jax compatible with
GPUs **before** pip installing BayeSN, following the instructions above. This is because installing via pip will also
pip install jax if not already installed, and the default version installed via pip only supports CPU. If you only want
to run on CPU, you can just pip install without worrying about this.

Install development version
----------------------------
BayeSN is also available on Github: