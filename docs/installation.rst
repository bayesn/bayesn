Installation
======================

BayeSN requires Python 3.10+ and the dependencies outlined below. **Please note** that special care must be
taken when installing the dependencies to allow BayeSN to run on GPUs - please see below.

Dependencies
------------

BayeSN depends on the following packages

* jax >= 0.4.27, <=0.6.0 (see note below)
* jaxlib >= 0.4.27, <=0.6.0
* numpy >= 1.23.5
* matplotlib >= 3.5
* pandas >= 1.4.0
* arviz >= 0.0.1
* astropy >= 6.0.0
* sncosmo >= 2.10
* h5py >= 3.6.0
* scipy >= 1.15.0
* ruamel.yaml 0.16.0
* tqdm >= 4.29.1
* numpyro >= 0.14.0

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

BayeSN is available on PyPI, and can be installed by running

``pip install bayesn``

**However, take care if you want to run using GPUs**. In this case, you must install a version of jax compatible with
GPUs **before** pip installing BayeSN, following the instructions above. This is because installing via pip will also
pip install jax if not already installed, and the default version installed via pip only supports CPU. If you only want
to run on CPU, you can just pip install without worrying about this.

Install development version
----------------------------
BayeSN is also available on Github and can be found here: https://github.com/bayesn/bayesn.
The repository includes https://github.com/djones1040/Foundation_DR1 as a submodule for testing T21_model training.
Most users will not need to interact with this submodule, but interested users should clone the BayeSN repo with
the ``--recurse-submodules`` flag to initialise the submodule.

To install BayeSN, download or clone the repo and run ``pip install .`` in the base directory.
To include the dependencies that enable testing instead run ``pip install -e ".[testing]"``.
