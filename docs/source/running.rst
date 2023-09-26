.. _running_bayesn:

Running BayeSN Jobs
==========================================

BayeSN jobs are run just by running the script ``run_bayesn`` (after installation, this script can be called from any
directory), with the specific job defined by an input yaml file which allow you to specify e.g. whether you want to run
model training or fitting, which data you want to use and where your filters are defined. To run a job, you can just run
the following command, making sure to specify the path to your ``input.yaml`` file.

``run_bayesn --input PATH\TO\input.yaml``

The ``input.yaml`` therefore underpins this code. Examples and explanations of all the keys are given below. Please note,
if you do not specify ``--input``, the code will look for a file called ``input.yaml`` in the directory where you run this
command from.

Specifying input to BayeSN
---------------------------------

The keys which can be specified are described below. Depending on whether you are training or fitting, different keys will be required:

- ``name``: Specifies the name of the folder that the output of the run will be saved in. In tandem with `outputdir`, this will specify exactly where the results are saved.
- ``outputdir``: The directory in which to save the output folder.
- ``mode``: The BayeSN mode to use, specifies what you want to do. Options are:

  - 'training_singleRV': Train a model with a single RV values across the population
  - 'training_popRV': Initialise chains to random samples from the priors.
  - 'fitting': Fit using fixed global parameters previously inferred during model training. The code will figure out whether the model assumes a fixed RV or population RV based on the bayesn.yaml which defines the model you use to fit.
- ``trunc_RV``: Specifies the value at which to truncate the RV distribution when training a model with a population of RV values. If not specified, a Normal distribution without truncation will be used.
- ``load_model``: The existing BayeSN to use if you are fitting. You can either specify one of the built-in models (one of 'M20_model', 'T21_model' or 'W22_model') or use a path to a model you have trained yourself (defined by a bayesn.yaml file).
- ``num_warmup``: The number of warmup steps for HMC. Typically 500 is sufficient for training and 250 for fitting for convergence, but you may need to increase this.
- ``num_samples``: The number of posterior samples to take.
- ``num_chains``: The number of MCMC chains to run. Using HMC, it is recommended to use at least 4 chains to assess model convergence.
- ``filters``: Path to a yaml file describing the filters you want to use. For more details, please see :ref:`filters`.
- ``chain_method``: The method to use for running multiple chains in numpyro. If 'sequential', chains will be run one-after-the-other until all are complete. If 'parallel', the chains will be run in parallel over multiple devices - with 4 chains and a node with 4 GPUs, the chains will be run simultaneously in parallel. If 'vectorized', chains will be run in parallel on a single device which may or may not be quicker than running them sequentially depending on the device you are using, and may result in memory issues unless you are using a large GPU.
- ``initialisation``: The strategy used to initialise the HMC chains. Must be one of:

  - 'median': Initialise chains to the prior media with some additonal scatter between chains applied by numpyro.
  - 'sample': Initialise chains to random samples from the priors.
  - Alternatively, when training the model you can pass either the name of a built-in model or the path to a custom model (as for ``load_model`` above). This will initialise some global parameters to those of the that BayeSN model, specifically W0 and W1. This is useful just to stop these complex, multidimensional parameters starting in a very bad region of parameter space. Other parameters are random samples from the priors. If using different wavelength knots to the specified model, the initial values for W0 and W1 matrices will be determined by cubic spline interpolation of the originals if within the wavelength range covered by T21, and initialised to 0 otherwise.
- ``l_knots``: The wavelength knot locations for the 2d cubic spline surfaces that define the BayeSN model. These only need to be specified when training, as they are already specified in the model definition when fitting.
- ``tau_knots``: The rest-frame phase knot locations for the 2d cubic spline surfaces that define the BayeSN model. These only need to be specified when training, as they are already specified in the model definition when fitting.
- ``map``: A set of ``key: value`` pairs specifying a mapping between filter names. You will want to use this if the filter names in the data files you are using do not match the names in your `filters.yaml` file (see below). For example, if your data files use the filter names _griz_ and you want ensure that you use DES filters you can set ``map: {g: g_DES, r: r_DES, i: i_DES, z: z_DES}``, and just make sure that the filter names _g_DES_ etc. are in your ``filters.yaml`` file. If the filter names in your data files already match those in ``filters.yaml``, this is not required.
- ``data_table``: The path to a table providing paths to data files and metadata for each SN - see below for the required format. This should be used **only** if working with real data, as if you are using an SNANA simulation we can expect all files to be located in the same folder and be confident about the information contained in the headers - in this case, use the ``data_dir`` key instead.
- ``data_root``: A root directory which will be pre-pended to all file paths in input_table. Not required if using ``data_dir``, only for ``data_table``.
- ``data_dir``: The path to a directory containing that the output of an SNANA simulation you wish to use an input data. This should be used **only** if you are using the output of an SNANA simulation, where the data structure is more predictable and standard than using real data. Otherwise, use a combination of ``data_table`` and ``data_root`` instead.
- ``yamloutputfile``: The name of an output yaml file, required only for SNANA runs to check job success, which will be saved in ``outputdir``. If not specified, defaults to `output.yaml`.
- ``drop_bands``: A list of bands which are present in the data files that you are using which you do not wish to include in the analysis, optional.

You can see an example of input.yaml files for training and fitting below.

Example yaml files
------------------------------

Training example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the input.yaml that could be used to train the BayeSN model presented in Thorp+2021.

.. code-block:: yaml

    name: T21_training_example
    mode: training
    num_chains: 4
    num_warmup: 500
    num_samples: 500
    filters: /PATH/TO/filters.yaml
    chain_method: parallel
    initialisation: T21
    l_knots: [3500.0, 4900.0, 6200.0, 7700.0, 8700.0, 9500.0]
    tau_knots: [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    map: {g: g_PS1, r: r_PS1, i: i_PS1, z: z_PS1}
    data_root: /PATH/TO/DATA/ROOT
    input_table: T21_training_set.txt
    outputdir: /PATH/TO/OUTPUT/DIR
    yamloutputfile: test_out.yaml


Fitting example
~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the input.yaml that could be used to fit some SNANA simulations using a custom BayeSN model defined in a bayesn.yaml file.

.. code-block:: yaml

    name: custom_fitting_example
    mode: fitting
    load_model: /PATH/TO/CUSTOM/bayesn.yaml
    num_chains: 4
    num_warmup: 250
    num_samples: 250
    filters: /PATH/TO/filters.yaml
    chain_method: parallel
    initialisation: median
    data_dir: /PATH/TO/SNANA/SIMULATION
    outputdir: /PATH/TO/OUTPUT/DIR
    yamloutputfile: test_out.yaml

Specifying data to use
-------------------------------

As discussed above, if you are using the output an SNANA simulation as input you need only pass the location of the
SNANA output to the ``data_dir`` key in the input file. However, if you are using real data, you may want to use data
spanning multiple surveys which means you won't necessarily be able to point to a single directory. In this case, you
should use the keys ``data_table`` and ``data_root`` in the input. ``data_table`` should contain file paths to the data
for each SN as well as associated metadata for the SN, with the following structure:

.. code-block:: text

    SNID SEARCH_PEAKMJD	REDSHIFT_CMB REDSHIFT_CMB_ERR files
    SN1	57400	0.02	0.0001	survey1/SN1.txt
    SN2	57500	0.03	0.0001  survey1/SN2.txt
    SN3	57600	0.04	0.0001	survey1/SN3.txt,survey2/SN3.txt
    SN4	57700	0.05	0.0001	survey3/SN3_optical.txt,survey3/SN3_NIR.txt
    SN5	57800	0.06	0.0001	survey4/SN4.txt

The table allows for multiple files per object if required, the file names just need to be separated by commas in the
files columns. This approach allows you to read in data from multiple surveys, including cases where the same object has
observations from multiple surveys which are contained in different data files. This is also relevant for cases where
one object may have both optical and NIR data which are contained in different files.

The table should include CMB-frame redshifts and associated uncertainties. This will be used to fix distance when
training, although distance is a free parameter when fitting with redshift used only to determine filter responses and
phase. The time of B-band maximum, SEARCH_PEAKMJD, need only be a rough estimate when fitting as the model will also
infer the time of maximum, using a uniform prior covering 10 rest-frame days either side of the specified
SEARCH_PEAKMJD.

The key ``data_root`` simply specifies the location that the file paths in ``data_table`` are defined with respect to.
For example, with ``data_root: /data/photometry/``, the full file path for the first file in the table above will be
``/data/photometry/survey1/SN1.txt`` and similar for the rest.
