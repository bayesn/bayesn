# BayeSN
This is a development version of BayeSN implemented in numpyro. A full, pip-installable release with documentation is to follow shortly.

A paper to accompany this code release is in prep and will accompany the full public release. In the meantime, if you make use of any code in this repo please cite:

- Mandel K.S., Thorp S., Narayan G., Friedman A.S., Avelino A., 2022, MNRAS, 510, 3939
- Thorp S., Mandel K.S., Jones D.O., Ward S.M., Narayan G., 2021, MNRAS, 508, 4310
- Thorp S., Mandel K.S., 2022, MNRAS, 517, 2360
- Ward, S. et al., arXiv:2209.10558 (accepted by ApJ)

Developed and maintained by: Matt Grayling (@mattgrayling), Stephen Thorp (@stevet40), Gautham Narayan (@gnarayan), and Kaisey S. Mandel (@CambridgeAstroStat) on behalf of the BayeSN Team (@bayesn)

This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times) increase in performance. To enable this, all SNe are treated in parallel with vectorized calculations handling all SNe simultaneously. As such, it is important to note that GPUs will show the most benefit running large scale jobs. If you want to fit samples of 100s of SNe, the fit time per object will be considerably shorter than fitting just a handful. With only 1 object, you are likely better off running on CPU than GPU.

At the present time, BayeSN is designed for use with spectroscopic redshifts although fitting with a free redshift based on a photometric redshift prior is feasible and planned in future.

BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily implement your own filter responses based on a simple yaml file. However, in order to allow for quick start-up, we provide as a separate download a large set of filter responses along with an associated filters.yaml file which can be used by BayeSN straight away.

# Installation Guide

## Dependencies

### Python packages:

BayeSN depends on the following packages

- numpy
- matplotlib
- pandas
- arviz
- astropy
- sncosmo
- extinction
- h5py
- jax (see note below)
- scipy
- ruamel.yaml
- tqdm
- numpyro

If you want to use GPUs, you must take care to install the correct version of jax following instructions below.

### Requirements for GPU:

- cudatoolkit > 11.8
- cudnn > 8.6
- jax version which matches cudatoolkit/cudnn version, instructions below

To use GPUs, you need to install a version of jax specific for GPUs - the default pip install is CPU only. In addition, the jax version will need to match the version of cudatoolkit and cudnn you have installed. Full installation instructions for jax GPU can be found here: https://github.com/google/jax#installation.


<!---
# BayeSN-numpyro
This is a numpyro implementation of BayeSN, the optical-NIR SED model for type Ia supernovae as outlined in Mandel+2020 MNRAS 510, 3. This code will allow you to train new models (obtain posteriors on global, population-level parameters while marginalising over latent, individual SN parameters) as well as fit using existing models (obtain posteriors on latent SN parameters conditioned on fixed global parameters inferred during training).

This code has been designed with GPU acceleration in mind, and running on a GPU should yield a considerable (~100 times) increase in performance. To enable this, all SNe are treated in parallel with vectorized calculations handling all SNe simultaneously. As such, it is important to note that GPUs will show the most benefit running large scale jobs. If you want to fit samples of 100s of SNe, the fit time per object will be considerably shorter than fitting just a handful. With only 1 object, you are likely better off running on CPU than GPU.

At the present time, BayeSN is designed for use with spectroscopic redshifts although fitting with a free redshift based on a photometric redshift prior is feasible and planned in future.

BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily implement your own filter responses based on a simple yaml file. However, in order to allow for quick start-up, we provide as a separate download a large set of filter responses along with an associated filters.yaml file which can be used by BayeSN straight away.

# Installation Guide

## Dependencies

### Python packages:

BayeSN depends on the following packages

- numpy
- matplotlib
- pandas
- arviz
- astropy
- sncosmo
- extinction
- h5py
- jax (see note below)
- scipy
- ruamel.yaml
- tqdm
- numpyro

If you want to use GPUs, you must take care to install the correct version of jax following instructions below.

### Requirements for GPU:

- cudatoolkit > 11.8
- cudnn > 8.6
- jax version which matches cudatoolkit/cudnn version, instructions below

To use GPUs, you need to install a version of jax specific for GPUs - the default pip install is CPU only. In addition, the jax version will need to match the version of cudatoolkit and cudnn you have installed. Full installation instructions for jax GPU can be found here: https://github.com/google/jax#installation.

## Installing using pip

BayeSN can be installed using pip:

```
pip install bayesn
```

**However, take care if you want to run using GPUs**. In this case, you must install a version of jax compatible with GPUs **before** pip installing BayeSN, following the instructions above. This is because installing via pip will also pip install jax, and the default version installed via pip only supports CPU. If you only want to run on CPU, just install using pip.

# Running BayeSN jobs

BayeSN jobs are run just by running the script `run_bayesn` (after installation, this script can be called from any directory), with the specific job defined by an input yaml file which allow you to specify e.g. whether you want to run model training or fitting, which data you want to use and where your filters are defined. To run a job, you can just run the following command, making sure to specify the path to your `input.yaml` file.

```
run_bayesn --input PATH\TO\input.yaml
```

The `input.yaml` therefore underpins this code. Examples and explanations of all the keys are given below.

Please note, if you do not specify `--input`, the code will look for a file called `input.yaml` in the directory where you run this command from.

## Specifying input to BayeSN

The keys which can be specified are described below. Depending on whether you are training or fitting, different keys will be required:

- `name`: Specifies the name of the folder that the output of the run will be saved in. In tandem with `outputdir`, this will specify exactly where the results are saved.
- `outputdir`: The directory in which to save the output folder. 
- `mode`: The BayeSN mode to use, can be used to specify whether you want to train or fit.
- `load_model`: The existing BayeSN to use if you are fitting. You can either specify one of the built-in models (one of 'M20_model', 'T21_model' or 'W22_model') or use a path to a model you have trained yourself (defined by a bayesn.yaml file, see details of the output below).
- `num_warmup`: The number of warmup steps for HMC. Typically 500 is sufficient for training and 250 for fitting for convergence, but you may need to increase this.
- `num_samples`: The number of posterior samples to take.
- `num_chains`: The number of MCMC chains to run. Using HMC, it is recommended to use at least 4 chains to assess model convergence. 
- `filters`: Path to a yaml file describing the filters you want to use, see below for more details.
- `chain_method`: The method to use for running multiple chains in numpyro. If 'sequential', chains will be run one-after-the-other until all are complete. If 'parallel', the chains will be run in parallel over multiple devices - with 4 chains and a node with 4 GPUs, the chains will be run simultaneously in parallel. If 'vectorized', chains will be run in parallel on a single device which may or may not be quicker than running them sequentially depending on the device you are using, and may result in memory issues unless you are using a large GPU.
- `initialisation`: The strategy used to initialise the HMC chains. Must be one of:
  - 'median': Initialise chains to the prior media with some additonal scatter between chains applied by numpyro.
  - 'sample': Initialise chains to random samples from the priors.
  - 'T21': Should only be used when training the model. Will initialise some global parameters to those of the T21 BayeSN model, specifically W0 and W1. This is useful just to stop these complex, multidimensional parameters starting in a very bad region of parameter space. Other parameters are random samples from the priors. If using different wavelength knots to T21, the inital values for W0 and W1 matrices will be determined by cubic spline interpolation of the originals if within the wavelength range covered by T21, and initialised to 0 otherwise.
  - 'M20': The same as above but using the M20 model rather than T21.
- `l_knots`: The wavelength knot locations for the 2d cubic spline surfaces that define the BayeSN model. These only need to be specified when training, as they are already specified in the model definition when fitting.
- `tau_knots`: The rest-frame phase knot locations for the 2d cubic spline surfaces that define the BayeSN model. These only need to be specified when training, as they are already specified in the model definition when fitting.
- `map`: A set of `key: value` pairs specifying a mapping between filter names. You will want to use this if the filter names in the data files you are using do not match the names in your `filters.yaml` file (see below). For example, if your data files use the filter names _griz_ and you want ensure that you use DES filters you can set `map: {g: g_DES, r: r_DES, i: i_DES, z: z_DES}`, and just make sure that the filter names _g_DES_ etc. are in your `filters.yaml` file. If the filter names in your data files already match those in `filters.yaml`, this is not required.
- `data_table`: The path to a table providing paths to data files and metadata for each SN - see below for the required format. This should be used **only** if working with real data, as if you are using an SNANA simulation we can expect all files to be located in the same folder and be confident about the information contained in the headers - in this case, use the `data_dir` key instead.
- `data_root`: A root directory which will be pre-pended to all file paths in input_table. Not required if using `data_dir`, only for `data_table`.
- `data_dir`: The path to a directory containing that the output of an SNANA simulation you wish to use an input data. This should be used **only** if you are using the output of an SNANA simulation, where the data structure is more predictable and standard than using real data. Otherwise, use a combination of `data_table` and `data_root` instead.
- `yamloutputfile`: The name of an output yaml file, required only for SNANA runs to check job success, which will be saved in `outputdir`. If not specified, defaults to `output.yaml`.
- `drop_bands`: A list of bands which are present in the data files that you are using which you do not wish to include in the analysis, optional.

You can see an example of input.yaml files for training and fitting below.

### Training example

This example demonstrates the input.yaml that could be used to train the BayeSN model presented in Thorp+2021

```
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
```

### Fitting example

This example demonstrates the input.yaml that could be used to fit some SNANA simulations using a custom BayeSN model defined in a bayesn.yaml file.

```
name: M20_training_example
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
```

## Specifying data to use

As discussed above, if you are using the output an SNANA simulation as input you need only pass the location of the SNANA output to the `data_dir` key in the input file. However, if you are using real data, you may want to use data spanning multiple surveys which means you won't necessarily be able to point to a single directory. In this case, you should use the keys `data_table` and `data_root` in the input. `data_table` should contain file paths to the data for each SN as well as associated metadata for the SN, with the following structure:

```
SNID	SEARCH_PEAKMJD	REDSHIFT_CMB	REDSHIFT_CMB_ERR	files
SN1	57400	0.02	0.0001	survey1/SN1.txt
SN2	57500	0.03	0.0001  survey1/SN2.txt
SN3	57600	0.04	0.0001	survey1/SN3.txt,survey2/SN3.txt
SN4	57700	0.05	0.0001	survey3/SN3_optical.txt,survey3/SN3_NIR.txt
SN5	57800	0.06	0.0001	survey4/SN4.txt
```
The table allows for multiple files per object if required, the file names just need to be separated by commas in the files columns. This approach allows you to read in data from multiple surveys, including cases where the same object has observations from multiple surveys which are contained in different data files. This is also relevant for cases where one object may have both optical and NIR data which are contained in different files.

The table should include CMB-frame redshifts and associated uncertainties. This will be used to fix distance when training, although distance is a free parameter when fitting with redshift used only to determine filter responses and phase. The time of B-band maximum, SEARCH_PEAKMJD, need only be a rough estimate when fitting as the model will also infer the time of maximum, using a uniform prior covering 10 rest-frame days either side of the specified SEARCH_PEAKMJD.

The key `data_root` simply specifies the location that the file paths in `data_table` are defined with respect to. For example, with `data_root: /data/photometry/`, the full file path for the first file in the table above will be `/data/photometry/survey1/SN1.txt` and similar for the rest.

## Specifying filters

One of the arguments for the `input.yaml` file above, `filters`, is used to specify a path to a separate yaml file which details the filters you wish to use. This can be a small file containing a small number of filters, or one large file containing all the filters you might ever possibly want to use which only needs to be made once. This file should have the following structure:

```
standards_root: /PATH/TO/STANDARDS/ROOT
standards:
  vega:
    path: VEGA_STANDARD.fits/.dat
  bd17:
    path: BD17_STANDARD.fits/.dat
filters_root: /PATH/TO/FILTERS/ROOT
filters:
  test_band_1:
    magsys: ab
    magzero: 0
    path: test_band_1_response.dat
  test_band_2:
    magsys: vega
    magzero: 0
    path: test_band_2_response.dat
```

These arguments are described as follows:
- `standards_root`:  A directory which all paths in `standard` are defined relative to. For example, if the standard spectrum for Vega is located at `\data\standards\VEGA_STANDARD.fits` and BD17 is at `\data\filters\BD17_STANDARD.fits`, you can just set `standards_root: \data\standards` and use `path: VEGA_STANDARD.fits` within the key for Vega and similar for BD17. Alternatively, if you use a relative path this will be treated as being relative to the location of the filters yaml file. You can also use an environment variable here as part of the path e.g. $SNDATA_ROOT. This is an optional argument present for convenience, if not specified it is assumed that the paths for each band are all full paths rather than paths relative to `standards_root`.
- `standards`: Keys in here define all of the standards you wish to use. For each standard, the key is the name (this can be any string of your choosing), and each must have a `path` specifying the location of the reference spectrum for each standard - this can be either a FITS file with named columns for WAVELENGTH and FLUX, or a text file with columns for each.
- `filters_root`: This specifies a directory which all paths in `filters` are defined relative to, behaving exactly as `standards_root` does for `standards`. Again, if you use a relative path this will be treated as being relative to the location of the filters yaml file.
- `filters`: Keys in here define all of the filters you wish you use. For each filter, the key is the name (again, this can be any string of your choosing). Each filter must have a `magsys` key which either corresponds to one of the standard names defined in `standards` or is set to 'ab' (see note below), defining the magnitude system for each band. Each filters must also have a `magzero` key, specifying the magnitude offset for the filter, and a `path` specifying the location of the filter response for each filter.

Please note, the AB reference source is treated as an analytic function within the code so nothing needs to be included in `standards` for the AB magnitude system, any filter with `magsys: ab` will automatically work. If your filters only use the AB magnitude system, you can just omit the `standards` and `standards_root` keys entirely.

**We optionally provide a collection of common filters along with an associated filters.yaml detailing all of them.** You can simply download bayesn-filters and set `filters: \PATH\TO\bayesn-filters/filters.yaml` in the input yaml file, which will enable you to use all of these inbuilt filters.

The wavelength range covered by the model will depend on exactly which model you use. Filters will automatically be dropped for individual SNe when they fall out of the rest-frame wavelength range covered by the based on their redshift. The uppper and lower cut off wavelengths for each filter are defined as the wavelength where the filter response first drops below 0.01 times the maximum value.

## Output

The output of BayeSN will vary depending on whether you are training or fitting. The output will be saved in `outputdir/name` where `outputdir` and `name` correspond to the keys present in the input file as described above.

### Training output

The output of a training job will have the following structure:

- `fit_summary.csv`: A summary of the MCMC output, showing parameter means/medians etc. as well as the Gelman-Rubin statistic and effective sample sizes to assess fit quality.
- `initial_chains.pkl`: The MCMC chains containing posterior samples, prior to any postprocessing, saved as a pickle file. This is a dictionary, with the keys corresponding to each parameter and the values the posterior samples for that parameter.
- `chains.pkl`: The same as above, except after postprocessing is applied. Postprocessing is required for a number of reasons. For example in the BayeSN model there exists a mirror degeneracy between theta and W1 whereby flipping the signs on both will lead to an identical output since they are multiplied together. As a result, sometimes different chains can move towards mirrored solutions. Postprocessing corrects for this to ensure that all chains have the same sign for elements of W1/theta values.

### Fitting output

The output of a fitting job will have the following structure:

- `fit_summary.csv`: A summary of the MCMC output, showing parameter means/medians etc. as well as the Gelman-Rubin statistic and effective sample sizes to assess fit quality.
- `chains.pkl`: The MCMC chains, as for the training output. Unlike for training, no postprocessing is required therefore only one set of chains needs to be saved.
  


!--->
