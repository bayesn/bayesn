.. _filters:

Defining Filters
===================

BayeSN does not include any inbuilt filters, favouring an approach separating filters from code allowing you to easily
implement your own filter responses based on a simple yaml file. This allows for new or updated filters to be easily
implemented independently of updates to the package. This section outlines the structure required for the yaml file
which defines the filters.

To allow for quick start-up, we provide as a separate download a large set of filter responses along
with an associated filters.yaml file which can be used by BayeSN straight away. Please see below for more details.

Specifying filters
-------------------

One of the arguments for the ``input.yaml`` file outlined in :ref:`running_bayesn`, ``filters``, is used to specify a
path to a separate yaml file which details the filters you wish to use. This can be a small file containing a small
number of filters, or one large file containing all the filters you might ever possibly want to use which only needs to
be made once. This file should have the following structure:

.. code-block:: yaml

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

These arguments are described as follows:

- ``standards_root``: A directory which all paths in ``standards`` are defined relative to. For example, if the standard spectrum for Vega is located at ``\data\standards\VEGA_STANDARD.fits`` and BD17 is at ``\data\filters\BD17_STANDARD.fits``, you can just set ``standards_root: \data\standards`` and use ``path: VEGA_STANDARD.fits`` within the key for Vega and similar for BD17. Alternatively, if you use a relative path this will be treated as being relative to the location of the filters yaml file. You can also use an environment variable here as part of the path e.g. $SNDATA_ROOT. This is an optional argument present for convenience, if not specified it is assumed that the paths for each band are all full paths rather than paths relative to ``standards_root``.
- ``standards``: Keys in here define all of the standards you wish to use. For each standard, the key is the name (this can be any string of your choosing), and each must have a ``path`` specifying the location of the reference spectrum for each standard - this can be either a FITS file with named columns for WAVELENGTH and FLUX, or a text file with columns for each.
- ``filters_root``: This specifies a directory which all paths in ``filters`` are defined relative to, behaving exactly as ``standards_root`` does for ``standards``. Again, if you use a relative path this will be treated as being relative to the location of the filters yaml file.
- ``filters``: Keys in here define all of the filters you wish you use. For each filter, the key is the name (again, this can be any string of your choosing). Each filter must have a ``magsys`` key which either corresponds to one of the standard names defined in ``standards`` or is set to 'ab' (see note below), defining the magnitude system for each band. Each filter must also have a ``magzero`` key, specifying the magnitude offset for the filter, and a ``path`` specifying the location of the filter response for each filter.

Please note, the AB reference source is treated as an analytic function within the code so nothing needs to be included
in ``standards`` for the AB magnitude system, any filter with ``magsys: ab`` will automatically work. If your filters
only use the AB magnitude system, you can just omit the ``standards`` and ``standards_root`` keys entirely.

bayesn-filters
-----------------

**We optionally provide a collection of common filters along with an associated filters.yaml detailing all of them.**
You can simply download ``bayesn-filters`` and set ``filters: /PATH/TO/bayesn-filters/filters.yaml`` in the input yaml
file, which will enable you to use all of these inbuilt filters.

Automatic filter dropping
--------------------------

The wavelength range covered by the model will depend on exactly which model you use. Filters will automatically be
dropped for individual SNe when they fall out of the rest-frame wavelength range covered based on their redshift. The
upper and lower cut off wavelengths for each filter are defined as the wavelength where the filter response first
drops below 1 per cent of the maximum value.
