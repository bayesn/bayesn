.. _filters:

Defining Filters
===================

BayeSN includes a selection of filters and standards for convenience. Alternatively, you can use your own through a
simple yaml file, allowing new or updated filters to be easily implemented independently of updates to the package.

Built-in filters
-----------------

BayeSN includes the following filters, grouped together by instrument, survey or system, including the name that should
be used when referring to the filter within BayeSN (e.g. `g_PS1` for Pan-STARRS 1 *g*-band). Note that if these names
do not match those contained within your data files (perhaps they contain single letter SNANA filter names), you can
provide a map to match up the names in the data files with the BayeSN names, as discussed in :ref:`running_bayesn`.

- Standard

    - Strinzinger *UBVRI* filters

        - Names: `U`, `B`, `V`, `R`, `I`
        - Source: Strinzinger+11, 2005PASP..117..810S
    - 2MASS Peters Automated Infrared Imaging Telescope *JHK* filters

        - Names: `J`, `H`, `K`
        - Source: Cohen03, 2003AJ....126.1090C
    - Persson *YJHK* filters

        - Names: `Y_P`, `J_P`, `H_P`, `K_P`
        - Source: 1998AJ....116.2475P
- SDSS

    - 2.5m Sloan Digital Sky Survey *ugriz* filters at airmass 1.3

        - Names: `u_SDSS`, `g_SDSS`, `r_SDSS`, `i_SDSS`, `z_SDSS`
        - Source: Doi+2010, 2010AJ....139.1628D
- CSP

    - Carnegie Supernova Project *BVgri* Swope filters

        - Names: `B_CSP`, `V_CSP`, `V_CSP_3009`, `V_CSP_3014`, `g_CSP`, `r_CSP`, `i_CSP`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters
    - Carnegie Supernova Project II *BVgri* Swope filters

        - Names: `B_CSP2`, `V_CSP2`, `g_CSP2`, `r_CSP2`, `i_CSP2`
        - Source: https://csp.obs.carnegiescience.edu/data/filters
    - Swope RetroCam *YJH* filters

        - Names: `Y_RC`, `J_RC1`, `J_RC2`, `H_RC`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters
    - Dupont WIRC *YJH* filters

        - Names: `Y_WIRC`, `J_WIRC`, `H_WIRC`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters
    - Dupont RetroCam *YJH* filters

        - Names: `Y_RCDP`, `J_RCDP`, `H_RCDP`
        - Source: https://csp.obs.carnegiescience.edu/data/filters
- DECam

    - Dark Energy Camera at Cerro Tololo Inter-American Observatory *griz* filters

        - Names: `g_DES`, `r_DES`, `i_DES`, `z_DES`
        - Source: https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera

- HST

    - Hubble Space Telescope WFC3IR/UVIS2 filters

        - Names: `F105W`, `F125W`, `F140W`, `F160W`, `F225W`, `F275W`, `F300X`, `F336W`, `F390W`, `F438W`, `F475W`, `F555W`, `F625W`, `F814W`
        - Source: https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs
- LSST

    - Legacy Survey of Space and Time at Vera Rubin Observatory *ugrizy* filters

        - Names: `u_LSST`, `g_LSST`, `r_LSST`, `i_LSST`, `z_LSST`, `y_LSST`
        - Source: https://github.com/lsst/throughputs
- PS1

    - PanSTARRS 1 *grizyw* and open filters

        - Names: `g_PS1`, `r_PS1`, `i_PS1`, `z_PS1`, `y_PS1`, `w_PS1`, `open_PS1`
        - Source: Tonry+12, 2012ApJ...750...99T
- SWIFT UVOT

    - SWIFT UVOT *UBV* and *UVW1/UVW2/UVM2* filters

        - Names: `U_SWIFT`, `B_SWIFT`, `V_SWIFT`, `UVW1`, `UVW2`, `UVM2`
        - Source: Poole+08, 2008MNRAS.383..627P
- USNO

    - United States Naval Observatory 40-inch telescope *u'g'r'i'z'* filters

        - Names: `u_prime`, `g_prime`, `r_prime`, `i_prime`, `z_prime`
        - Source: Fukugita+96, 1996AJ....111.1748F; Smith+02, 2002AJ....123.2121S
- ZTF

    - Zwicky Transient Facility *gri* filters

        - Names: `p48g`, `p48r`, `p48i`
        - Source: Bellm+19, 2019PASP..131a8002B
- ANDICAM

    - ANDICAM at Cerro Tololo Inter-American Observatory *YJHK* filters

        - Names: `Y_AND`, `J_AND`, `H_AND`, `K_AND`
        - Source:

- UKIRT

    - WFCAM *zYJHK* filters

        - Names: `z_WFCAM`, `Y_WFCAM`, `J_WFCAM`, `H_WFCAM`, `K_WFCAM`
        - Source: Hewett+06, 2009MNRAS.394..675H

- ATLAS

    - ATLAS *co* filters

        - Names: `c_ATLAS`, `o_ATLAS`
        - Source: Tonry+18, 2018PASP..130f4505T


Specifying custom filters
---------------------------

One of the arguments for the ``input.yaml`` file outlined in :ref:`running_bayesn`, ``filters``, is used to specify a
path to a separate yaml file which details any custom filters and standards you wish to add beyond those already
included. Any custom filters or standards will get included along with those built-in, so you'll be able to mix and
match between in-built filters and custom ones. Note that if you give a custom filter/standard the same name as a
built-in filter/standard, your custom one will be used instead of the built-in one.

The filter yaml to specify custom filters and standards should have the following structure:

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
        lam_unit: nm
      test_band_2:
        magsys: vega
        magzero: 0
        path: test_band_2_response.dat

These arguments are described as follows:

- ``standards_root``: A directory which all paths in ``standards`` are defined relative to. For example, if the standard spectrum for Vega is located at ``\data\standards\VEGA_STANDARD.fits`` and BD17 is at ``\data\filters\BD17_STANDARD.fits``, you can just set ``standards_root: \data\standards`` and use ``path: VEGA_STANDARD.fits`` within the key for Vega and similar for BD17. Alternatively, if you use a relative path this will be treated as being relative to the location of the filters yaml file. You can also use an environment variable here as part of the path e.g. $SNDATA_ROOT. This is an optional argument present for convenience, if not specified it is assumed that the paths for each band are all full paths rather than paths relative to ``standards_root``.
- ``standards``: Keys in here define all of the standards you wish to use. For each standard, the key is the name (this can be any string of your choosing), and each must have a ``path`` specifying the location of the reference spectrum for each standard - this can be either a FITS file with named columns for WAVELENGTH and FLUX, or a text file with columns for each.
- ``filters_root``: This specifies a directory which all paths in ``filters`` are defined relative to, behaving exactly as ``standards_root`` does for ``standards``. Again, if you use a relative path this will be treated as being relative to the location of the filters yaml file.
- ``filters``: Keys in here define all of the filters you wish you use. For each filter, the key is the name (again, this can be any string of your choosing). Each filter must have a ``magsys`` key which either corresponds to one of the built-in standards ('vega', 'bd17' or 'ab') or a custom standard name defined in ``standards``, defining the magnitude system for each band. Each filter must also have a ``magzero`` key, specifying the magnitude offset for the filter, and a ``path`` specifying the location of the filter response for each filter. Optionally, you can provide a ``lam_unit`` key - by default, BayeSN expects you to use filter responses with wavelength in Angstroms, but you can specify either 'nm' or 'micron' if your filter responses use nanometres or micrometres respectively and the units will be converted into Angstroms under-the-hood.

Automatic filter dropping
--------------------------

The wavelength range covered by the model will depend on exactly which model you use. Filters will automatically be
dropped for individual SNe when they fall out of the rest-frame wavelength range covered based on their redshift. The
upper and lower cut off wavelengths for each filter are defined as the wavelength where the filter response first
drops below 1 per cent of the maximum value.
