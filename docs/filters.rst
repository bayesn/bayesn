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

- Apache Point Observatory (APO)

    - 2.5m Sloan Digital Sky Survey (SDSS) *ugriz* filters at airmass 1.3

        - Names: `u_SDSS`, `g_SDSS`, `r_SDSS`, `i_SDSS`, `z_SDSS`
        - Source: Doi+2010, 2010AJ....139.1628D

    - 0.6m Monitor Telescope *ugriz* filters at airmass 1.2

        - Names: `u_SDSS_MT`, `g_SDSS_MT`, `r_SDSS_MT`, `i_SDSS_MT`, `z_SDSS_MT`
        - Source: Fukugita+1996, 1996AJ....111.1748F, as tabulated in SNooPy

- Asteroid Terrestrial-Impact Last Alert System (ATLAS)

    - ATLAS *co* filters

        - Names: `c_ATLAS`, `o_ATLAS`
        - Source: Tonry+18, 2018PASP..130f4505T

- Centro Astronómico Hispano en Andalucía (CAHA)

    - O2K *JHK* filters

        - Names: `J_O2K`, `H_O2K`, `K_O2K`
        - Source: Tomas Muller-Bravo

- Canada France Hawaii Telescope (CFHT)

    - Megacam *griz* filters

        - Names: `g_Megacam`, `r_Megacam`, `i_Megacam`, `z_Megacam`
        - Source: SNooPy

- CMO

    - 2.5m *JHK* filters

        - Names: `J_CMO`, `H_CMO`, `K_CMO`
        - Source: Tomas Muller-Bravo

- Cerro Tololo Inter-American Observatory (CTIO)

    - 0.9m *UBVRI* filters

        - Names: `U_CTIO`, `B_CTIO`, `V_CTIO`, `R_CTIO`, `I_CTIO`
        - Source: SNooPy

    - 4m *RI* filters for ESSENCE

        - Names: `R_ESSENCE`, `I_ESSENCE`
        - Source: SNooPy

    - ANDICAM *YJHK* filters

        - Names: `Y_AND`, `J_AND`, `H_AND`, `K_AND`
        - Source: https://noirlab.edu/science/programs/ctio/filters/andicam

    - Dark Energy Camera (DECam) *griz* filters

        - Names: `g_DES`, `r_DES`, `i_DES`, `z_DES`
        - Source: https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera

    - PROMPT *BVgri* filters on telescopes 1/3/5

        - Names: `B_PROMPT_1`, `B_PROMPT_35`, `V_PROMPT_135`, `g_PROMPT_3`, `r_PROMPT_5`, `i_PROMPT_5`
        - Source: SNooPy

- Fred Lawrence Whipple Observatory (FLWO)

    - 1.2m + 4Shooter *UBVRI* filters

        - Names: `U_4sh`, `B_4sh`, `V_4sh`, `R_4sh`, `I_4sh`
        - Source: SNooPy

    - 1.2m + Keplercam *BV* filters from Oct 2010 (CfA period 2) and June 2011 (Cfa3 + CfA4 period 1), *u'r'i'* filters

        - Names: `B_K_201010`, `B_K_201106`, `V_K_201010`, `V_K_201106`, `u_prime_k`, `r_prime_k`, `i_prime_k`
        - Source: SNooPy

- GALEX

    - Imaging *NUV* and *FUV* filters

        - Names: `NUV_GALEX`, `FUV_GALEX`
        - Source: SNooPy

- Gemini North

    - NIRI *YJHKKsK'L'M'* filters and  *H-K notch* filter (all scanned warm), the estimated *Y* filter transmission at 65 K, and that convolved with the (warm) transmission profile of the PK50 filter.

        - Names: `Y_NIRI`, `J_NIRI`, `H_NIRI`, `K_NIRI`, `Ks_NIRI`, `K_prime_NIRI`, `L_prime_NIRI`, `M_prime_NIRI`, `HK_notch_NIRI`, `Y_NIRI_COLD`, `Y_NIRI_COLD_PK50`
        - Source: https://www.gemini.edu/instrumentation/niri/components#Filters

- HCT

    - TIRSPEC *JHKs* filters

        - Names: `J_TIRSPEC`, `H_TIRSPEC`, `Ks_TIRSPEC`
        - Source: SVO

- HST

    - Hubble Space Telescope WFC3IR/UVIS2 filters

        - Names: `F105W`, `F125W`, `F140W`, `F160W`, `F225W`, `F275W`, `F300X`, `F336W`, `F390W`, `F438W`, `F475W`, `F555W`, `F625W`, `F814W`
        - Source: https://www.stsci.edu/hst/instrumentation/wfc3/performance/throughputs

- IRSF

    - SIRIUS *JHK* filters

        - Names: `Y_SIRIUS`, `J_SIRIUS`, `H_SIRIUS`
        - Source: https://www-ir.u.phys.nagoya-u.ac.jp/~irsf/sirius/tech/index.html

- IRTF

    - NSFCam *JHKs* filters

        - Names: `J_NSFCam`, `H_NSFCam`, `Ks_NSFCam`
        - Source: SVO

- KPNO

    - WHIRC *JHKs* filters

        - Names: `J_WHIRC`, `H_WHIRC`, `Ks_WHIRC`
        - Source: SNooPy

- La Silla

    - NTT *JHKs* filters

        - Names: `J_NTT`, `H_NTT`, `Ks_NTT`
        - Source: SNooPy

- Las Campanas Observatory (LCO)

    - Baade FourStar *JHK* broad filters and *J1* medium filter

        - Names: `J_FS`, `H_FS`, `K_FS`, `J1_FS`
        - Source: SNooPy, `K_FS` from https://instrumentation.obs.carnegiescience.edu/FourStar/OPTICS/filters.html. `K_FS` does not include atmospheric, telescopic, and QE transmittance.

    - Baade PANIC *YJHKs* filters

        - Names: `Y_PANIC`, `J_PANIC`, `H_PANIC`, `Ks_PANIC`
        - Source: SNooPy

    - Carnegie Supernova Project *BVgri* Swope filters

        - Names: `B_CSP`, `V_CSP`, `V_CSP_3009`, `V_CSP_3014`, `g_CSP`, `r_CSP`, `i_CSP`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Carnegie Supernova Project II *BVgri* Swope filters

        - Names: `B_CSP2`, `V_CSP2`, `g_CSP2`, `r_CSP2`, `i_CSP2`
        - Source: https://csp.obs.carnegiescience.edu/data/filters

    - Dupont WIRC *YJH* filters

        - Names: `Y_WIRC`, `J_WIRC`, `H_WIRC`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Dupont RetroCam *YJH* filters

        - Names: `Y_RCDP`, `J_RCDP`, `H_RCDP`
        - Source: https://csp.obs.carnegiescience.edu/data/filters

    - Swope RetroCam *YJH* filters

        - Names: `Y_RC`, `J_RC1`, `J_RC2`, `H_RC`
        - Source: Krisciunas+2017, 2017AJ....154..211K, https://csp.obs.carnegiescience.edu/data/filters

    - Abandoned *Ks* Swope filter
        - Names: `Ks_CSP`
        - Source: Contreras+2010, 2010AJ....139..519C

    - Different Persson *YJHK* filters?

        - Names: `Y_P1`, `J_P1`, `H_P1`, `K_P1`
        - Source: SNooPy

- Lick

    - KAIT *UBVRI* filters

      - Names: `U_KAIT`, `B_KAIT`, `V_KAIT`, `R_KAIT`, `I_KAIT`
      - Source: SNooPy

- Lick

    - KAIT *UBVRI* filters

      - Names: `U_KAIT`, `B_KAIT`, `V_KAIT`, `R_KAIT`, `I_KAIT`
      - Source: SNooPy

- Liverpool

    - IOO *BVgriz* filters

        - Names: `B_IOO`, `V_IOO`, `g_IOO`, `r_IOO`, `i_IOO`, `z_IOO`,
        - Source: SVO

- LSST

    - Legacy Survey of Space and Time at Vera Rubin Observatory *ugrizy* filters

        - Names: `u_LSST`, `g_LSST`, `r_LSST`, `i_LSST`, `z_LSST`, `y_LSST`
        - Source: https://github.com/lsst/throughputs

- Nordic Optical Telescope (NOT)

    - ALFOSC *UBVRIgriz* filters and *UBVRI* natural filters

        - Names:  `U_ALFOSC`, `B_ALFOSC`, `V_ALFOSC`, `R_ALFOSC`, `I_ALFOSC`, `g_ALFOSC`, `r_ALFOSC`, `i_ALFOSC`, `z_ALFOSC`, `U_ALFOSC_nat`, `B_ALFOSC_nat`, `V_ALFOSC_nat`, `R_ALFOSC_nat`, `I_ALFOSC_nat`
        - Source: https://www.not.iac.es/instruments/filters/filters.php

    - MOSCA *UBVRI* effective filters

        - Names: `U_MOSCA`, `B_MOSCA`, `V_MOSCA`, `R_MOSCA`, `I_MOSCA`
        - Source: Tomas Muller-Bravo

    - NOTCam *ZYJHKK'Ks* filters and effective filters (total throughput) for *JHKKs*

        - Names: `Z_NOT`, `Y_NOT`, `J_NOT`, `H_NOT`, `K_NOT`, `K_prime_NOT`, `Ks_NOT`, `J_NOT_eff`, `H_NOT_eff`, `K_NOT_eff`, `Ks_NOT_eff`
        - Source: https://www.not.iac.es/instruments/notcam/filters/, effective filters from Tomas Muller-Bravo

- OAM

    - TJO MEIA *UBVIc* filters

        - Names: `U_TJO`, `B_TJO`, `V_TJO`, `Ic_TJO`
        - Source: Tomas Muller-Bravo

- Palomar

    - P48

        - CFH12K *gri* filters

            - Names: `g_P48`, `r_P48`, `i_P48`
            - Source: Tomas Muller-Bravo

        - Zwicky Transient Facility (ZTF) *gri* filters

            - Names: `g_ZTF`, `r_ZTF`, `i_ZTF`
            - Source: Bellm+19, 2019PASP..131a8002B

    - P60 SED Machine (SEDM) *ugri* filters

        - Names: `u_SEDM`, `g_SEDM`, `r_SEDM`, `i_SEDM`
        - Source: Tomas Muller-Bravo, from Uli

- Panoramic Survey Telescope and Rapid Response System (PanSTARRS)

    - PanSTARRS 1 (PS1) *grizyw* and open filters

        - Names: `g_PS1`, `r_PS1`, `i_PS1`, `z_PS1`, `y_PS1`, `w_PS1`, `open_PS1`
        - Source: Tonry+12, 2012ApJ...750...99T

- Paranal

    - UT4 High Acuity Wide field K-band Imager (HAWK-I) *YJH* filters and two *Ks* filters, one used from Aug 2007 until Dec 3, 2007 and the other used starting Jan 2008

        - Names: `Y_HAWKI`, `J_HAWKI`, `H_HAWKI`, `Ks_HAWKI_1`, `Ks_HAWKI_2`
        - Source: https://www.eso.org/sci/facilities/paranal/instruments/hawki/inst.html

    - UT3 Infrared Spectrometer And Array Camera (ISAAC) *JHKs* filters

        - Names: `J_ISAAC`, `H_ISAAC`, `Ks_ISAAC`
        - Source: SVO

    - VISTA InfraRed Camera (VIRCAM) *ZYJHKs* filters

        - Names: `Z_VISTA`, `Y_VISTA`, `J_VISTA`, `H_VISTA`, `Ks_VISTA`
        - Source: https://www.eso.org/sci/facilities/paranal/decommissioned/vircam/inst.html

- Spitzer

    - IRAC *3.6/4.5/5.8/8.0* filters

        - Names: `S36`, `S45`, `S58`, `S80`
        - Source: Tomas Muller-Bravo

- SPM

    - RATIR *rizYJH* filters in the AB mag system and *YJH* in the Vega mag system

        - Names: `r_SPM`, `i_SPM`, `z_SPM`, `Y_SPM_AB`, `J_SPM_AB`, `H_SPM_AB`, `Y_SPM`, `J_SPM`, `H_SPM`
        - Source: Tomas Muller-Bravo

- SWIFT UVOT

    - SWIFT UVOT *UBV* and *UVW1/UVW2/UVM2* filters

        - Names: `U_SWIFT`, `B_SWIFT`, `V_SWIFT`, `UVW1`, `UVW2`, `UVM2`
        - Source: Poole+08, 2008MNRAS.383..627P

- Telescopio Nazionale Galileo (TNG)

    - Near Infrared Camera Spectrometer (NICS) *JHKK'Ks* filters

        - Names: `J_TNG`, `H_TNG`, `K_TNG`, `K_prime_TNG`, `Ks_TNG`
        - Source: https://www.tng.iac.es/instruments/nics/imaging.html#filters

- United States Naval Observatory (USNO)

    - 40-inch telescope *u'g'r'i'z'* filters

        - Names: `u_prime`, `g_prime`, `r_prime`, `i_prime`, `z_prime`
        - Source: Fukugita+96, 1996AJ....111.1748F; Smith+02, 2002AJ....123.2121S

- UKIRT

    - WFCAM *zYJHK* filters

        - Names: `z_WFCAM`, `Y_WFCAM`, `J_WFCAM`, `H_WFCAM`, `K_WFCAM`
        - Source: Hewett+06, 2009MNRAS.394..675H


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
