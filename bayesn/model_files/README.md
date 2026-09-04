# BayeSN Model Files

This directory contains the raw files which describe several trained BayeSN models.

## Provided Models

Details of currently provided models are given in the table below.

Model | Training Set | Wavelength Range (Å) | Phase Range (days) | Description | Model Reference | Training Set Reference
--- | --- | --- | --- | --- | --- | ---
`G26` | Kenworthy+21 with Dovekie calibration updates | [2800, 10800] | [-10, 40] | Grayling+26 BayeSN model. Covers rest wavelength range of 2800-10800Å (_ugriz_). Intended for cosmology; jointly fits filter wavelength and zero-point cross-calibration shifts alongside the SED, using Dovekie as an informative prior. Population _R<sub>V</sub>_ distribution. Trained on 1024 SNe Ia from Foundation, CfA3, CfA4, CSP, SDSS, PS1, DES and SNLS. | Grayling et al. (2026; [arXiv:2606.19429](https://arxiv.org/abs/2606.19429)) | Kenworthy et al. (2021; [arXiv:2104.07795](https://arxiv.org/abs/2104.07795)) with Dovekie updates (Popovic et al. 2025; [arXiv:2506.05471](https://arxiv.org/abs/2506.05471))
`G25` | Avelino+19 + Foundation DR1 + CSP-I | [2800, 18500] | [-10, 85] | Grayling+26 phase-extended optical+NIR BayeSN model (BayeSN-TD). Covers rest wavelength range of 2800-18500Å (_UBgVrizYJH_). Phase coverage extended to +85 days, motivated by fitting late-time observations of strongly-lensed SNe Ia. Trained on 278 SNe Ia combining Avelino+19, Foundation DR1, and CSP-I. | Grayling et al. (2026; [MNRAS 548, stag340](https://arxiv.org/abs/2510.11719)) | Avelino et al. (2019; [arXiv:1902.03261](https://arxiv.org/abs/1902.03261)), Foley et al. (2018; [arXiv:1711.02474](https://arxiv.org/abs/1711.02474)), Jones et al. (2019; [arXiv:1811.09286](https://arxiv.org/abs/1811.09286))
`W22` | `M20_training_set` + `T21_training_set` | [3000, 18500] | [-10, 40] | Ward+22 BayeSN model. Covers rest wavelength range of 3000-18500Å (_BgVrizYJH_). No treatment of host mass effects. Global _R<sub>V</sub>_ assumed. Trained on Avelino+19 compilation plus Foundation DR1 (Foley+18, Jones+19). | Ward et al. (2023; [ApJ 956, 111](https://ui.adsabs.harvard.edu/abs/2023ApJ...956..111W/abstract)) | Avelino et al. (2019; [arXiv:1902.03261](https://arxiv.org/abs/1902.03261), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...887..106A/abstract)), Foley et al. (2018; [arXiv:1711.02474](https://arxiv.org/abs/1711.02474), [ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F/abstract)), Jones et al. (2019; [arXiv:1811.09286](https://arxiv.org/abs/1811.09286), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J/abstract), [github](https://github.com/djones1040/Foundation_DR1))
`T21` | `T21_training_set` | [3500, 9500] | [-10, 40] | Thorp+21 BayeSN _No-Split_ model. Covers rest wavelength range of 3500-9500Å (_griz_). No treatment of host mass effects. Global _R<sub>V</sub>_ assumed. Trained on Foundation DR1 (Foley+18, Jones+19). | Thorp et al. (2021; [arXiv:2102.05678](https://arxiv.org/abs/2102.05678)) | Foley et al. (2018; [arXiv:1711.02474](https://arxiv.org/abs/1711.02474), [ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F/abstract)), Jones et al. (2019; [arXiv:1811.09286](https://arxiv.org/abs/1811.09286), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J/abstract), [github](https://github.com/djones1040/Foundation_DR1))
`T21_partial-split` | `T21_training_set` | [3500, 9500] | [-10, 40] | Thorp+21 BayeSN _Partial-Split_ model. Covers rest wavelength range of 3500-9500Å (_griz_). Separate (_M<sub>0</sub>_, _σ<sub>0</sub>_, _R<sub>V</sub>_, _τ<sub>A</sub>_) for low- and high-mass hosts. Split at median host mass of Foundation sample (log<sub>10</sub>_M_=10.331). Trained on Foundation DR1 (Foley+18, Jones+19). | Thorp et al. (2021; [arXiv:2102.05678](https://arxiv.org/abs/2102.05678)) | Foley et al. (2018; [arXiv:1711.02474](https://arxiv.org/abs/1711.02474), [ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.475..193F/abstract)), Jones et al. (2019; [arXiv:1811.09286](https://arxiv.org/abs/1811.09286), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...881...19J/abstract), [github](https://github.com/djones1040/Foundation_DR1))
`M20` | `M20_training_set` | [3000, 18500] | [-10, 40] | Mandel+20 BayeSN model. Covers rest wavelength range of 3000-18500Å (_BVriYJH_). No treatment of host mass effects. Global _R<sub>V</sub>_ assumed. Trained on low-_z_ Avelino+19 compilation of CfA, CSP and others. | Mandel et al. (2022; [MNRAS 510, 3939-3966](https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.3939M/abstract)) | Avelino et al. (2019; [arXiv:1902.03261](https://arxiv.org/abs/1902.03261), [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...887..106A/abstract)), and references therein

## Using The Models

To load a model, the following lines of code would be needed:
```python
from BayeSNmodel import bayesn_model
b = bayesn_model.SEDmodel(model=Model)
```
where `Model` is the shorthand name from the first column of the table above. To load a training set, one would need to run:
```python
from BayeSNmodel import io
lcs = io.read_sn_sample_file(sampfile=Training_Set, metafile='lcs/meta/' + Training_Set + '_meta.txt')
```
where `Training_Set` is the shorthand name listed in the second column of the table above.

## The Files Which Specify a BayeSN Model

Generically, a BayeSN model is specified by six `.txt` files which specify the various components. These are:
 * `l_knots.txt`: A list of rest frame wavelength knot locations, in Angstroms.
 * `tau_knots.txt`: A list of rest frame phase knot locations, in days.
 * `W0.txt`: The _W<sub>0</sub>_ matrix describing the zeroth order warping of the Hsiao template which gives the mean intrinsic SED. This has a column for every phase knot, and a row for every wavelength know (plus one extra row at the top and bottom).
 * `W1.txt`: The _W<sub>1</sub>_ matrix describing the first functional principal component. This should be the same shape as `W0.txt`.
 * `L_Sigma_epsilon.txt`: The Cholesky factor of the residual covariance matrix.
 * `M0_sigma0_RV_tauA.txt`: Other global model parameters (intrinsic, _M<sub>0</sub>_, _σ<sub>0</sub>_; dust, _R<sub>V</sub>_, _τ<sub>A</sub>_). For _No-Split_ models, this will just be a single line with the 4 numbers. For _Partial-Split_ models, there will be two lines in the file, the first listing the high-mass values, the second listing the low-mass values. For a _Partial-Split_ model, there should be a fifth column listing the split mass (this should be duplicated in both rows).