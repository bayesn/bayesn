Built-in BayeSN Models
========================

Summarised here are the different models currently built-in to this BayeSN code. Alternatively, you can train your own
models and use those within the code.

M20 model:

- Mandel et al., 2022 (MNRAS 510, 3, 3939-3966, https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.3939M/abstract)
- Trained on 86 SNe Ia compiled in Avelino+19 (as described in Mandel+22)
- Training data includes BVRIYJH only
- Wavelength range 3000 - 18500 A
- Not suitable for u/U band, even if included in wavelength range
- Milky Way extinction based on SF11

T21 model:

- Thorp et al., 2021 (MNRAS 508, 3, 4310-4331, https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.4310T/abstract)
- Trained on 157 SNe Ia from Foundation DR1 (Foley+18, Jones+19)
- Training data includes griz only
- Wavelength range 3500 - 9500 A
- Milky Way extinction based on SF11

W22 model:

- Ward et al., 2022 (ApJ 956, 2, 111, https://ui.adsabs.harvard.edu/abs/2023ApJ...956..111W/abstract)
- Trained on Foundation DR1 (Foley+18, Jones+19) and Avelino+19 compilation, combination of data sets for M20 and T21
- Training data includes BgVrizYJH
- Wavelength range 3000 - 18500 A
- Milky Way extinction based on SF11