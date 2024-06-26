import numpy as np
import matplotlib.pyplot as plt
import sncosmo
from bayesn import SEDmodel

model = SEDmodel(load_model='T21_model')

filt_map = {'g': 'g_PS1', 'r': 'r_PS1', 'i': 'i_PS1', 'z': 'z_PS1'}
samples, sn_props = model.fit_from_file(model.example_lc, filt_map=filt_map)

peak_mjd = samples['peak_MJD']
print(peak_mjd.mean(), peak_mjd.std())
