import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev

airmass = 1
filters_4sh = pd.read_csv("table5.dat", sep=r"\s+", header=None, usecols=[0, 1, 5], names=['bp', 'wl', 'trans'])
atmosphere = pd.read_csv("modtran_atm_stubbs_Jan21.dat", sep=r"\s+", header=None, names=['wl', 'trans', '?', 'trans_again?'])

bps_4sh = 'U'

tck = splrep(atmosphere['wl']*10, atmosphere['trans'])

for bp in bps_4sh:
    filt = filters_4sh[filters_4sh['bp'] == bp]
    filt.loc[:, 'wl'] *= 10
    # 4sh functions are energy transmissions, dividing by wl to get photon transmissions.
    filt.loc[:, 'trans'] /= filt['wl']
    # renormalizing
    filt.loc[:, 'trans'] /= max(filt['trans'])
    if bp != 'U':
        filt.loc[:, 'trans'] *= splev(filt['wl'].values, tck)**airmass
    filt.to_csv(f"4sh_{bp}_modtran.dat", index=False, columns=['wl', 'trans'], float_format='%g', header=False, sep=" ")

bps_KC = 'U'
filters_KC = pd.read_csv('keplercam.passbands.dat', header=None, comment='n', skiprows=2, delimiter='\s+', names=['wl', 'trans'])
# all in two columns with lines ("nm {bp} band") to separate things.
# Detecting new bandpass when wl decreases
dlam = np.array(filters_KC['wl'][1:]) - np.array(filters_KC['wl'][:-1])
bp_bounds = [0] + list(np.where(dlam < 0)[0]+1) + [len(filters_KC['wl'])]
for i, bp in enumerate(bps_KC):
    filt = filters_KC.loc[bp_bounds[i]:bp_bounds[i+1]-1]
    filt.loc[:, 'wl'] *= 10
    filt.loc[:, 'trans'] *= splev(filt['wl'].values, tck)**airmass
    filt.to_csv(f"KC_{bp}_modtran.dat", index=False, float_format='%g', header=False, sep=" ")
