import numpy as np
import pandas as pd

df = pd.read_csv('Paranal_ISAAC.Js-SVO.dat', names=['wave', 'trans'], sep='\s+')
# convert duplicated values to NaN
nan_values = np.array([np.nan if b == True else val for 
                       val, b in zip(df.wave.values, df.duplicated(subset=['wave']).values)])
df['wave'] = nan_values
df.interpolate(inplace=True)
# save output, removing the last row as the wavelength repeats
np.savetxt('Paranal_ISAAC.Js.dat', df.values[:-1], fmt='%.6f')
