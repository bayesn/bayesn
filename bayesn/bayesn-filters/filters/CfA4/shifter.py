import pandas as pd

filename = 'cfa4_i_p2_modtran.dat_weighted'
# #wave(A) trans

df = pd.read_csv(filename, sep=r'\s+', names=['#wave(A)', 'trans'])

shiftval = +20

df['#wave(A)'] += shiftval

df.to_csv(filename+f"+{shiftval}", float_format="%g", index=False, sep=' ')
