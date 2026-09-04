import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

#files = ["SNLS3_4shooter2_V.dat", "SNLS3_4shooter2_B.dat", "SNLS3_4shooter2_R.dat", "SNLS3_4shooter2_I.dat"]

files = ['cfa4_B_p2_modtran.dat', 'cfa4_V_p2_modtran.dat', 'cfa4_r_p2_modtran.dat', 'cfa4_i_p2_modtran.dat']

lazy = ["V", "B", "r", "i"]

#dic = {"U":3562.1, "B":4355.83, "V":5409.74, "r":6242.35, "i":7674.08} #KCAM vals
dic = {"U":3562.1, "B":4355.83, "V":5409.74, "r":6242.35, "i":7674.08} #allegedly the same as KCAM?

for n,filt in enumerate(files):
    df = pd.read_csv(filt, sep=r"\s+", names=["wav", "trans"])

    waveeff = (dic[lazy[n]])

    print(filt, waveeff)

    df['trans'] = df.trans*(waveeff/df.wav)**-1

    df.to_csv(filt+"_weighted", index=False, float_format='%g', sep=" ", header=False)
