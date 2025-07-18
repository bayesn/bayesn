import numpy as np
import matplotlib.pyplot as plt

from bayesn import SEDmodel

t = np.array([0])
N = 1000
bands = ['g_PS1', 'r_PS1', 'i_PS1', 'z_PS1']
cint = np.linspace(-0.3, 0.3, N)

model = SEDmodel(load_model='T21_model')

lc = model.simulate_light_curve(t, 1, bands, theta=0, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
c = lc[0, :] - lc[1, :]
print(np.mean(c), np.std(c))

lc = model.simulate_light_curve(t, N, bands, theta=0, AV=0, mu=0, del_M=0, eps=None, mag=True)[0]
c = lc[0, :] - lc[1, :]
print(np.mean(c), np.std(c))

lc = model.simulate_light_curve_cint(t, 1, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train/chains.pkl',
                                theta=0, AV=0, mu=0, del_M=0, eps=0, cint=0, mag=True)[0]
c = lc[0, :] - lc[1, :]
print(np.mean(c), np.std(c))
c0 = lc[0, :] - lc[1, :]
c1 = lc[1, :] - lc[2, :]
c2 = lc[0, :] - lc[2, :]
c3 = lc[0, :] - lc[3, :]

lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train/chains.pkl',
                                theta=0, AV=0, mu=0, del_M=0, eps=0, cint=None, mag=True)[0]
c = lc[0, :] - lc[1, :]
print(np.mean(c), np.std(c))

lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train/chains.pkl',
                                theta=0, AV=0, mu=0, del_M=0, eps=None, cint=0, mag=True)[0]
c = lc[0, :] - lc[1, :]
print(np.mean(c), np.std(c))

# Make plots------------------------------------------
N = 10
cint = np.linspace(-0.3, 0.3, N)

lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train/chains.pkl',
                                theta=0, AV=0, mu=0, del_M=0, eps=0, cint=cint, mag=True)[0]
c = lc[0, :] - lc[1, :]
delta_c = c - c0
plt.scatter(cint, delta_c)
plt.plot([-0.3, 0.3], [-0.3, 0.3], ls='--')
plt.xlabel(r'$c_{int}$ parameter')
plt.ylabel(r'$\Delta g-r_{peak}$')
plt.show()
c = lc[1, :] - lc[2, :]
delta_c = c - c1
plt.scatter(cint, delta_c)
plt.plot([-0.3, 0.3], [-0.3, 0.3], ls='--')
plt.show()
c = lc[0, :] - lc[2, :]
delta_c = c - c2
plt.scatter(cint, delta_c)
plt.plot([-0.3, 0.3], [-0.3, 0.3], ls='--')
plt.show()
c = lc[0, :] - lc[3, :]
delta_c = c - c3
plt.scatter(cint, delta_c)
plt.plot([-0.3, 0.3], [-0.3, 0.3], ls='--')
plt.show()
