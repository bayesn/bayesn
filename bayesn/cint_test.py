import numpy as np
import matplotlib.pyplot as plt

from bayesn import SEDmodel

t = np.array([0, 10])
N = 10
bands = ['r_PS1']
cint = np.linspace(-0.3, 0.3, N)
theta = np.linspace(-5, 5, N)

model = SEDmodel(load_model='/Users/matt/Documents/bayesn-input/cint_train/T21_train_v2_test/bayesn.yaml')

print(model.W0)
print(model.W1)

lc = model.simulate_light_curve(t, 1, bands, theta=0, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
print(lc.shape)
m150 = lc[0, 0] - lc[1, 0]

t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve(t2, N, bands, theta=theta, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
for i in range(N):
    plt.plot(t2, lc[:, i], label=theta[i])
# m15 = lc[0, :] - lc[1, :]
# plt.plot(theta, m15)
plt.gca().invert_yaxis()
plt.show()

# print(lc.flatten())
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))

# lc = model.simulate_light_curve(t, N, bands, theta=0, AV=0, mu=0, del_M=0, eps=None, mag=True)[0]
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))

raise ValueError('Nope')

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

# lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train_fixeps/chains.pkl',
#                                 theta=0, AV=0, mu=0, del_M=0, eps=None, cint=0, mag=True)[0]
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
