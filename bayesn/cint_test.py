import numpy as np
import matplotlib.pyplot as plt
import os

from bayesn import SEDmodel

t = np.array([0, 10, 20])
N = 5
bands = ['g_PS1', 'r_PS1']
cint = np.linspace(-0.1, 0.1, N)
theta = np.linspace(-0.2, 0.2, N)

model_path = '/Users/matt/Documents/bayesn-input/cint_train/T21_sim_train_full_eps'
model_chains = os.path.join(model_path, 'chains.pkl')

model = SEDmodel(load_model=os.path.join(model_path, 'bayesn.yaml'))

lc = model.simulate_light_curve(t, 1, bands, theta=0, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
print(lc.shape)
m150 = lc[0, 0] - lc[1, 0]

t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve(t2, N, bands, theta=theta, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
for i in range(N):
    plt.plot(t2, lc[:len(t2), i], label=f'{theta[i]:.2f}')
# m15 = lc[0, :] - lc[1, :]
# plt.plot(theta, m15)
plt.legend()
plt.gca().invert_yaxis()
plt.show()

t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve(t, N, bands, theta=theta, AV=0, mu=0, del_M=0, eps=0, mag=True)[0]
m15 = lc[0, :] - lc[1, :]
plt.plot(theta, m15)
plt.show()

# print(lc.flatten())
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))

# lc = model.simulate_light_curve(t, N, bands, theta=0, AV=0, mu=0, del_M=0, eps=None, mag=True)[0]
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))

# lc = model.simulate_light_curve_cint(t, 1, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train_v2_full/chains.pkl',
#                                 theta=0, AV=0, mu=0, del_M=0, eps=0, cint=0, mag=True)[0]
# print(lc.shape)
# c = lc[0, :] - lc[3, :]
# print(np.mean(c), np.std(c))
# c0 = lc[0, :] - lc[3, :]
# m100 = lc[0, :] - lc[1, :]

# lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train/chains.pkl',
#                                 theta=0, AV=0, mu=0, del_M=0, eps=0, cint=None, mag=True)[0]
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))
#
# # lc = model.simulate_light_curve_cint(t, N, bands, '/Users/matt/Documents/bayesn-input/cint_train/T21_train_fixeps/chains.pkl',
# #                                 theta=0, AV=0, mu=0, del_M=0, eps=None, cint=0, mag=True)[0]
# c = lc[0, :] - lc[1, :]
# print(np.mean(c), np.std(c))

# Make plots------------------------------------------
t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve_cint(t2, N, bands, model_chains,
                                     theta=theta, AV=0, mu=0, del_M=0, eps=0, cint=0, mag=True)[0]
for i in range(N):
    plt.plot(t2, lc[:len(t2), i], label=rf'$\theta={theta[i]:.2f}$')
# m15 = lc[0, :] - lc[1, :]
# plt.plot(theta, m15)
plt.gca().invert_yaxis()
plt.legend()
plt.show()

t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve_cint(t2, N, bands, model_chains,
                                     theta=0, AV=0, mu=0, del_M=0, eps=0, cint=cint, mag=True)[0]
for i in range(N):
    plt.plot(t2, lc[:len(t2), i], label=rf'$c_\mathrm{{int}}={cint[i]:.2f}$')
# m15 = lc[0, :] - lc[1, :]
# plt.plot(theta, m15)
plt.gca().invert_yaxis()
plt.legend()
plt.show()

t2 = np.arange(-10, 40, 1)
lc = model.simulate_light_curve_cint(t2, N, bands, model_chains,
                                     theta=0, AV=0, mu=0, del_M=0, eps=0, cint=cint, mag=True)[0]
c = lc[:len(t2), :] - lc[len(t2):, :]
for i in range(N):
    plt.plot(t2, c[:, i], label=rf'$c_\mathrm{{int}}={cint[i]:.2f}$')
# m15 = lc[0, :] - lc[1, :]
# plt.plot(theta, m15)
plt.gca().invert_yaxis()
plt.legend()
plt.show()
