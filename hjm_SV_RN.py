import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "hjm"
name = "hjm_SV_RN"
np.random.seed(0)
b1 = time.time()

I = 1
J = 5
K = 3

M1 = 200
M2 = 20
M3 = 80

mu = 4.0
kappa = 0.9
sigma = 0.5

T = 1.0
dt = T/M2
tt = np.linspace(0.0, T, M2, dtype = np.float32)
tt1 = np.reshape(tt, [1, 1, M2, 1])

uu = np.loadtxt(path + "/" + path + "_u.csv")
uu1 = np.reshape(uu, [1, 1, 1, M3])

xi_alpha = np.reshape(np.loadtxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_xi_alpha.csv"), [-1, M1])
nr_alpha = xi_alpha.shape[0]

r = np.loadtxt(path + "/" + path + "_r.csv")

# Training properties
N = 75
val_split = 0.2
M1train = int((1-val_split)*M1)
ind_train = np.arange(M1train)
ind_test = np.arange(M1train, M1)
act = np.tanh
act_der = lambda x: 1.0/np.square(np.cosh(x))

# Initializations
A0 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.8)
A1 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.5)
B = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.6)

# Compute true solution
X_true = np.expand_dims(r, -1)*np.exp(-kappa*uu1[0]) + mu/kappa*(1.0-np.exp(-kappa*uu1[0])) - sigma**2/(2*kappa**2)*np.square(1.0-np.exp(-kappa*uu1[0]))
X_0 = np.expand_dims(r, -1)
X_der = -kappa*np.expand_dims(r, -1)*np.exp(-kappa*uu1[0]) + mu*np.exp(-kappa*uu1[0]) - sigma**2/kappa*(1.0-np.exp(-kappa*uu1[0]))*np.exp(-kappa*uu1[0])

# Train linear readout
begin = time.time()
hid = A0*tt1 + A1*uu1 - B
rndn = act(hid)
hid_0 = A0*tt1 - B
rndn_0 = act(hid_0)
rndn_0_xi_train = np.expand_dims(rndn_0, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_der = act_der(hid)*A1
rndn_der_xi_train = np.expand_dims(rndn_der, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_xi_train1 = np.transpose(np.reshape(np.concatenate([rndn_0_xi_train, rndn_der_xi_train/np.sqrt(M3)], axis = -1), [nr_alpha*N, -1]))
X_train1 = np.reshape(np.concatenate([X_0[ind_train], X_der[ind_train]/np.sqrt(M3)], axis = -1), -1)
Y1 = scla.lstsq(rndn_xi_train1, X_train1, lapack_driver = 'gelsy')[0]
Y = np.reshape(Y1, [nr_alpha, N, 1, 1])

# Compute random neural network approximation
RN = np.sum(Y*rndn_0, axis = 1)
X_IJK = np.sum(np.expand_dims(RN, axis = 1)*np.expand_dims(xi_alpha, axis = [2, 3]), axis = 0)
RN_0 = np.sum(Y*rndn_0, axis = 1)
RN_der = np.sum(Y*rndn_der, axis = 1)
X_IJK_0 = np.sum(np.expand_dims(RN_0, axis = 1)*np.expand_dims(xi_alpha, axis = [2, 3]), axis = 0)
X_IJK_der = np.sum(np.expand_dims(RN_der, axis = 1)*np.expand_dims(xi_alpha, axis = [2, 3]), axis = 0)
end = time.time()

# Compute loss and print
res_loss = np.zeros(2)
res_loss[0] = np.sqrt(np.mean(np.square(X_0[ind_train] - X_IJK_0[ind_train])) + np.mean(np.square(X_der[ind_train] - X_IJK_der[ind_train])))
res_loss[1] = np.sqrt(np.mean(np.square(X_0[ind_test] - X_IJK_0[ind_test])) + np.mean(np.square(X_der[ind_test] - X_IJK_der[ind_test])))
print("Time {}s, loss in-s {:g}, loss out-of-s {:g}".format(round(end-begin, 1), res_loss[0], res_loss[1]))

# Plot
ind_plot = M1train+1
uuplot1 = np.linspace(0.0, 2.0, M3, dtype = np.float32)
uuplot = np.reshape(uuplot1, [1, 1, 1, M3])
hid = A0*tt1 + A1*uuplot - B
RN_plot = np.sum(Y*act(hid), axis = 1)
X_IJK_plot = np.sum(np.expand_dims(RN_plot, axis = 1)*np.expand_dims(xi_alpha[:, ind_plot:(ind_plot+1)], axis = [2, 3]), axis = 0)
X_true_plot = np.expand_dims(r[ind_plot:(ind_plot+1)], -1)*np.exp(-kappa*uuplot[0]) + mu/kappa*(1.0-np.exp(-kappa*uuplot[0])) - sigma**2/(2*kappa**2)*np.square(1.0-np.exp(-kappa*uuplot[0]))

u2, t2 = np.meshgrid(uuplot1, tt)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(u2, t2, X_true_plot[0])
ax.plot_surface(u2, t2, X_IJK_plot[0])
ax.set_xlabel("$u_1$")
ax.set_ylabel("$t$")
ax.set_zlabel("$X_t(\omega)(u_1,0,...,0)$")
ax.view_init(20)
plt.show()

e1 = time.time()
tms = e1 - b1

np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_X_true.csv", X_true_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_X_IJK.csv", X_IJK_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_err.csv", res_loss)
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_tms.csv", [tms])