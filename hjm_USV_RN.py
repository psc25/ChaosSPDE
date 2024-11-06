import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "hjm"
name = "hjm_USV_RN"
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

W_ij = np.expand_dims(np.loadtxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_W_ij.csv"), 0)
xi_alpha = np.reshape(np.loadtxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_xi_alpha.csv"), [-1, M1])
nr_alpha = xi_alpha.shape[0]

r = np.loadtxt(path + "/" + path + "_r.csv", dtype = np.float32)

# Training properties
N = 75
val_split = 0.2
M1train = int((1-val_split)*M1)
ind_train = np.arange(M1train)
ind_test = np.arange(M1train, M1)
act = np.tanh
act_der = lambda x: 1.0/np.square(np.cosh(x))
act_der2 = lambda x: -2.0*np.tanh(x)*act_der(x)

# Initializations
A0 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.8)
A1 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.5)
B = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.6)

# Compute true solution
X_true = np.expand_dims(r, -1)*np.exp(-kappa*uu1[0]) + mu/kappa*(1.0-np.exp(-kappa*uu1[0])) - sigma**2/(2*kappa**2)*np.square(1.0-np.exp(-kappa*uu1[0]))
X_true_0 = np.expand_dims(r, -1)
X_true_der = -kappa*np.expand_dims(r, -1)*np.exp(-kappa*uu1[0]) + mu*np.exp(-kappa*uu1[0]) - sigma**2/kappa*(1.0-np.exp(-kappa*uu1[0]))*np.exp(-kappa*uu1[0])

# Compute random neurons
hid = A0*tt1 + A1*uu1 - B
rndn = act(hid)
rndn_der = act_der(hid)*A1
rndn_der2 = act_der2(hid)*np.square(A1)

rndn_xi = np.expand_dims(rndn, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_der_xi = np.expand_dims(rndn_der, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_der2_xi = np.expand_dims(rndn_der2, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])

hid_0 = A0*tt1 - B
rndn_0 = act(hid_0)
rndn_der_0 = act_der(hid_0)*A1

rndn_0_xi = np.expand_dims(rndn_0, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_der_0_xi = np.expand_dims(rndn_der_0, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])

initl_0 = X_true_0[ind_train, 0:1]
drift_0 = np.cumsum(rndn_der_0_xi, axis = 3)*dt
diffu_0 = sigma*np.expand_dims(W_ij[0, ind_train], axis = -1)

initl_der = X_true_der[ind_train, 0:1]
drift1_der = np.cumsum(rndn_der2_xi, axis = 3)*dt
drift2_der = sigma**2*(2.0*np.exp(-2.0*kappa*uu1[0]) - np.exp(-kappa*uu1[0]))*tt1[0]
diffu_der = -kappa*np.exp(-kappa*uu1[0])*sigma*np.expand_dims(W_ij[0, ind_train], axis = -1)

# Train linear readout
begin = time.time()
A_lsq1 = np.concatenate([rndn_0_xi - drift_0, (rndn_der_xi - drift1_der)/np.sqrt(M3)], axis = -1)
A_lsq = np.transpose(np.reshape(A_lsq1, [nr_alpha*N, -1]))
b_lsq = np.reshape(np.concatenate([initl_0 + diffu_0, (initl_der + drift2_der + diffu_der)/np.sqrt(M3)], -1), -1)
Y1 = scla.lstsq(A_lsq, b_lsq, lapack_driver = 'gelsy')[0]
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
res_loss[0] = np.sqrt(np.mean(np.square(X_true_0[ind_train] - X_IJK_0[ind_train])) + np.mean(np.square(X_true_der[ind_train] - X_IJK_der[ind_train])))
res_loss[1] = np.sqrt(np.mean(np.square(X_true_0[ind_test] - X_IJK_0[ind_test])) + np.mean(np.square(X_true_der[ind_test] - X_IJK_der[ind_test])))
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