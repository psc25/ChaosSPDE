import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "stochheat"
name = "stochheat_USV_RN"
np.random.seed(0)
b1 = time.time()

I = 1
J = 5
K = 1

m = 10

M1 = 200
M2 = 20
M3 = 1000

T = 0.25
dt = T/M2
tt = np.linspace(0.0, T, M2)
tt1 = np.reshape(tt, [1, 1, M2, 1])

uu = np.loadtxt(path + "/" + path + "_" + str(m) + "_u.csv")
uu1 = np.reshape(uu, [1, 1, 1, M3, m])

W = np.expand_dims(np.loadtxt(path + "/" + path + "_" + str(m) + "_W.csv"), 0)
W_ij = np.expand_dims(np.loadtxt(path + "/" + path + "_" + str(m) + "_W_ij.csv"), 0)
xi_alpha = np.loadtxt(path + "/" + path + "_" + str(m) + "_xi_alpha.csv")
nr_alpha = xi_alpha.shape[0]

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
A0 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 2.0)
A1 = np.random.normal(size = (nr_alpha, N, 1, 1, m), scale = 0.2/np.sqrt(m))
B = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.5)

# Compute true solution
sigma = 6.0
sigma2 = 2.0*tt1[0] + sigma**2
phiX0 = 10.0*np.power(sigma**2/sigma2, m/2.0)*np.exp(-np.sum(np.square(uu1[0]), axis = -1)/(2.0*sigma2))
X_true = phiX0 + np.expand_dims(W[0], axis = -1)

# Compute random neurons
hid = A0*tt1 + np.sum(A1*uu1, axis = -1) - B
rndn = act(hid)
rndn_lapl = act_der2(hid)*np.sum(np.square(A1), axis = -1)

X_0 = X_true[ind_train, 0:1]
drift = np.cumsum(rndn_lapl, axis = 2)*dt
diffu = np.expand_dims(W_ij[0, ind_train], -1)

# Train linear readout
begin = time.time()
A_lsq = np.transpose(np.reshape(np.expand_dims(rndn - drift, 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4]), [nr_alpha*N, -1]))
b_lsq = np.reshape(X_0 + diffu, -1)
Y1 = scla.lstsq(A_lsq, b_lsq, lapack_driver = 'gelsy')[0]
Y = np.reshape(Y1, [nr_alpha, N, 1, 1])

# Compute random neural network approximation
RN = np.sum(Y*rndn, axis = 1)
X_IJK = np.sum(np.expand_dims(RN, axis = 1)*np.expand_dims(xi_alpha, axis = [2, 3]), axis = 0)
end = time.time()

# Compute loss and print
res_loss = np.zeros(2)
res_loss[0] = np.sqrt(np.mean(np.square(X_true[ind_train] - X_IJK[ind_train])))
res_loss[1] = np.sqrt(np.mean(np.square(X_true[ind_test] - X_IJK[ind_test])))
print("Time {}s, loss in-s {:g}, loss out-of-s {:g}".format(round(end-begin, 1), res_loss[0], res_loss[1]))

# Plot
ind_plot = M1train
uuplot1 = np.linspace(-2.0, 2.0, M3)
uuplot = np.zeros([1, 1, 1, M3, m])
uuplot[0, 0, 0, :, 0] = uuplot1

hid = A0*tt1 + np.sum(A1*uuplot, axis = -1) - B
RN_plot = np.sum(Y*act(hid), axis = 1)
X_IJK_plot = np.sum(np.expand_dims(RN_plot, axis = 1)*np.expand_dims(xi_alpha[:, ind_plot:(ind_plot+1)], axis = [2, 3]), axis = 0)

sigma2 = 2.0*tt1[0] + sigma**2
phiX0_plot = 10.0*np.power(sigma**2/sigma2, 0.5*m)*np.exp(-np.sum(np.square(uuplot[0]), axis = -1)/(2.0*sigma2))
X_true_plot = phiX0_plot + np.expand_dims(W[0, ind_plot:(ind_plot+1)], axis = -1)

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

np.savetxt(path + "/" + name + "_" + str(m) + "_X_true.csv", np.reshape(X_true_plot, [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(m) + "_X_IJK.csv", np.reshape(X_IJK_plot, [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(m) + "_err.csv", res_loss)
np.savetxt(path + "/" + name + "_" + str(m) + "_tms.csv", [tms])