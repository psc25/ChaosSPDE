import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "zakai"
name = "zakai_SV_RN"
np.random.seed(0)
b1 = time.time()

m = 2

I = 2*m
J = 4
K = 1

M1 = 300
M2 = 20
M3 = 150

T = 0.5
dt = T/M2
tt = np.linspace(0.0, T, M2)
tt1 = np.reshape(tt, [1, 1, M2, 1])

uu = np.loadtxt(path + "/" + path + "_u.csv")
uu1 = np.reshape(uu, [1, 1, 1, M3, m])

W = np.reshape(np.loadtxt(path + "/" + path + "_W.csv"), [I, M1, M2])
xi_alpha = np.loadtxt(path + "/" + path + "_" + str(J) + "_xi_alpha.csv")
nr_alpha = xi_alpha.shape[0]

X = np.reshape(np.loadtxt(path + "/" + path + "_X.csv"), [M1, M2, M3])
X_plot = np.reshape(np.loadtxt(path + "/" + path + "_X_plot.csv"), [1, M2, M3])

# Training properties
N = 75
val_split = 0.2
M1train = int((1-val_split)*M1)
ind_train = np.arange(M1train)
ind_test = np.arange(M1train, M1)
act = np.tanh

# Initializations
A0 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 1.0)
A1 = np.random.normal(size = (nr_alpha, N, 1, 1, m), scale = 0.8/np.sqrt(m))
B = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.5)

# Train linear readout
begin = time.time()
hid = A0*tt1 + np.sum(A1*uu1, axis = -1) - B
rndn = act(hid)
rndn_xi_train = np.expand_dims(rndn, axis = 2)*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4])
rndn_xi_train1 = np.transpose(np.reshape(rndn_xi_train, [nr_alpha*N, -1]))
X_train1 = np.reshape(X[ind_train], -1)
Y1 = scla.lstsq(rndn_xi_train1, X_train1, lapack_driver = 'gelsy')[0]
Y = np.reshape(Y1, [nr_alpha, N, 1, 1])

# Compute random neural network approximation
RN = np.sum(Y*rndn, axis = 1)
X_IJK = np.sum(np.expand_dims(RN, axis = 1)*np.expand_dims(xi_alpha, axis = [2, 3]), axis = 0)
end = time.time()

# Compute loss and print
res_loss = np.zeros(2)
res_loss[0] = np.sqrt(np.mean(np.square(X[ind_train] - X_IJK[ind_train])))
res_loss[1] = np.sqrt(np.mean(np.square(X[ind_test] - X_IJK[ind_test])))
print("Time {}s, loss in-s {:g}, loss out-of-s {:g}".format(round(end-begin, 1), res_loss[0], res_loss[1]))

# Plot
ind_plot = M1train+8
uuplot1 = np.linspace(-1.5, 1.5, M3)
uuplot = np.zeros([1, 1, 1, M3, m])
uuplot[0, 0, 0, :, 0] = uuplot1

hid = A0*tt1 + np.sum(A1*uuplot, axis = -1) - B
RN_plot = np.sum(Y*act(hid), axis = 1)
X_IJK_plot = np.sum(np.expand_dims(RN_plot, axis = 1)*np.expand_dims(xi_alpha[:, ind_plot:(ind_plot+1)], axis = [2, 3]), axis = 0)

u2, t2 = np.meshgrid(uuplot1, tt)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(u2, t2, X_plot[0])
ax.plot_surface(u2, t2, X_IJK_plot[0])
ax.set_xlabel("$u_1$")
ax.set_ylabel("$t$")
ax.set_zlabel("$X_t(\omega)(u_1,0)$")
ax.dist = 10.5
ax.view_init(20)
plt.show()

e1 = time.time()
tms = e1 - b1

np.savetxt(path + "/" + name + "_" + str(J) + "_X_true.csv", np.reshape(X_plot, [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(J) + "_X_IJK.csv", np.reshape(X_IJK_plot, [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(J) + "_err.csv", res_loss)
np.savetxt(path + "/" + name + "_" + str(J) + "_tms.csv", [tms])