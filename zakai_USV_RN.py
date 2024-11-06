import numpy as np
import scipy.linalg as scla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "zakai"
name = "zakai_USV_RN"
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
W_ij = np.reshape(np.loadtxt(path + "/" + path + "_" + str(J) + "_W_ij.csv"), [I, M1, M2])
xi_alpha = np.loadtxt(path + "/" + path + "_" + str(J) + "_xi_alpha.csv")
nr_alpha = xi_alpha.shape[0]

X = np.reshape(np.loadtxt(path + "/" + path + "_X.csv"), [M1, M2, M3])
X_plot = np.reshape(np.loadtxt(path + "/" + path + "_X_plot.csv"), [1, M2, M3])

Y = np.reshape(np.loadtxt(path + "/" + path + "_" + str(J) + "_Y.csv"), [M1, M2, m])

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
A0 = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 1.0)
A1 = np.random.normal(size = (nr_alpha, N, 1, 1, m), scale = 0.8/np.sqrt(m))
B = np.random.normal(size = (nr_alpha, N, 1, 1), scale = 0.5)

# Compute random neurons
hid = A0*tt1 + np.sum(A1*uu1, axis = -1) - B
rndn = act(hid)
rndn_der = np.expand_dims(act_der(hid), -1)*A1
rndn_der2 = act_der2(hid)*np.square(np.sum(A1, axis = -1))

# Functions
sigma2 = 1.0
beta = 0.25
def mu(x):
    nrm2 = np.sum(np.square(x), axis = -1, keepdims = True)
    return beta*x/(1.0+nrm2)

def divmu(x):
    nrm2 = np.sum(np.square(x), axis = -1)
    return beta*(m+(m-2.0)*nrm2)/np.square(1.0+nrm2)

gamma = 0.5

# Loss function
X_0 = np.exp(-np.sum(np.square(uu1[0]), axis = -1)/(2.0*sigma2))/np.power(2.0*np.pi*sigma2, 0.5*m)
drift1 = np.cumsum(np.expand_dims(0.5*m*rndn_der2[:, :, 1:] - np.sum(mu(uu1)*rndn_der[:, :, 1:], axis = -1), 2), axis = 3)*dt
drift2 = np.cumsum(gamma**2*np.expand_dims(rndn[:, :, 1:], 2)*np.expand_dims(np.sum(np.expand_dims(Y[ind_train, 1:], [0, 3])*uu1, axis = -1), 0), axis = 3)*dt
drift3 = -np.cumsum(np.expand_dims(rndn[:, :, 1:], 2)*divmu(uu1), axis = 3)*dt

diffu = np.cumsum(gamma*np.expand_dims(rndn[:, :, 1:], 2)*np.expand_dims(np.sum(uu1*np.expand_dims(np.transpose(W_ij[m:, ind_train, 1:] - W_ij[m:, ind_train, :-1], [1, 2, 0]), [0, 3]), axis = -1), 1), axis = 3)
drift1 = np.concatenate([np.zeros_like(drift1)[:, :, :, 0:1], drift1], axis = 3)
drift2 = np.concatenate([np.zeros_like(drift2)[:, :, :, 0:1], drift2], axis = 3)
drift3 = np.concatenate([np.zeros_like(drift3)[:, :, :, 0:1], drift3], axis = 3)
diffu = np.concatenate([np.zeros_like(diffu)[:, :, :, 0:1], diffu], axis = 3)

# Train linear readout
begin = time.time()
A_lsq = np.transpose(np.reshape((np.expand_dims(rndn, 2) - (drift1 + drift2 + drift3 + diffu))*np.expand_dims(xi_alpha[:, ind_train], axis = [1, 3, 4]), [nr_alpha*N, -1]))
b_lsq = np.reshape(np.tile(X_0, [M1train, M2, 1]), -1)
V1 = scla.lstsq(A_lsq, b_lsq, lapack_driver = 'gelsy')[0]
V = np.reshape(V1, [nr_alpha, N, 1, 1])

# Compute random neural network approximation
RN = np.sum(V*rndn, axis = 1)
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
RN_plot = np.sum(V*act(hid), axis = 1)
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

np.savetxt(path + "/" + name + "_" + str(J) + "_X_true.csv", X_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_X_IJK.csv", X_IJK_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_err.csv", res_loss)
np.savetxt(path + "/" + name + "_" + str(J) + "_tms.csv", [tms])