import numpy as np
import scipy.special as scsp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "zakai"
np.random.seed(0)
b1 = time.time()

m = 2

I = 2*m
J = 4
K = 1 # currently only implemented for K <= 1

M1 = 300
M2 = 20
M3 = 150

val_split = 0.2
M1train = int((1-val_split)*M1)

T = 0.5
dt = T/M2
tt = np.linspace(0.0, T, M2)

uu = np.random.normal(size = [M3, m])

def g(j, t):
    res1 = np.sqrt(1.0/T)
    res2 = np.sqrt(2.0/T)*np.cos((j-1.0)*np.pi*t/T)
    return res1*(j==1) + res2*(j>1)

def G(j, t):
    res1 = np.sqrt(1.0/T)*t
    res2 = np.sqrt(2.0*T)/np.pi*np.sin((j-1.0)*np.pi*t/T)/(j-1.0)
    res2[np.isnan(res2)] = 0.0
    return res1*(j==1) + res2*(j>1)

# Brownian motions
I1 = I
J1 = 100
xi = np.random.normal(size = [I1, J1, M1])
t1, j1 = np.meshgrid(tt, np.arange(J1))
G_j = G(j1, t1)
W = np.einsum('ijl,jm->ilm', xi, G_j)

# Approximating Brownian motions
xi_ij = xi[:I, :J]
t1, j1 = np.meshgrid(tt, np.arange(J))
G_j = G(j1, t1)
W_ij = np.einsum('ijl,jm->ilm', xi_ij, G_j) 

# Wick polynomials (works only for K = 1)
alpha = [np.zeros([I, J], dtype = np.int32)]
if K > 0:
    for i in range(I):
        for j in range(J):
            tst = np.zeros([I, J], dtype = np.int32)
            tst[i, j] = 1
            alpha.append(tst)

alpha = np.stack(alpha, axis = 0)
alpha1 = np.expand_dims(alpha, axis = -1)
xi_ij1 = np.expand_dims(xi_ij, axis = 0)
xi_alpha = np.prod(scsp.eval_hermitenorm(alpha1, xi_ij1)/np.sqrt(scsp.gamma(alpha1+1)), axis = (1, 2))
nr_alpha = xi_alpha.shape[0]

# Compute Y and Z processes
sigma2 = 1.0
beta = 0.25
def mu(x):
    den = 1.0 + np.sum(np.square(x), axis = -1, keepdims = True)
    return beta*x/den

gamma = 0.5

W1 = np.transpose(W[:m, ], [1, 2, 0])
W2 = np.transpose(W[m:, ], [1, 2, 0])
Y = np.zeros([M1, M2, m])
Y[:, 0] = np.random.normal(size = [M1, m], scale = np.sqrt(sigma2))
Z = np.zeros([M1, M2, m])
for t in range(M2-1):
    Y[:, t+1] = Y[:, t] + mu(Y[:, t])*dt + np.sum(W1[:, t+1]-W1[:, t], axis = -1, keepdims = True)/np.sqrt(m)
    Z[:, t+1] = Z[:, t] + 0.5*gamma*(Y[:, t] + Y[:, t+1])*dt + W2[:, t+1]-W2[:, t]

# Approximate the solution using the Monte-Carlo approach in:
# https://arxiv.org/abs/2210.13530 (Paper)
# https://github.com/seb-becker/zakai (Code)
M4 = 500
N = np.random.normal(size = [1, M4, M2, 1], scale = np.sqrt(dt))
X = np.zeros([M1, M2, M3])
X[:, 0] = np.exp(-np.sum(np.square(uu), axis = -1)/(2.0*sigma2))/np.power(2.0*np.pi*sigma2, 0.5*m)
for t in range(0):#(1, M2):
    print("Approximate X for t = " + str(t) + "/" + str(M2))
    R = np.expand_dims(uu, [0, 1])
    B = np.zeros([M1, M4, M3])
    for n in range(1, t+1):
        B += 0.5*dt * 0.5*gamma**2*(np.expand_dims(np.square(np.sum(Z[:, t-n+1], axis = -1)), [1, 2]) - np.sum(np.square(R), axis = -1))
        B -= 0.5*dt * gamma*np.sum(mu(R)*np.expand_dims(Z[:, t-n+1], [1, 2]), axis = -1)
        B -= 0.5*dt * m*beta/(1.0 + np.sum(np.square(R), axis = -1))
        B += 0.5*dt * 2.0*beta*np.sum(np.square(R), axis = -1)/np.square(1.0 + np.sum(np.square(R), axis = -1))
        R = R + (gamma*m*np.expand_dims(np.sum(Z[:, t-n+1], axis = -1), [1, 2, 3]) - mu(R))*dt + N[:, :, n:(n+1)]
        B += 0.5*dt * 0.5*gamma**2*(np.expand_dims(np.square(np.sum(Z[:, t-n], axis = -1)), [1, 2]) - np.sum(np.square(R), axis = -1))
        B -= 0.5*dt * gamma*np.sum(mu(R)*np.expand_dims(Z[:, t-n], [1, 2]), axis = -1)
        B -= 0.5*dt * m*beta/(1.0 + np.sum(np.square(R), axis = -1))
        B += 0.5*dt * 2.0*beta*np.sum(np.square(R), axis = -1)/np.square(1.0 + np.sum(np.square(R), axis = -1))
        
    ex = np.exp(B + gamma*np.sum(np.expand_dims(Z[:, t], [1, 2])*np.expand_dims(uu, [0, 1]), axis = -1))
    RX = np.exp(-np.sum(np.square(R), axis = -1)/(2.0*sigma2))/np.power(2.0*np.pi*sigma2, 0.5*m)
    X[:, t] = np.mean(RX*ex, axis = 1)

# Approximate the solution for plots
ind_plot = M1train+8
uuplot1 = np.linspace(-1.5, 1.5, M3)
uuplot = np.zeros([M3, m])
uuplot[:, 0] = uuplot1

X_plot = np.zeros([1, M2, M3])
X_plot[0, 0] = np.exp(-np.sum(np.square(uuplot), axis = -1)/(2.0*sigma2))/np.power(2.0*np.pi*sigma2, 0.5*m)
for t in range(1, M2):
    print("Approximate X_plot for t = " + str(t) + "/" + str(M2))
    R = np.expand_dims(uuplot, [0, 1])
    B = np.zeros([1, M4, M3])
    for n in range(1, t+1):
        B += 0.5*dt * 0.5*gamma**2*(np.expand_dims(np.square(np.sum(Z[ind_plot:(ind_plot+1), t-n+1], axis = -1)), [1, 2]) - np.sum(np.square(R), axis = -1))
        B -= 0.5*dt * gamma*np.sum(mu(R)*np.expand_dims(Z[ind_plot:(ind_plot+1), t-n+1], [1, 2]), axis = -1)
        B -= 0.5*dt * m*beta/(1.0 + np.sum(np.square(R), axis = -1))
        B += 0.5*dt * 2.0*beta*np.sum(np.square(R), axis = -1)/np.square(1.0 + np.sum(np.square(R), axis = -1))
        R = R + (gamma*m*np.expand_dims(np.sum(Z[ind_plot:(ind_plot+1), t-n], axis = -1), [1, 2, 3]) - mu(R))*dt + N[:, :, n:(n+1)]
        B += 0.5*dt * 0.5*gamma**2*(np.expand_dims(np.square(np.sum(Z[ind_plot:(ind_plot+1), t-n], axis = -1)), [1, 2]) - np.sum(np.square(R), axis = -1))
        B -= 0.5*dt * gamma*np.sum(mu(R)*np.expand_dims(Z[ind_plot:(ind_plot+1), t-n], [1, 2]), axis = -1)
        B -= 0.5*dt * m*beta/(1.0 + np.sum(np.square(R), axis = -1))
        B += 0.5*dt * 2.0*beta*np.sum(np.square(R), axis = -1)/np.square(1.0 + np.sum(np.square(R), axis = -1))
        
    ex = np.exp(B + gamma*np.sum(np.expand_dims(Z[ind_plot:(ind_plot+1), t], [1, 2])*np.expand_dims(uuplot, [0, 1]), axis = -1))
    RX = np.exp(-np.sum(np.square(R), axis = -1)/(2.0*sigma2))/np.power(2.0*np.pi*sigma2, 0.5*m)
    X_plot[0, t] = np.mean(RX*ex, axis = 1)
    
uu2, tt2 = np.meshgrid(uuplot1, tt)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_surface(uu2, tt2, X_plot[0])
ax.set_xlabel("$u_1$")
ax.set_ylabel("$t$")
ax.set_zlabel("$X_t(u_1,0)$")
ax.view_init(20)
plt.show()

np.savetxt(path + "/" + path + "_u.csv", uu)
np.savetxt(path + "/" + path + "_W.csv", np.reshape(W, [I1, M1*M2]))
np.savetxt(path + "/" + path + "_" + str(J) + "_W_ij.csv", np.reshape(W_ij, [I, M1*M2]))
np.savetxt(path + "/" + path + "_" + str(J) + "_xi_alpha.csv", xi_alpha)
#np.savetxt(path + "/" + path + "_X.csv", np.reshape(X, [M1, M2*M3]))
np.savetxt(path + "/" + path + "_X_plot.csv", X_plot[0])
np.savetxt(path + "/" + path + "_" + str(J) + "_Y.csv", np.reshape(Y, [M1, M2*m]))