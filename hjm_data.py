import numpy as np
import itertools as itt
import scipy.special as scsp
import time

path = "hjm"
np.random.seed(0)
b1 = time.time()

I = 1
J = 5
K = 3

M1 = 200
M2 = 20
M3 = 80

T = 1.0
dt = T/M2
tt = np.linspace(0.0, T, M2)

# Sampling via inverse CDF
c = 0.1
M = 3.0
C = (np.exp(c*M)-1.0)/c
unif = np.random.uniform(low = 0.0, high = 1.0, size = M3)
uu = np.log(c*C*unif+1.0)/c

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

# Wick polynomials
ind1 = itt.product(np.arange(K+1), repeat = I*J)
ind2 = itt.filterfalse(lambda x: np.sum(x) > K, ind1)
alpha = np.reshape(np.array([seq for seq in ind2]), [-1, I, J])
alpha1 = np.expand_dims(alpha, axis = -1)
xi_ij1 = np.expand_dims(xi_ij, axis = 0)
xi_alpha = np.prod(scsp.eval_hermitenorm(alpha1, xi_ij1)/np.sqrt(scsp.gamma(alpha1+1)), axis = (1, 2))
nr_alpha = xi_alpha.shape[0]

# Vasicek process
mu = 4.0
kappa = 0.9
sigma = 0.5
r = np.zeros([M1, M2])
r[:, 0] = 4.0
for t in range(M2-1):
    r[:, t+1] = r[:, t] + (mu - kappa*r[:, t])*dt + sigma*(W[0, :, t+1] - W[0, :, t])
    
np.savetxt(path + "/" + path + "_u.csv", uu)
np.savetxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_W.csv", W[0])
np.savetxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_W_ij.csv", W_ij[0])
np.savetxt(path + "/" + path + "_" + str(J) + "_" + str(K) + "_xi_alpha.csv", xi_alpha)
np.savetxt(path + "/" + path + "_r.csv", r)