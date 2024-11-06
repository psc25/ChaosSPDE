import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "zakai"
name = "zakai_USV_NN"
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
N = 25
val_split = 0.2
M1train = int((1.0-val_split)*M1)
ind_train = np.arange(M1train)
ind_test = np.arange(M1train, M1)
ep = 10000
eval_every = 100
lr = 5e-4
batch_size = 40
nrbatch = int(M1train/batch_size)
act = tf.tanh
act_der = lambda x: 1.0/tf.square(tf.cosh(x))
act_der2 = lambda x: -2.0*tf.tanh(x)*act_der(x)

# Plot true solution
ind_plot = M1train+8
uuplot1 = np.linspace(-1.5, 1.5, M3)
uuplot = np.zeros([1, 1, 1, M3, m])
uuplot[0, 0, 0, :, 0] = uuplot1

# Tensors and initializations
tt1_tf = tf.placeholder(shape = (1, 1, M2, 1), dtype = tf.float32)
uu1_tf = tf.placeholder(shape = (1, 1, 1, M3, m), dtype = tf.float32)
xi_alpha_tf = tf.placeholder(shape = (nr_alpha, None), dtype = tf.float32)
W_tf = tf.placeholder(shape = (I, None, M2), dtype = tf.float32)
W_ij_tf = tf.placeholder(shape = (I, None, M2), dtype = tf.float32)
X_tf = tf.placeholder(shape = (None, M2, M3), dtype = tf.float32)
Y_tf = tf.placeholder(shape = (None, M2, m), dtype = tf.float32)

init = tf.random_normal_initializer(stddev = 1e-5)
A0 = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)
A1 = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1, m)), dtype = tf.float32)
B = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)
L = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)

# Neural network approximation
hid = A0*tt1_tf + tf.reduce_sum(A1*uu1_tf, axis = -1) - B
NN = tf.reduce_sum(L*act(hid), axis = 1)
X_IJK_tf = tf.reduce_sum(tf.expand_dims(NN, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1]), axis = 0)

NN_der = tf.reduce_sum(tf.expand_dims(L*act_der(hid), -1)*A1, axis = 1)
X_IJK_der_tf = tf.reduce_sum(tf.expand_dims(NN_der, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1, 1]), axis = 0)

NN_der2 = tf.reduce_sum(L*act_der2(hid)*tf.square(tf.reduce_sum(A1, -1)), axis = 1)
X_IJK_der2_tf = tf.reduce_sum(tf.expand_dims(NN_der2, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1]), axis = 0)

# Functions
sigma2 = 1.0
beta = 0.25
def mu(x):
    nrm2 = tf.reduce_sum(tf.square(x), axis = -1, keepdims = True)
    return beta*x/(1.0+nrm2)

def divmu(x):
    nrm2 = tf.reduce_sum(tf.square(x), axis = -1)
    return beta*(m+(m-2.0)*nrm2)/tf.square(1.0+nrm2)

gamma = 0.4

# Loss function
X_0 = tf.exp(-tf.reduce_sum(tf.square(uu1_tf[0]), axis = -1)/(2.0*sigma2))/tf.pow(2.0*np.pi*sigma2, 0.5*m)
drift1 = tf.cumsum(0.5*m*X_IJK_der2_tf[:, 1:] - tf.reduce_sum(mu(uu1_tf[0])*X_IJK_der_tf[:, 1:], axis = -1), axis = 1)*dt
drift2 = tf.cumsum(gamma**2*X_IJK_tf[:, 1:]*tf.reduce_sum(tf.expand_dims(Y_tf[:, 1:], 2)*uu1_tf[0], axis = -1), axis = 1)*dt
drift3 = -tf.cumsum(X_IJK_tf[:, 1:]*divmu(uu1_tf[0]), axis = 1)*dt
diffu = tf.cumsum(gamma*X_IJK_tf[:, 1:]*tf.einsum('ui,iot->otu', uu1_tf[0, 0, 0], W_ij_tf[m:, :, 1:] - W_ij_tf[m:, :, :-1]), axis = 1)
RHS = X_0 + tf.concat([tf.zeros_like(X_IJK_tf[:, 0:1]), drift1 + drift2 + drift3 + diffu], axis = 1)
loss_train = tf.reduce_mean(tf.square(X_IJK_tf - RHS))

loss = tf.reduce_mean(tf.square(X_tf - X_IJK_tf))

# Training
global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss_train)
train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

res_loss = np.nan*np.ones([ep, 2])
uu2, tt2 = np.meshgrid(uuplot1, tt)
for i in range(ep):
    begin = time.time()
    loss1 = np.zeros([nrbatch, 1])
    np.random.shuffle(ind_train)
    for j in range(nrbatch):
        ind_batch = ind_train[(j*batch_size):((j+1)*batch_size)]
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_batch], tt1_tf: tt1, uu1_tf: uu1, W_tf: W[:, ind_batch], W_ij_tf: W_ij[:, ind_batch], X_tf: X[ind_batch], Y_tf: Y[ind_batch]}
        _, loss1[j] = sess.run([train_op, loss], feed_dict)
        
    end = time.time()
    res_loss[i, 0] = np.sqrt(np.mean(loss1))
    print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
    
    if (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_test], tt1_tf: tt1, uu1_tf: uu1, W_tf: W[:, ind_test], W_ij_tf: W_ij[:, ind_test], X_tf: X[ind_test], Y_tf: Y[ind_test]}
        loss1 = sess.run(loss, feed_dict)
        end = time.time()
        res_loss[i, 1] = np.sqrt(loss1)
        print("\nEvaluation on test data:")
        print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
        print("")
        
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_plot:(ind_plot+1)], tt1_tf: tt1, uu1_tf: uuplot, W_tf: W[:, ind_plot:(ind_plot+1)], W_ij_tf: W_ij[:, ind_plot:(ind_plot+1)], X_tf: X[ind_plot:(ind_plot+1)], Y_tf: Y[ind_plot:(ind_plot+1)]}
        X_IJK = sess.run(X_IJK_tf, feed_dict)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.plot_surface(uu2, tt2, X_plot[0])
        ax.plot_surface(uu2, tt2, X_IJK[0])
        ax.set_xlabel("$u_1$")
        ax.set_ylabel("$t$")
        ax.set_zlabel("$X_t(\omega)(u_1,0)$")
        ax.dist = 10.5
        ax.view_init(20)
        plt.show()

feed_dict = {xi_alpha_tf: xi_alpha[:, ind_plot:(ind_plot+1)], tt1_tf: tt1, uu1_tf: uuplot, W_tf: W[:, ind_plot:(ind_plot+1)], X_tf: X[ind_plot:(ind_plot+1)], Y_tf: Y[ind_plot:(ind_plot+1)]}
X_IJK_plot = sess.run(X_IJK_tf, feed_dict)

e1 = time.time()
tms = e1 - b1

np.savetxt(path + "/" + name + "_" + str(J) + "_X_true.csv", np.reshape(X_plot[0], [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(J) + "_X_IJK.csv", np.reshape(X_IJK_plot, [M2, -1]))
np.savetxt(path + "/" + name + "_" + str(J) + "_err.csv", res_loss[-1])
np.savetxt(path + "/" + name + "_" + str(J) + "_tms.csv", [tms])
