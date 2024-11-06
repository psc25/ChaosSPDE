import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

path = "hjm"
name = "hjm_SV_NN"
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
N = 25
val_split = 0.2
M1train = int((1-val_split)*M1)
ind_train = np.arange(M1train)
ind_test = np.arange(M1train, M1)
ep = 10000
eval_every = 100
lr = 5e-4
batch_size = 40
nrbatch = int(M1train/batch_size)
act = tf.tanh
act_der = lambda x: 1.0/tf.square(tf.cosh(x))

# Plot true solution
ind_plot = M1train+1
uuplot1 = np.linspace(0.0, 2.0, M3)
uuplot = np.reshape(uuplot1, [1, 1, 1, M3])

# Tensors and initializations
tt1_tf = tf.placeholder(shape = (1, 1, M2, 1), dtype = tf.float32)
uu1_tf = tf.placeholder(shape = (1, 1, 1, M3), dtype = tf.float32)
xi_alpha_tf = tf.placeholder(shape = (nr_alpha, None), dtype = tf.float32)
r_tf = tf.placeholder(shape = (None, M2), dtype = tf.float32)

init = tf.random_normal_initializer(stddev = 1e-2)
A0 = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)
A1 = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)
B = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)
Y = tf.Variable(initial_value = init(shape = (nr_alpha, N, 1, 1)), dtype = tf.float32)

# Neural network approximation
hid = A0*tt1_tf + A1*uu1_tf - B
NN = tf.reduce_sum(Y*act(hid), axis = 1)
X_IJK_tf = tf.reduce_sum(tf.expand_dims(NN, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1]), axis = 0)

hid_0 = A0*tt1_tf - B
NN_0 = tf.reduce_sum(Y*act(hid_0), axis = 1)
X_IJK_0_tf = tf.reduce_sum(tf.expand_dims(NN_0, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1]), axis = 0)

NN_der = tf.reduce_sum(Y*act_der(hid)*A1, axis = 1)
X_IJK_der_tf = tf.reduce_sum(tf.expand_dims(NN_der, 1)*tf.reshape(xi_alpha_tf, [nr_alpha, -1, 1, 1]), axis = 0)

# True solution
X_tf = tf.expand_dims(r_tf, -1)*tf.exp(-kappa*uu1_tf[0]) + mu/kappa*(1.0-tf.exp(-kappa*uu1_tf[0])) - sigma**2/(2*kappa**2)*tf.square(1.0-tf.exp(-kappa*uu1_tf[0]))
X_0_tf = tf.expand_dims(r_tf, -1)
X_der_tf = -kappa*tf.expand_dims(r_tf, -1)*tf.exp(-kappa*uu1_tf[0]) + mu*tf.exp(-kappa*uu1_tf[0]) - sigma**2/kappa*(1.0-tf.exp(-kappa*uu1_tf[0]))*tf.exp(-kappa*uu1_tf[0])

# Loss function
loss1 = tf.reduce_mean(tf.square(X_0_tf - X_IJK_0_tf))
loss2 = tf.reduce_mean(tf.square(X_der_tf - X_IJK_der_tf))
loss = loss1 + loss2

# Training
global_step = tf.Variable(0, trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
grads_and_vars = optimizer.compute_gradients(loss)
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
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_batch], tt1_tf: tt1, uu1_tf: uu1, r_tf: r[ind_batch]}
        _, loss1[j] = sess.run([train_op, loss], feed_dict)
        
    end = time.time()
    res_loss[i, 0] = np.sqrt(np.mean(loss1))
    print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 0]))
    
    if (i+1) % eval_every == 0:
        begin = time.time()
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_test], tt1_tf: tt1, uu1_tf: uu1, r_tf: r[ind_test]}
        loss1 = sess.run(loss, feed_dict)
        end = time.time()
        res_loss[i, 1] = np.sqrt(loss1)
        print("\nEvaluation on test data:")
        print("Step {}, time {}s, loss {:g}".format(i+1, round(end-begin, 1), res_loss[i, 1]))
        print("")
        
        feed_dict = {xi_alpha_tf: xi_alpha[:, ind_plot:(ind_plot+1)], tt1_tf: tt1, uu1_tf: uuplot, r_tf: r[ind_plot:(ind_plot+1)]}
        X_true, X_IJK = sess.run([X_tf, X_IJK_tf], feed_dict)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.plot_surface(uu2, tt2, X_true[0])
        ax.plot_surface(uu2, tt2, X_IJK[0])
        ax.set_xlabel("$u$")
        ax.set_ylabel("$t$")
        ax.set_zlabel("$X_t(\omega)(u)$")
        ax.view_init(20)
        plt.show()
        
feed_dict = {xi_alpha_tf: xi_alpha[:, ind_plot:(ind_plot+1)], tt1_tf: tt1, uu1_tf: uuplot, r_tf: r[ind_plot:(ind_plot+1)]}
X_true_plot, X_IJK_plot = sess.run([X_tf, X_IJK_tf], feed_dict)

e1 = time.time()
tms = e1 - b1

np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_X_true.csv", X_true_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_X_IJK.csv", X_IJK_plot[0])
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_err.csv", res_loss[-1])
np.savetxt(path + "/" + name + "_" + str(J) + "_" + str(K) + "_tms.csv", [tms])