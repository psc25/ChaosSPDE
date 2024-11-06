import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'legend.fontsize': 9,
                     'axes.labelsize': 9,
                     'xtick.labelsize': 8,
                     'ytick.labelsize': 8})

path = "zakai"
names = ["zakai_SV_NN", 
         "zakai_SV_RN", 
         "zakai_USV_NN", 
         "zakai_USV_RN"]
ln = len(names)
plt_names = ["SV & $\\mathcal{NN}$",
             "SV & $\\mathcal{RN}$",
             "USV & $\\mathcal{NN}$",
             "USV & $\\mathcal{RN}$"]
cols = ["blue", "deepskyblue", "seagreen", "yellowgreen"]

JJ = [1, 2, 3, 4]
lJ = len(JJ)
np.random.seed(0)

M1 = 300
M2 = 20
M3 = 150

ylim1 = [0.0, 0.03]
ylim2 = [0.0, 4800.0]
zlim = [0.0, 0.15]

T = 1.0
dt = T/M2
tt = np.linspace(0.0, T, M2)
tt1 = np.reshape(tt, [1, 1, M2, 1])

X_true = np.zeros([lJ, ln, M2, M3])
X_IJK = np.zeros([lJ, ln, M2, M3])
err = np.zeros([lJ, ln, 2])
tms = np.zeros([lJ, ln])
for i in range(lJ):
    for j in range(ln):
        X_true[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(JJ[i]) + "_X_true.csv")
        X_IJK[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(JJ[i]) + "_X_IJK.csv")
        err[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(JJ[i]) + "_err.csv")
        tms[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(JJ[i]) + "_tms.csv").item()

# Error and time for different J
fig, ax1 = plt.subplots(figsize = (5.3, 4.0))
for j in range(ln):
    ax1.plot(JJ, err[:, j, 1], marker = "<", color = cols[j], alpha = 0.7)
    
ax1.set_xticks(JJ)
ax1.set_xlabel('$J$')
ax1.set_ylabel('OOS empirical error')
ax1.set_ylim(ylim1)
ax2 = ax1.twinx()
for j in range(ln):
    if j % 2 == 0:
        ts = 6.0*tms[:, j]
    else:
        ts = tms[:, j]
    ax2.plot(JJ, ts, linestyle = "dotted", marker = ">", color = cols[j], alpha = 0.7)
    
ax2.set_xticks(JJ)
ax2.set_ylabel('Computational time [in $s$]')
ax2.set_ylim(ylim2)
for j in range(ln):
    ax1.plot(np.nan, np.nan, color = cols[j], alpha = 0.7, label = plt_names[j])
    
ax1.plot(np.nan, np.nan, marker = "<", color = "black", label = "OOS error")
ax1.plot(np.nan, np.nan, marker = ">", linestyle = "dotted", color = "black", label = "Time")
ax1.legend(loc = "upper center", ncol = 3)
plt.savefig(path + "/" + path + "_err_tms.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

# Approximation for different J
uuplot1 = np.linspace(-1.5, 1.5, M3)
u2, t2 = np.meshgrid(uuplot1, tt)
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for j in range(ln):
    ax.plot_surface(u2, t2, X_IJK[-1, j], color = cols[j], alpha = 0.7)
    
ax.plot_wireframe(u2, t2, X_true[-1, 0], color = "black", alpha = 0.7, linewidth = 0.9, rstride = 2, cstride = 4)
for j in range(ln):
    ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], linestyle = "None", color = cols[j], alpha = 0.7, marker = "s", label = plt_names[j])
    
ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], linestyle = "None", color = "black", alpha = 0.7, marker = "$\\#$", markersize = 10, label = "True")
ax.set_xlabel("$u_1$")
ax.set_ylabel("$t$")
ax.set_zlabel("$X_t(\omega)(u_1,0)$")
ax.set_zlim(zlim)
ax.view_init(20)
ax.dist = 10.61
ax.legend(loc = "upper center", ncol = 3)
plt.savefig(path + "/" + path + ".png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)