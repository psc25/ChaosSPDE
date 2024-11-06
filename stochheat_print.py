import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({'legend.fontsize': 9,
                     'axes.labelsize': 9,
                     'xtick.labelsize': 8,
                     'ytick.labelsize': 8})

path = "stochheat"
names = ["stochheat_SV_NN", 
         "stochheat_SV_RN", 
         "stochheat_USV_NN", 
         "stochheat_USV_RN"]
ln = len(names)
plt_names = ["SV & $\\mathcal{NN}$",
             "SV & $\\mathcal{RN}$",
             "USV & $\\mathcal{NN}$",
             "USV & $\\mathcal{RN}$"]
cols = ["blue", "deepskyblue", "seagreen", "yellowgreen"]

mm = [1, 5, 10]
lm = len(mm)
np.random.seed(0)

M1 = 200
M2 = 20
M3 = 1000

T = 0.25
dt = T/M2
tt = np.linspace(0.0, T, M2)
tt1 = np.reshape(tt, [1, 1, M2, 1])

X_true = np.zeros([lm, ln, M2, M3])
X_IJK = np.zeros([lm, ln, M2, M3])
err = np.zeros([lm, ln, 2])
tms = np.zeros([lm, ln])

for i in range(lm):
    for j in range(ln):
        X_true[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(mm[i]) + "_X_true.csv")
        X_IJK[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(mm[i]) + "_X_IJK.csv")
        err[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(mm[i]) + "_err.csv")
        tms[i, j] = np.loadtxt(path + "/" + names[j] + "_" + str(mm[i]) + "_tms.csv")

fig, ax1 = plt.subplots(figsize = (5.3, 4.0)) 
for j in range(ln):
    ax1.plot(mm, err[:, j, 1], marker = "<", color = cols[j], alpha = 0.7)
    
ax1.set_xticks(mm)
ax1.set_xlabel('Dimension $m$')
ax1.set_ylabel('OOS empirical error')
ax1.set_ylim([0, 0.33])
ax2 = ax1.twinx()
for j in range(ln):
    if j % 2 == 0:
        ts = 6.0*tms[:, j]
    else:
        ts = tms[:, j]
    ax2.plot(mm, ts, linestyle = "dotted", marker = ">", color = cols[j], alpha = 0.7)
    
ax2.set_xticks(mm)
ax2.set_ylabel('Computational time [in $s$]')
ax2.set_ylim([0, 3300.0])
for j in range(ln):
    ax1.plot(np.nan, np.nan, color = cols[j], label = plt_names[j])

ax1.plot(np.nan, np.nan, marker = "<", color = "black", label = "OOS error")
ax1.plot(np.nan, np.nan, marker = ">", linestyle = "dotted", color = "black", label = "Time")
ax1.legend(loc = "upper center", ncol = 3)
plt.savefig(path + "/" + path + "_err_tms.png", bbox_inches = 'tight', dpi = 500)
plt.show()
plt.close(fig)

for i in range(lm):
    uuplot1 = np.linspace(-2.0, 2.0, M3)
    uuplot = np.zeros([1, 1, 1, M3, mm[i]])
    uuplot[0, 0, 0, :, 0] = uuplot1
    
    fig = plt.figure()
    u2, t2 = np.meshgrid(uuplot1, tt)
    ax = fig.add_subplot(projection = '3d')
    for j in range(ln):
        ax.plot_surface(u2, t2, X_IJK[i, j], color = cols[j], alpha = 0.7)
        
    ax.plot_wireframe(u2, t2, X_true[i, 0], color = "black", alpha = 0.7, linewidth = 0.9, rstride = 1, cstride = 40)

    for j in range(ln):
        ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], linestyle = "None", color = cols[j], alpha = 0.7, marker = "s", label = plt_names[j])
        
    ax.plot([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], linestyle = "None", color = "black", alpha = 0.7, marker = "$\\#$", markersize = 10, label = "True")
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$t$")
    if mm[i] == 1:
        ax.set_zlabel("$X_t(\omega)(u_1)$")
    else:
        ax.set_zlabel("$X_t(\omega)(u_1,0,...,0)$")
        
    ax.set_zlim([9.0, 11.0])
    ax.view_init(20)
    ax.dist = 10.61
    ax.legend(loc = "upper center", ncol = 3)
    plt.savefig(path + "/" + path + "_" + str(mm[i]) + ".png", bbox_inches = 'tight', dpi = 500)
    plt.show()
    plt.close(fig)