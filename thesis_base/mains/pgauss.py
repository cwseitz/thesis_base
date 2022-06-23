import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm
from matplotlib.gridspec import GridSpec

def sample_pdf(lam,dt,sigma):
    p = poisson(lam*dt)
    g = norm(loc=0, scale=sigma)
    xx = p.rvs(size=10000)
    yy = xx + g.rvs(size=10000)
    return np.array([xx,yy])

def joint_pdf(x,y,lam,dt,sigma):
    
    #joint density over x and x + y
    fx = poisson.pmf(x, lam*dt)
    mat = np.zeros((100,100))
    #iterate over the integer poisson axis
    for i, fx_i in enumerate(fx):
        mat[i] = norm.pdf(x, loc=x[i], scale=sigma)*fx_i
    fy = np.sum(mat,axis=0)
    return mat, fx, fy

x = np.arange(0,100,1) #Poisson axis (discrete photon (shot) noise)
y = np.linspace(-10,10,100) #Gaussian axis (continous readout noise)

dt = 5 #ms
lam = 10 #photons/ms or cps
sigma = 5 #readout noise s.d.

mat, fx, fy = joint_pdf(x,y,lam,dt,sigma)

fig = plt.figure()
gs = GridSpec(4,4)

ax_pdf = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0,0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_pdf.imshow(mat,aspect=1,origin='lower',cmap='plasma')
ax_pdf.set_xlabel(r'$H_{k} \;[\mathrm{e^{-}}]$')
ax_pdf.set_ylabel(r'$S_{k} \;[\mathrm{p}]$')

ax_hist_x.plot(fx,x,color='black')
ax_hist_y.plot(x,fy,color='black')
ax_hist_y.set_xlabel(r'$H_{k}$')
ax_hist_y.set_ylabel(r'$P(H_{k})$')
ax_hist_x.set_ylabel(r'$S_{k}$')
ax_hist_x.set_xlabel(r'$P(S_{k})$')

plt.tight_layout()
######################################

zz = sample_pdf(lam,dt,sigma)
y_vals, y_bins = np.histogram(zz[1],bins=10,density=True)
x_vals, x_bins = np.histogram(zz[0],bins=10,density=True)

fig = plt.figure()
gs = GridSpec(4,4)

ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_y = fig.add_subplot(gs[0,0:3])
ax_hist_x = fig.add_subplot(gs[1:4, 3])

ax_scatter.scatter(zz[1],zz[0],color='black',s=0.1)
ax_scatter.set_xlabel(r'$H_{k} \;[\mathrm{e^{-}}]$')
ax_scatter.set_ylabel(r'$S_{k} \;[\mathrm{p}]$')

ax_hist_x.plot(x_vals,x_bins[:-1],color='black')
ax_hist_y.plot(y_bins[:-1],y_vals,color='black')
ax_hist_y.set_xlabel(r'$H_{k}$')
ax_hist_y.set_ylabel(r'$P(H_{k})$')
ax_hist_x.set_ylabel(r'$S_{k}$')
ax_hist_x.set_xlabel(r'$P(S_{k})$')
ax_hist_y.set_xlim([0,100])
ax_hist_x.set_ylim([0,100])

ax_scatter.set_xlim([0,100])
ax_scatter.set_ylim([0,100])
ax_scatter.set_aspect(1)


plt.tight_layout()
plt.show()


