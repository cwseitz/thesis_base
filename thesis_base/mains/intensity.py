import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def get_counts_naive(x):

    """
    This function retrives the RNA count in a naive way, just as the number of spots
    """
    
    x[x>0] = 1
    counts = np.sum(x,axis=0)
    return counts
    
def get_active_idx(x,median):

    y = deepcopy(x)
    mx = np.amax(y,axis=0) #determine which of the ON nuclei have an active TS
    active_idx = np.argwhere(mx >= 2*med).flatten() #determine which cells have an active TS

    return active_idx


def get_ts_counts(x_nuc, median):

    ts = np.amax(x_nuc,axis=0)
    counts = np.round(ts/median)
            
    return counts

def get_median(x):

    med = np.median(x[np.nonzero(x)])
    return med

def add_int_hist(ax,x,bins):

    """
    Get intensity histogram
    """

    y = x[np.nonzero(x)]
    ax.hist(y,bins=bins,color='blue',alpha=0.5,density=True)

def get_on_idx(x):

    y = deepcopy(x)
    counts = get_counts_naive(y)
    on_idx = np.argwhere(counts >= 3) #get cells which are ON (have more than 3 spots)

    return on_idx.flatten()

  
#################################################################################################################
#Normalize the data per gene, based on maximum value observed in all time points and all conditions for that gene
#################################################################################################################

path = '/home/cwseitz/Desktop/munsky_data/'

step02_stl1 = np.load(path+'step02_stl1.npz')
step02_stl1_all = step02_stl1['all']
step02_stl1_nuc = step02_stl1['nuc']
step02_stl1_tt = step02_stl1['tt'][0]

step04_stl1 = np.load(path+'step04_stl1.npz')
step04_stl1_all = step04_stl1['all']
step04_stl1_nuc = step04_stl1['nuc']
step04_stl1_tt = step04_stl1['tt'][0]


step02_stl1_max = np.max(step02_stl1_all)
step04_stl1_max = np.max(step04_stl1_all)

step02_stl1_all = step02_stl1_all/step02_stl1_max
step04_stl1_all = step04_stl1_all/step04_stl1_max
step02_stl1_nuc = step02_stl1_nuc/step02_stl1_max
step04_stl1_nuc = step04_stl1_nuc/step04_stl1_max


##########################################################################
#Show intensity histogram for all spots observed for the step04 condition
##########################################################################

fig, ax = plt.subplots()
bins = 100
add_int_hist(ax, step04_stl1_all, bins)
ymin, ymax = ax.get_ylim()
med = get_median(step04_stl1_all)
ax.vlines(med,ymin,ymax,color='black',label=f'median={np.round(med,2)}')
ax.set_xlim([0,0.25])
ax.set_xticks([0,0.25])
ax.set_xlabel('Peak intensity (a.u.)',fontsize=12)
ax.set_ylabel('PDF')
ax.legend(loc='upper right',prop={'size': 12})
plt.show()

##########################################################################
#Find the brightest spot in each nucleus (putative TS) and plot dist
##########################################################################

fig, ax = plt.subplots(3,5,sharex=False)
ax = ax.ravel()

fig2, ax2 = plt.subplots(3,5,sharex=False)
ax2 = ax2.ravel()

for i in range(15):
    time = step04_stl1_tt[i]
    idx = get_on_idx(step04_stl1_all[i]) #determine if on based on all RNAs
    x = step04_stl1_nuc[i,:,idx].T
    active_idx = get_active_idx(x, med)
    if len(active_idx) > 0:
        x = x[:,active_idx]
        counts = get_ts_counts(x, med)
        bins = np.arange(0,10,1)
        ax[i].hist(counts,color='blue',alpha=0.5,bins=bins,density=True)
        ax[i].set_xlabel('TS count')
        ax[i].set_ylabel('PDF')
        ax[i].set_xticks([0,5,10])
        ax[i].set_title(f'0.4M NaCl @ {time} min', fontsize=8, fontweight='bold')
        
        #for cells with an active TS, demonstrate variability in the nuclear counts
        nuc_counts = get_counts_naive(x)
        ax2[i].hist(nuc_counts,color='blue',bins=bins,alpha=0.5,density=True)
        ax2[i].set_xlabel('Nuclear STL1 mRNA')
        ax2[i].set_ylabel('PDF')
        ax2[i].set_title(f'0.4M NaCl @ {time} min', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()



