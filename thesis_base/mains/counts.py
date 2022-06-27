import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def get_count_hist(x,bins=10):
    counts = get_counts_naive(x)
    vals, bins = np.histogram(counts,bins=bins,density=True)
    return vals, bins

def get_counts_naive(x):

    """
    This function retrives the RNA count in a naive way, just as the number of spots
    """
    
    x[x>0] = 1
    counts = np.sum(x,axis=0)
    return counts

def get_on_idx(x):

    y = deepcopy(x)
    counts = get_counts_naive(y)
    on_idx = np.argwhere(counts >= 3) #get cells which are ON (have more than 3 spots)

    return on_idx.flatten()

def normalize(a,b,c,d):
	maxima = [a.max(),b.max(),c.max(),d.max()]
	m = max(maxima)
	return a/m, b/m, c/m, d/m


def count_stats_time(t, ax, x, filter=False):

    means = []
    stdvs = []
    nsamp = []
    nt = len(t)
    
    for n in range(nt):
    
        if filter:
            idx = get_on_idx(x[n])
            xf = x[n,:,idx].T
        else:
            xf = x[n]
            
        counts = get_counts_naive(xf)
        means.append(np.mean(counts))
        stdvs.append(np.std(counts))
        nsamp.append(len(counts))
        
    means = np.array(means)
    stdvs = np.array(stdvs)
    nsamp = np.array(nsamp)
    stderr = stdvs
    ax.errorbar(t, means, yerr=stderr,color='black',capsize=4,ecolor='black')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Average total count')
        
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


#####################################################
#Analyze counts without filtering out zero count cells
#####################################################

fig, ax = plt.subplots(3,5,sharex=False)
ax = ax.ravel()

for i in range(15):
    count_vals, count_bins = get_count_hist(step04_stl1_all[i])
    ax[i].plot(count_bins[:-1],count_vals,color='purple')
    ax[i].set_title(f'{step04_stl1_tt[i]} min')
    ax[i].set_xlabel('STL1 mRNA',fontsize=10)
    ax[i].set_ylabel('PDF')
    #ax[i].legend(loc='upper right',prop={'size': 6})

fig, ax = plt.subplots()
count_stats_time(step04_stl1_tt, ax, step04_stl1_all)
plt.show()


#####################################################
#Analyze counts while filtering out zero count cells
#####################################################

fig, ax = plt.subplots(3,5,sharex=False)
ax = ax.ravel()

for i in range(15):

    idx = get_on_idx(step04_stl1_all[i])
    count_vals, count_bins = get_count_hist(step04_stl1_all[i,:,idx].T)
    ax[i].plot(count_bins[:-1],count_vals,color='purple')
    ax[i].set_title(f'{step04_stl1_tt[i]} min')
    ax[i].set_xlabel('STL1 mRNA',fontsize=10)
    ax[i].set_ylabel('PDF')
    #ax[i].legend(loc='upper right',prop={'size': 6})
   
fig, ax = plt.subplots()
count_stats_time(step04_stl1_tt, ax, step04_stl1_all,filter=True)
plt.show()

#####################################################
#Analyze the Poissonyness of the distributions
#####################################################


