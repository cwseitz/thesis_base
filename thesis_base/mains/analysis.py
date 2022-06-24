import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def unpack_and_pool(loaded_ls):

    #match the shapes
    ts_arr = []
    ts_nuc_arr = []
    for i, loaded in enumerate(loaded_ls):
        tt = loaded_ls[i]['tt']
        ts = loaded_ls[i]['ts']
        ts_nuc = loaded_ls[i]['ts_nuc']
        if tt.shape[0] == 16:
            tt = np.delete(tt,1,axis=0)
            ts = np.delete(ts,1,axis=0)
            ts_nuc = np.delete(ts_nuc,1,axis=0)
        ts_arr.append(ts)
        ts_nuc_arr.append(ts_nuc)

    ts = np.concatenate(ts_arr, axis=-1)
    ts_nuc = np.concatenate(ts_nuc_arr, axis=-1)
    return ts, ts_nuc

def get_median(x):
    max_idx = np.argmax(x, axis=0)
    #zero out the maximum
    for n, idx in enumerate(max_idx):
        x[idx,n] = 0
    #compute the median
    med = np.median(x[np.nonzero(x)])
    return med

def get_counts(x):

    """
    This function retrives the RNA count, per cell including the number at the transcription site
    
    Similar to get_ts_counts(), we remove the putative TS from the cell and compute the median. 
    Then, we replace the putative TS and each spot is divided by the median to get a total RNA count
    """


    x[np.isnan(x)] = 0 #set nans to zero
    x = x/x.max() #normalize values
    med = get_median(x)  

def get_ts_counts(x):

    """
    0. If a nucleus contains less than 3 RNAs, it is considered OFF, otherwise it is ON

    1. We remove the maximum for each nucleus, and compute the median intensity of all spots in all nuclei

    2. We identify the brightest spot in each nucleus and label it as a putative TS

    4. If a cell is ON, the number of RNAs at the TS is defined as the nearest integer multiple of the median value from (1)
    """

    x[np.isnan(x)] = 0 #set nans to zero
    x = x/x.max() #normalize values
    med = get_median(x)

    #get nuclei which are ON (have more than 3 RNAs)
    y = deepcopy(x)
    y[x > 0] = 1
    s = np.sum(y,axis=0)
    on_idx = np.squeeze(np.argwhere(s >= 3))
    x_on = x[:,on_idx]

    #determine which of the ON nuclei have an active TS
    nuc_max = np.amax(x_on,axis=0)
    active_idx = np.squeeze(np.argwhere(nuc_max >= 2*med))
    x_on_active = x_on[:,active_idx]
    ts = np.amax(x_on_active,axis=0)
    counts = np.round(ts/med)

    return counts


step02_ctt1_rep1 = np.load('step02_ctt1_rep1.npz')
step02_ctt1_rep2 = np.load('step02_ctt1_rep2.npz')
step02_stl1_rep1 = np.load('step02_stl1_rep1.npz')
step02_stl1_rep2 = np.load('step02_stl1_rep2.npz')

step04_ctt1_rep1 = np.load('step04_ctt1_rep1.npz')
step04_ctt1_rep2 = np.load('step04_ctt1_rep2.npz')
step04_ctt1_rep3 = np.load('step04_ctt1_rep3.npz')
step04_stl1_rep1 = np.load('step04_stl1_rep1.npz')
step04_stl1_rep2 = np.load('step04_stl1_rep2.npz')
step04_stl1_rep3 = np.load('step04_stl1_rep3.npz')

step02_ctt1 = unpack_and_pool([step02_ctt1_rep1,step02_ctt1_rep2])
step02_stl1 = unpack_and_pool([step02_stl1_rep1,step02_stl1_rep2])

step04_ctt1 = unpack_and_pool([step04_ctt1_rep1,step04_ctt1_rep2,step04_ctt1_rep3])
step04_stl1 = unpack_and_pool([step04_stl1_rep1,step04_stl1_rep2,step04_stl1_rep3])

step04_stl1_ts, step04_stl1_ts_nuc = step04_stl1



for i in range(step04_stl1_ts_nuc.shape[0]):
    counts = get_ts_counts(step04_stl1_ts_nuc[i])
    print(counts)






