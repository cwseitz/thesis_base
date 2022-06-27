import numpy as np

#def preprocess(x):
#    for i in range(x.shape[0]):
#        x[i][np.isnan(x[i])] = 0 #set nans to zero
#        x[i] = x[i]/x[i].max() #normalize each time point
#    return x

def unpack_and_pool(loaded_ls):

    #match the shapes
    tt_arr = []
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
        tt_arr.append(tt)
        ts_arr.append(ts)
        ts_nuc_arr.append(ts_nuc)

    tt_arr = np.array(tt_arr)
    ts = np.concatenate(ts_arr, axis=-1)
    ts_nuc = np.concatenate(ts_nuc_arr, axis=-1)
    return tt_arr, ts, ts_nuc

path = '/home/cwseitz/Desktop/munsky_data/'
#step02_ctt1_rep1 = np.load(path+'step02_ctt1_rep1.npz')
#step02_ctt1_rep2 = np.load(path+'step02_ctt1_rep2.npz')
step02_stl1_rep1 = np.load(path+'step02_stl1_rep1.npz')
step02_stl1_rep2 = np.load(path+'step02_stl1_rep2.npz')

#step02_ctt1 = unpack_and_pool([step02_ctt1_rep1,step02_ctt1_rep2])
step02_stl1 = unpack_and_pool([step02_stl1_rep1,step02_stl1_rep2])

#step02_ctt1_all = step02_ctt1[1]
#step02_ctt1_nuc = step02_ctt1[2]
#step02_ctt1_all[np.isnan(step02_ctt1_all)] = 0
#step02_ctt1_nuc[np.isnan(step02_ctt1_nuc)] = 0
#step02_ctt1_all = np.clip(step02_ctt1_all,a_min=0,a_max=None)
#step02_ctt1_nuc = np.clip(step02_ctt1_nuc,a_min=0,a_max=None)

step02_stl1_tt = step02_stl1[0]
step02_stl1_all = step02_stl1[1]
step02_stl1_nuc = step02_stl1[2]
step02_stl1_all[np.isnan(step02_stl1_all)] = 0
step02_stl1_nuc[np.isnan(step02_stl1_nuc)] = 0
step02_stl1_all = np.clip(step02_stl1_all,a_min=0,a_max=None)
step02_stl1_nuc = np.clip(step02_stl1_nuc,a_min=0,a_max=None)

#np.savez_compressed(path+'step02_ctt1', nuc=step02_ctt1_nuc, all=step02_ctt1_all)
np.savez_compressed(path+'step02_stl1', tt=step02_stl1_tt, nuc=step02_stl1_nuc, all=step02_stl1_all)
