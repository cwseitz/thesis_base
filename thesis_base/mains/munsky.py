import h5py
import numpy as np
filename = "RNAspots.mat"

def extract_name(f, obj):

    data1 = obj[0][0]
    data2 = obj[1][0]
    xx = f[data1]

    return np.array(xx)

def extract_tt(f, obj):

    data1 = obj[0][0]
    data2 = obj[1][0]
    xx = f[data1]
    
    return np.array(xx)

def extract_data(f, obj):
    
    obj = np.array(obj)
    tt_ls = []
    ts_nuc_ls = []
    ts_ls = []
    for i in range(obj.shape[0]):
        dataref = obj[i][0]
        data = f[dataref]['low']
        ts_nuc = ts_nuc_ls.append(np.array(data['RNAintTSNuc']))
        ts = ts_ls.append(np.array(data['RNAintTS']))
        tt = tt_ls.append(np.array(data['tt']))

    return tt_ls, ts_nuc_ls, ts_ls

####################################

f = h5py.File(filename, "r")
a = f.get('/RNAdata/Step02/rep/CTT1')
b = f.get('/RNAdata/Step02/rep/STL1')
c = f.get('/RNAdata/Step02/rep/Name')
d = f.get('/RNAdata/Step02/rep/tt')

ctt1_tt, ctt1_ts_nuc, ctt1_ts = extract_data(f,a)
stl1_tt, stl1_ts_nuc, stl1_ts = extract_data(f,b)
np.savez_compressed('step02_ctt1_rep1', tt=ctt1_tt[0], ts_nuc=ctt1_ts_nuc[0], ts=ctt1_ts[0])
np.savez_compressed('step02_ctt1_rep2', tt=ctt1_tt[1], ts_nuc=ctt1_ts_nuc[1], ts=ctt1_ts[1])
np.savez_compressed('step02_stl1_rep1', tt=stl1_tt[0], ts_nuc=stl1_ts_nuc[0], ts=stl1_ts[0])
np.savez_compressed('step02_stl1_rep2', tt=stl1_tt[1], ts_nuc=stl1_ts_nuc[1], ts=stl1_ts[1])

#name = extract_name(f,c)i
#tt = extract_tt(f,d)

#####################################

a = f.get('/RNAdata/Step04/rep/CTT1')
b = f.get('/RNAdata/Step04/rep/STL1')
c = f.get('/RNAdata/Step04/rep/Name')
d = f.get('/RNAdata/Step04/rep/tt')

ctt1_tt, ctt1_ts_nuc, ctt1_ts = extract_data(f,a)
stl1_tt, stl1_ts_nuc, stl1_ts = extract_data(f,b)
np.savez_compressed('step04_ctt1_rep1', tt=ctt1_tt[0], ts_nuc=ctt1_ts_nuc[0], ts=ctt1_ts[0])
np.savez_compressed('step04_ctt1_rep2', tt=ctt1_tt[1], ts_nuc=ctt1_ts_nuc[1], ts=ctt1_ts[1])
np.savez_compressed('step04_ctt1_rep3', tt=ctt1_tt[2], ts_nuc=ctt1_ts_nuc[2], ts=ctt1_ts[2])
np.savez_compressed('step04_stl1_rep1', tt=stl1_tt[0], ts_nuc=stl1_ts_nuc[0], ts=stl1_ts[0])
np.savez_compressed('step04_stl1_rep2', tt=stl1_tt[1], ts_nuc=stl1_ts_nuc[1], ts=stl1_ts[1])
np.savez_compressed('step04_stl1_rep1', tt=stl1_tt[2], ts_nuc=stl1_ts_nuc[2], ts=stl1_ts[2])



#name = extract_name(f,c)
#tt = extract_tt(f,d)

