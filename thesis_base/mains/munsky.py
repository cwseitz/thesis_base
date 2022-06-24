import h5py
import numpy as np
filename = "RNAspots.mat"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = f.get('/RNAdata/Step02/rep/CTT1')
    data1 = data[0][0]
    data2 = data[1][0]
    xx = f[data1]
    yy = xx['low']
    zz = np.array(yy['RNAintTSNuc'])
    bb = np.array(yy['RNAintTS'])
    print(bb.shape, zz.shape)
    print(zz[5,:,100])
