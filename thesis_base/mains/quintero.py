import argparse
import collections
import numpy as np
import pims
import cellquant as cq
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from cellquant.smt import detect_blobs, detect_blobs_batch

#read in the data
path = '/home/cwseitz/Desktop/quintero_data/'
file = 'FastImg_Cell01.tif'
im = imread(path+file)
nt, nx, ny, nc = im.shape

#detect RNA in all frames
blobs_df, plt_array = detect_blobs_batch(im[:100,:,:,2],threshold=0.05)
