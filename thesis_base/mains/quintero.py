import argparse
import collections
import numpy as np
import pims
import cellquant as cq
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from cellquant.smt import detect_blobs, detect_blobs_batch, fit_psf, fit_psf_batch

#read in the data
path = '/home/cwseitz/Desktop/quintero_data/'
file = 'FastImg_Cell01.tif'
im = imread(path+file)
nt, nx, ny, nc = im.shape
test_im = im[:3,:,:,2]

#detect RNA in all frames
blobs_df, plt_array = detect_blobs_batch(test_im,threshold=0.05)
psf_df, plt_array = fit_psf_batch(test_im,blobs_df,diagnostic=True,pltshow=True)
