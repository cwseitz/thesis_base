import argparse
import collections
import numpy as np
import pims
import cellquant as cq
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from cellquant.smt import detect_blobs, detect_blobs_batch, fit_psf, fit_psf_batch, track_blobs
from cellquant.video import anim_blob, anim_traj

def add_traj_length(physdf):
    """
    Add column to physdf: 'traj_length'
    Parameters
    ----------
    physdf : DataFrame
        DataFrame containing 'x', 'y', 'frame', 'particle'
    Returns
    -------
    df: DataFrame
        DataFrame with added 'traj_length' column
    """

    particles = physdf['particle'].unique()

    for particle in particles:
        traj_length = len(physdf[ physdf['particle']==particle ])
        physdf.loc[physdf['particle']==particle, 'traj_length'] = traj_length

    return physdf

#read in the data
path = '/home/cwseitz/Desktop/quintero_data/'
file = 'FastImg_Cell01.tif'
mask = 'FastImg_Cell01-mask.tif'
im = imread(path+file)
mask = imread(path+mask)
mask = mask/mask.max()
mask = mask.astype(np.int)
nt, nx, ny, nc = im.shape
test_im = im[:100,:,:,2]

#detect RNA in all frames
blobs_df, plt_array = detect_blobs_batch(test_im,threshold=0.015,diagnostic=False,pltshow=False)

#filter blobs_df according to the mask
blobs_df['in_mask'] = mask[blobs_df['x'].to_numpy().astype(np.int),blobs_df['y'].to_numpy().astype(np.int)]
blobs_df = blobs_df.loc[blobs_df['in_mask'] == 1]
#psf_df, plt_array, batch_error = fit_psf_batch(test_im,blobs_df,diagnostic=False,pltshow=False)

search_range = 5
memory = 5
pixel_size = 1
frame_rate = 1
blobs_df, im = track_blobs(blobs_df, search_range, memory, pixel_size, frame_rate, divide_num=5, do_filter=True)
blobs_df = add_traj_length(blobs_df)
blobs_df = blobs_df.loc[blobs_df['traj_length'] > 25]
anim_tif = anim_traj(blobs_df, test_im)
imsave(path+'anim.tif',anim_tif)

