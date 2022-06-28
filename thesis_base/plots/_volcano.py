import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from ._format_ax import *

def volcano(ax,df,fc_col_name,pval_col_name,log10p_thres=10,log2fc_color_thres=0.58,log2fc_thres=0.58,annotate=False,goi=None,s=1):

    """
    If goi is a list of gene names, only those genes will be annotated. Otherwise
    the genes which meet the log10p and log2fc thresholds will be annotated
    """


    df_red = df.loc[(df[fc_col_name] > log2fc_thres) & (-np.log10(df[pval_col_name]) > log10p_thres)]
    df_blue = df.loc[(df[fc_col_name] < -log2fc_thres) & (-np.log10(df[pval_col_name]) > log10p_thres)]
    df_gray = df.loc[(df[fc_col_name].abs() < log2fc_thres) | (-np.log10(df[pval_col_name]) < log10p_thres)]

    ax.scatter(df_red[fc_col_name], -np.log10(df_red[pval_col_name]), color='red', s=s)
    ax.scatter(df_blue[fc_col_name], -np.log10(df_blue[pval_col_name]), color='blue', s=s)
    ax.scatter(df_gray[fc_col_name], -np.log10(df_gray[pval_col_name]), color='gray', s=s)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.vlines(log2fc_thres,ymin,ymax,color='gray',linestyle='--')
    ax.vlines(-log2fc_thres,ymin,ymax,color='gray',linestyle='--')
    ax.hlines(log10p_thres,xmin,xmax,color='gray',linestyle='--')
    
    format_ax(ax,xlabel=r'$\log_{2}\; \mathrm{FC}$',ylabel=r'$-\log_{10}\;\mathrm{P}$',ax_is_box=False,label_fontsize='large')
    red_patch = mpatches.Patch(color='red', label=f'{len(df_red)}')
    blue_patch = mpatches.Patch(color='blue', label=f'{len(df_blue)}')
    gray_patch = mpatches.Patch(color='gray', label=f'{len(df_gray)}')
    patches = [red_patch,blue_patch,gray_patch]

    ax.legend(handles=patches,loc='upper right')
