import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from thesis_base.utils import add_gene_name
from thesis_base.plots import *

path = '/home/cwseitz/Desktop/'
df = pd.read_csv(path+'1-s2.0-S1097276520306882-mmc2.csv')

fig, ax = plt.subplots()
df = add_gene_name(df,col='GeneID')
volcano(ax, df,'log2(FC)','P-value')
plt.show()

#log10p_thres=10
#log2fc_color_thres=0.58
#log2fc_thres = 0.58

#df_red = df.loc[(df['log2(FC)'] > log2fc_thres) & (-np.log10(df['P-value']) > log10p_thres)]
#df_blue = df.loc[(df['log2(FC)'] < -log2fc_thres) & (-np.log10(df['P-value']) > log10p_thres)]
#df_gray = df.loc[(df['log2(FC)'].abs() < log2fc_thres) | (-np.log10(df['P-value']) < log10p_thres)]

#s=1

#fig, ax = plt.subplots()
#ax.scatter(df_red['log2(FC)'],-np.log10(df_red['P-value']),color='red',s=s)
#ax.scatter(df_blue['log2(FC)'],-np.log10(df_blue['P-value']),color='blue',s=s)
#ax.scatter(df_gray['log2(FC)'],-np.log10(df_gray['P-value']),color='gray',s=s)
#xmin, xmax = ax.get_xlim()
#ymin, ymax = ax.get_ylim()
#ax.hlines(log10p_thres, xmin, xmax,color='gray',linestyle='--')
#ax.vlines(log2fc_thres, ymin, ymax,color='gray',linestyle='--')
#ax.vlines(-log2fc_thres, ymin, ymax,color='gray',linestyle='--')

#plt.show()
