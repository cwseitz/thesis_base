import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from thesis_base.utils import add_gene_name
from thesis_base.plots import *
from stats import MomentInference, MaximumLikelihood, dBP

path = '/home/cwseitz/Desktop/'
#df = pd.read_csv(path+'1-s2.0-S1097276520306882-mmc2.csv')
df = pd.read_csv(path+'GSE150198_single_cell_RNA-seq_naive_readcounts_corrected.csv',sep='\t')

#fig, ax = plt.subplots()
#df = add_gene_name(df,col='GeneID')
#volcano(ax, df,'log2(FC)','P-value')
#plt.show()

x = 'NM_000019'
df_x = df.loc[df['gene IDs'] == x].to_numpy()[0,1:]
theta = MomentInference(df_x.astype(np.float32))
#theta = MaximumLikelihood(df_x.astype(np.float32))

fig, ax = plt.subplots()
x = np.arange(150)
ax.plot(x,dBP(x,*theta),color='red')
plt.show()
