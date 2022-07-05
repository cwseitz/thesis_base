import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from thesis_base.utils import add_gene_name
from thesis_base.plots import *
from stats import MomentInference, MaximumLikelihood, dBP
from stats import kraskov_mi

def match(df1, df2):
    genes1 = df1['gene IDs'].to_numpy()
    genes2 = df2['gene IDs'].to_numpy()
    return np.intersect1d(genes1,genes2)

path = '/home/cwseitz/Desktop/'
volc_df = pd.read_csv(path+'1-s2.0-S1097276520306882-mmc2.csv')
log2fc_thres = 0.58
ns = [100,100,100,100]
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(2,2)
ax2 = ax2.ravel()

volc_df = add_gene_name(volc_df,col='GeneID')
volcano(ax1, volc_df,'log2(FC)','P-value',log2fc_thres=log2fc_thres)

df_naive = pd.read_csv(path+'GSE150198_single_cell_RNA-seq_naive_readcounts_corrected.tab',sep='\t')
df_prime = pd.read_csv(path+'GSE150198_single_cell_RNA-seq_priming_readcounts_corrected.tab',sep='\t')


df_naive['avg'] = df_naive.mean(axis=1)
df_prime['avg'] = df_prime.mean(axis=1)
df_naive = df_naive.loc[df_naive['avg'] > 0]
df_prime = df_prime.loc[df_prime['avg'] > 0]

common = match(df_naive,df_prime)
df_naive = df_naive.loc[df_naive['gene IDs'].isin(common)]
df_prime = df_prime.loc[df_prime['gene IDs'].isin(common)]
df_naive = df_naive.set_index('gene IDs')
df_naive = df_naive.reindex(index=df_prime['gene IDs'])
df_prime = df_prime.set_index('gene IDs')


#############################################################
#Select N genes at random
#############################################################

def select_n(n, df_naive, df_prime):
    rand = np.random.randint(0,len(df_naive),size=(n,))
    df_naive = df_naive.iloc[rand]
    df_prime = df_prime.iloc[rand]
    return df_naive, df_prime

#############################################################
#Select genes based on their fold change from naive to primed
#############################################################


"""
r = df_prime['avg']/df_naive['avg']
r = np.log2(r.to_numpy())
idx = np.argwhere(np.abs(r) > log2fc_thres).flatten()

df_naive = df_naive.iloc[idx]
df_prime = df_prime.iloc[idx]

df_naive = df_naive.drop(columns='avg')
df_prime = df_prime.drop(columns='avg')
"""

#############################################################
#Compute the mutual information using the Kraskov method
#############################################################

def get_adj(df_naive, df_prime, k=3):

    arr_naive = df_naive.to_numpy()
    arr_prime = df_prime.to_numpy()

    ngenes = len(df_naive)
    mi_mat = np.zeros((ngenes,ngenes),dtype=np.float32)
    for i in range(ngenes):
        for j in range(ngenes):
            x = arr_prime[i,:]
            y = arr_prime[j,:]
            mi = kraskov_mi(x,y,k)
            mi_mat[i,j] = mi
            #print(f'Mutual information {i},{j} is: {mi}')

    np.fill_diagonal(mi_mat,0)
    adj = np.zeros_like(mi_mat)
    mi_thres = 0.1
    adj[mi_mat > mi_thres] = 1

    return adj, mi_mat
        
for i in range(len(ns)):

    this_df_naive, this_df_prime = select_n(ns[i], df_naive, df_prime)
    this_adj, this_mi_mat = get_adj(this_df_naive,this_df_prime)
    deg = np.sum(this_adj,axis=0)
    G = nx.from_numpy_matrix(this_adj)
    G.remove_nodes_from(list(nx.isolates(G)))
    nx.draw(G,pos=nx.spring_layout(G),ax=ax2[i],node_size=10,node_color='red')
    ax2[i].set_title(f'N=100')
    
plt.show()

#############################################################
#Degree distribution from the adjacency matrix
#############################################################




