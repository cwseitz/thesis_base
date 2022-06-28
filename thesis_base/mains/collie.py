import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _format_ax import *
import matplotlib.patches as mpatches


fp = open('data.json', 'r')
dict = json.load(fp)
gtf_files = glob('*.gtf')

for file in gtf_files:
    print(f'Processing file: {file}')
    df = read_gtf(file)
    df['reads'] = pd.to_numeric(df['reads'])
    df['length'] = pd.to_numeric(df['length'])
    df['gene_name'] = df.apply(lambda row : map_ensembl_to_genesymbol(row['gene_id']),axis=1)
    filename = file.split('.')[0]
    df.to_csv(f'{filename}.csv')

log10p_thres=2
log2fc_color_thres=0.58
log2fc_thres = 0.58

goi = [
'OAS1',
'IRF1',
'STAT1',
'MX1',
'TAP1',
'IFI35',
'OAS2',
'STAT2',
'EIF2AK2',
'ADAR',
'TNFRSF1A',
'ATF3',
'CPEB2',
'DYNC1H1',
'RBFOX1',
'BCL2'
]


df_2h = pd.read_excel('RNASeq.xlsx',sheet_name='2h')
df_8h = pd.read_excel('RNASeq.xlsx',sheet_name='8h')
df_24h = pd.read_excel('RNASeq.xlsx',sheet_name='24h')

#fig0, ax0 = plt.subplots(figsize=(8,6))
#fig1, ax1 = plt.subplots(figsize=(8,6))
#fig2, ax2 = plt.subplots(figsize=(8,6))

fig, (ax0, ax1, ax2) = plt.subplots(1,3)
goi = None


#2h
temp = df_2h.loc[df_2h['Pvalue_2h'] < 1e-312]
df_2h_pzero_labels = temp['gene_symbol'].to_list()
df1 = df_2h.loc[df_2h['log2FC_2h'] >= log2fc_color_thres]
df2 = df_2h.loc[df_2h['log2FC_2h'] <= -log2fc_color_thres]

log2fc_2h_1 = df1['log2FC_2h'].to_numpy()
log10p_2h_1 = -np.log10(df1['Pvalue_2h'].to_numpy())
labels_2h_1 = df1['gene_symbol'].to_list()

log2fc_2h_2 = df2['log2FC_2h'].to_numpy()
log10p_2h_2 = -np.log10(df2['Pvalue_2h'].to_numpy())
labels_2h_2 = df2['gene_symbol'].to_list()

#8h
temp = df_8h.loc[df_8h['Pvalue_8h'] < 1e-312]
df_8h_pzero_labels = temp['gene_symbol'].to_list()
df1 = df_8h.loc[df_8h['log2FC_8h'] >= log2fc_color_thres]
df2 = df_8h.loc[df_8h['log2FC_8h'] <= -log2fc_color_thres]

log2fc_8h_1 = df1['log2FC_8h'].to_numpy()
log10p_8h_1 = -np.log10(df1['Pvalue_8h'].to_numpy())
labels_8h_1 = df1['gene_symbol'].to_list()

log2fc_8h_2 = df2['log2FC_8h'].to_numpy()
log10p_8h_2 = -np.log10(df2['Pvalue_8h'].to_numpy())
labels_8h_2 = df2['gene_symbol'].to_list()

#24h
temp = df_24h.loc[df_24h['Pvalue_24h'] < 1e-312]
df_24h_pzero_labels = temp['gene_symbol'].to_list()
df1 = df_24h.loc[df_24h['log2FC_24h'] >= log2fc_color_thres]
df2 = df_24h.loc[df_24h['log2FC_24h'] <= -log2fc_color_thres]

log2fc_24h_1 = df1['log2FC_24h'].to_numpy()
log10p_24h_1 = -np.log10(df1['Pvalue_24h'].to_numpy())
labels_24h_1 = df1['gene_symbol'].to_list()

log2fc_24h_2 = df2['log2FC_24h'].to_numpy()
log10p_24h_2 = -np.log10(df2['Pvalue_24h'].to_numpy())
labels_24h_2 = df2['gene_symbol'].to_list()

volcano(ax0,log2fc_2h_1,log10p_2h_1,labels_2h_1,log2fc_2h_2,log10p_2h_2,labels_2h_2,df_2h_pzero_labels,log10p_thres=log10p_thres,log2fc_color_thres=log2fc_color_thres,log2fc_thres=log2fc_thres,goi=goi)
volcano(ax1,log2fc_8h_1,log10p_8h_1,labels_8h_1,log2fc_8h_2,log10p_8h_2,labels_8h_2,df_8h_pzero_labels,log10p_thres=log10p_thres,log2fc_color_thres=log2fc_color_thres,log2fc_thres=log2fc_thres,goi=goi)
volcano(ax2,log2fc_24h_1,log10p_24h_1,labels_24h_1,log2fc_24h_2,log10p_24h_2,labels_24h_2,df_24h_pzero_labels,log10p_thres=log10p_thres,log2fc_color_thres=log2fc_color_thres,log2fc_thres=log2fc_thres,goi=goi)

ax0.set_title('2h',fontsize=16)
ax1.set_title('8h',fontsize=16)
ax2.set_title('24h',fontsize=16)


plt.show()
