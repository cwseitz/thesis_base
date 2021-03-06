from glob import glob
import pandas as pd
import numpy as np
import biomart
import mygene
import pickle
import os

def download_ensembl_mappings():
    # Set up connection to server
    server = biomart.BiomartServer('http://uswest.ensembl.org/biomart')
    mart = server.datasets['hsapiens_gene_ensembl']

    # List the types of data we want
    attributes = ['ensembl_transcript_id', 'hgnc_symbol',
                  'ensembl_gene_id', 'ensembl_peptide_id']

    # Get the mapping between the attributes
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')

    ensembl_to_genesymbol = {}
    # Store the data in a dict
    for line in data.splitlines():
        line = line.split('\t')
        # The entries are in the same order as in the `attributes` variable
        transcript_id = line[0]
        gene_symbol = line[1]
        ensembl_gene = line[2]
        ensembl_peptide = line[3]

        ensembl_to_genesymbol[transcript_id] = gene_symbol
        ensembl_to_genesymbol[ensembl_gene] = gene_symbol
        ensembl_to_genesymbol[ensembl_peptide] = gene_symbol

    return ensembl_to_genesymbol


def map_ensembl_to_genesymbol(ensembl_name,ensembl_map):
    try:
        common_name = ensembl_map[ensembl_name]
        return common_name
    except:
        return ensembl_name

def get_ensembl_mappings():
    
    print('Getting ENSEMBL mappings...')
    if not os.path.exists('ensembl_map.json'):
        ensembl_map = download_ensembl_mappings()
        print('Done. Dumping to disk...')
        fp = open('ensembl_map.json', 'wb')
        pickle.dump(ensembl_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Found ensembl_map.json for ENSEMBL mappings...')
        fp = open('ensembl_map.json', 'rb') 
        ensembl_map = pickle.load(fp)
    return ensembl_map

def add_gene_name(df,col):
    ensembl_map = get_ensembl_mappings()
    df['gene_name'] = df.apply(lambda row : map_ensembl_to_genesymbol(row[col],ensembl_map),axis=1)
    return df

def refseq_to_gene_name(ids):

    """
    Too lazy to find a db with all of the mappings, but I can convert
    a list of refseq ids, if it is provided. This function will pickle the dict 
    and save in the wd and try to load it by default
    """

    print('Getting RefSeq mappings...')
    if not os.path.exists('refseq_map.json'):
        mg = mygene.MyGeneInfo()
        refseq_map = mg.querymany(ids, scopes='refseq', as_dataframe=True)
        print(refseq_map)
        refseq_map = dict(zip(refseq_map.index, refseq_map['symbol']))
        print('Done. Dumping to disk...')
        fp = open('refseq_map.json', 'wb')
        pickle.dump(refseq_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Found refseq_map.json for RefSeq mappings...')
        fp = open('refseq_map.json', 'rb') 
        refseq_map = pickle.load(fp)
    return refseq_map







