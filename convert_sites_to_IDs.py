import networkx as nx
import numpy as np
import scipy.sparse as sp
# import tensorflow as tf
import pandas as pd
import pdb
import os 
from ast import literal_eval

def getSitesFromPoses(siteList, site_pos_list):
    siteIDList = []
    for site in siteList:
        if not(site is None):
            for id, list_site in enumerate(site_pos_list):
                # pdb.set_trace()
                if abs(site[0] - list_site[0] + site[1] - list_site[1]) < 0.001:
                    siteIDList.append(id)
        else:
            siteIDList.append(None)
    return siteIDList

def main():
    folder = './sim_results/random_test/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv')]
    files = np.sort(files)
    lst = []
    data_files = []
    metadata_files = []

    for i in range(0,len(files)-1,2):
        # pdb.set_trace()
        data_files.append(folder+files[i])
        metadata_files.append(folder+files[i+1])
    
    for dat_file, metadat_file in zip(data_files, metadata_files):
        dat = pd.read_csv(dat_file)
        metadat = pd.read_csv(metadat_file)
        dat.agent_sites = dat.agent_sites.apply(literal_eval)
        metadat.site_positions = metadat.site_positions.apply(literal_eval)
        dat['siteIDs'] = dat.agent_sites.apply(lambda x: getSitesFromPoses(x, metadat.site_positions.values[0]))
        dat.to_csv(dat_file)

main()