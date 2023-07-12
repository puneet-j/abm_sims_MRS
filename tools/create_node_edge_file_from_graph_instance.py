import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
import pandas as pd
import pdb
from ast import literal_eval
import copy


def main():
    folder = '../sim_results/node_edges/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    data_files = []

    for i in range(0,len(files)-1,2):
        data_files.append(files[i])
    
    for dat_file in zip(data_files):
        graph_edges_with_weights = {}
        dat = pd.read_csv(folder + dat_file)
        metadat = pd.read_csv(folder + metadat_file)
        prev_state = None
        # dat.agent_sites = dat.agent_sites.apply(literal_eval)
        # metadat.num_sites = metadat.num_sites.apply(literal_eval)
        dat.agent_states = dat.agent_states.apply(literal_eval)
        dat.siteIDs = dat.siteIDs.apply(literal_eval)
        dat['currentState'] = dat.apply(lambda x: get_current_state(x.agent_states, x.siteIDs), axis=1)
        for id, cs in enumerate(dat.currentState.values):
            if id == 0:
                prev_state = copy.deepcopy(cs)
                # continue
            else:
                if tuple([prev_state, cs]) in graph_edges_with_weights.keys():
                    graph_edges_with_weights[tuple([prev_state, cs])] += 1
                else:
                    graph_edges_with_weights[tuple([prev_state, cs])] = 1
                prev_state = copy.deepcopy(cs)
        
        (pd.DataFrame.from_dict(data=graph_edges_with_weights, orient='index').to_csv(folder+'dict_'+dat_file, header=False))

main()