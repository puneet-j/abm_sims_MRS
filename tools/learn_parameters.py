import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
# import tensorflow as tf
import pandas as pd
import pdb
from ast import literal_eval
import copy
STATES = {'REST':0, 'DANCE':1, 'ASSESS':2, 'TRAVEL_SITE':3, 'TRAVEL_HOME_TO_DANCE':4}

def getSiteID(site):
    if site == 'H':
        return 0
    else:
        try:
           return site + 1
        except:
            pdb.set_trace()

        # pdb.set_trace()
        # return (int(site[-1]) + 1)

def get_current_state(astate, asite):
    current_state = np.zeros((10, 5))
    for site, state in zip(asite, astate):
        if state == 'DANCE':
            current_state[0, STATES[state]] += 1
        else:
            if state == 'EXPLORE' or state == 'TRAVEL_HOME_TO_REST':
                continue

            elif state == 'REST':
                current_state[0, STATES[state]] += 1

            elif not(site is None):
                current_state[getSiteID(site),STATES[state]] += 1

            else:
                pdb.set_trace()
                # raise(Exception)
    
    return tuple(current_state.reshape(50))


def main():
    folder = './sim_results/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    data_files = []
    metadata_files = []

    for i in range(0,len(files)-1,2):
        data_files.append(files[i])
        metadata_files.append(files[i+1])
    
    for dat_file, metadat_file in zip(data_files, metadata_files):
        graph_edges_with_weights = {}
        dat = pd.read_csv(folder + dat_file)
        metadat = pd.read_csv(folder + metadat_file)
        prev_state = None
        # dat.agent_sites = dat.agent_sites.apply(literal_eval)
        # metadat.num_sites = metadat.num_sites.apply(literal_eval)
        dat.agent_states = dat.agent_states.apply(literal_eval)
        dat.agent_sites = dat.agent_sites.apply(literal_eval)
        dat['currentState'] = dat.apply(lambda x: get_current_state(x.agent_states, x.agent_sites), axis=1)
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
        
        df = pd.DataFrame.from_dict(data=graph_edges_with_weights, orient='index')
        # pdb.set_trace()
        df['quals'] = np.array([metadat.site_qualities.values]).repeat(len(df))
        df['poses'] = np.array([metadat.site_positions.values]).repeat(len(df))
        df.to_csv(folder+'node_edges/'+dat_file, header=False)

main()

