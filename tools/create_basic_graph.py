import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
# import tensorflow as tf
import pandas as pd
import pdb
from ast import literal_eval
import copy
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from params import ENVIRONMENT_BOUNDARY_X, TIME_LIMIT, COMMIT_THRESHOLD
from math import sqrt
from itertools import compress

STATES = {'OBSERVE':0, 'RECRUIT':1, 'ASSESS':2, 'TRAVEL_SITE':3, 
          'TRAVEL_HOME_TO_RECRUIT':4,'TRAVEL_HOME_TO_OBSERVE':5, 'EXPLORE': 6}

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

def oneHotSiteVector(numsites, site):
    vec = [0]*(numsites+1)
    # if site==None:
    #     vec[0] = 1
    vec[site+1] = 1
    return vec

# OUR ONE HOT VECTOR FOR SITES: [Hub, Site 1, Site 2]

def get_flattened_state(arr):
    # [explorers, observers, TO, TR, TA, R, A]
    # pdb.set_trace()
    flat_arr = []
    for a in arr:
        if type(a)==int or type(a)==float:
            flat_arr.append(a)
        else:
            for b in a:
                if type(b)==int or type(b)==float:
                    flat_arr.append(b)
                else:
                    for c in b:
                        if type(c)==int or type(c)==float:
                            flat_arr.append(c)
                        else:
                            for d in c:
                                if type(d)==int or type(d)==float:
                                    flat_arr.append(d)
                                else:
                                    pdb.set_trace()
    # return np.hstack([np.array(a).flatten() if isinstance(a, (np.ndarray, list)) else a for a in arr])
    return flat_arr


def get_current_state_basic(astate, asite, apose, sitePoses, siteQuals, numAg, s, t, tnow):
    # pdb.set_trace()
    current_state = [tnow]
    current_state.append(COMMIT_THRESHOLD)
    environment = [(x,y,z) for x,(y,z) in zip(siteQuals,sitePoses)]  #np.zeros((10, 5))
    
    for states in environment:
        current_state.append(states[0])
        current_state.append(states[1])
        current_state.append(states[2])


    for id, (site, state, pose) in enumerate(zip(asite, astate, apose)):
        # current_state.append(id)
        current_state.append(pose[0])
        current_state.append(pose[1])
        current_state.append(STATES[state])
        current_state.append(site if site else -1)
    
    current_state.append(s)
    current_state.append(t)

    

    return current_state #.reshape(np.shape(current_state)[0]*np.shape(current_state)[1]*np.shape(current_state)[2]))

def get_current_state_basic_sorted_sites(astate, asite, apose, sitePoses, siteQuals, numAg, s, t, tnow):
    # pdb.set_trace()
    current_state = [tnow]
    current_state.append(COMMIT_THRESHOLD)
    environment = [(x,y,z) for x,(y,z) in sorted(zip(siteQuals,sitePoses))]  #np.zeros((10, 5))
    
    for states in environment:
        current_state.append(states[0])
        current_state.append(states[1])
        current_state.append(states[2])


    for id, (site, state, pose) in enumerate(zip(asite, astate, apose)):
        # current_state.append(id)
        current_state.append(pose[0])
        current_state.append(pose[1])
        current_state.append(STATES[state])
        current_state.append(site if site else -1)
    
    current_state.append(s)
    current_state.append(t)

    

    return current_state #.reshape(np.shape(current_state)[0]*np.shape(current_state)[1]*np.shape(current_state)[2]))

def round_function(x, d):
    new = []
    for r in x:
        new.append(np.round(r,decimals=d))
    # pdb.set_trace()
    return tuple(new)



def main():
    folder = './new_results/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'metadata.csv'
    folder_graph = './graphs/basic_test/'

    metadata = pd.read_csv(metadata_file) 
    metadata.site_qualities=metadata.site_qualities.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    biglist = []
    biglist_sorted = []
    for some_id, entry in enumerate(df.iterrows()):
        print(entry)
        for fileName, site_conv, time_conv in zip(entry[1][3], entry[1][5], entry[1][6]):
            ''' TEMP BREAK'''
            quals = entry[1][0]
            fl = pd.read_csv(folder + fileName)
            fl.agent_states = fl.agent_states.apply(literal_eval)
            fl.agent_sites = fl.agent_sites.apply(literal_eval)
            fl.agent_positions = fl.agent_positions.apply(literal_eval)
            # pdb.set_trace()
            success_now = site_conv/max(quals) #1 if site_conv == max(quals) else 0
            fl['currentState'] = fl.apply(lambda x: get_current_state_basic(x.agent_states, x.agent_sites, x.agent_positions, entry[1][1], entry[1][0], int(entry[1][2]), success_now, time_conv, x.time), axis=1)
            fl['currentState_sorted'] = fl.apply(lambda x: get_current_state_basic(x.agent_states, x.agent_sites, x.agent_positions, entry[1][1], entry[1][0], int(entry[1][2]), success_now, time_conv, x.time), axis=1)

            # pdb.set_trace()
            biglist += fl.currentState.values.tolist()
            biglist_sorted += fl.currentState_sorted.values.tolist()

        # pdb.set_trace()
    fname = 'combined_data_basic.pickle'
    # pdb.set_trace()
    # meta_arr.append([entry[1][0], entry[1][1], entry[1][2], entry[1][6], list(np.nan_to_num(entry[1][5], nan=0.0, posinf=1.0, neginf=0.0))])
    ''' PUNEET: TODO: TEST'''
    fil =  open(folder_graph+fname, 'wb')
    pickle.dump(biglist, fil)   
    fil.close() 

    fname2 = 'combined_data_basic_sorted.pickle'
    # pdb.set_trace()
    # meta_arr.append([entry[1][0], entry[1][1], entry[1][2], entry[1][6], list(np.nan_to_num(entry[1][5], nan=0.0, posinf=1.0, neginf=0.0))])
    ''' PUNEET: TODO: TEST'''
    fil =  open(folder_graph+fname2, 'wb')
    pickle.dump(biglist_sorted, fil)   
    fil.close() 

    #     ''' TEMP BREAK'''
    #     # break
    #     # fil2 = open(new_metadata_file, 'wb')
    # df_new_metadata = pd.DataFrame(meta_arr, columns=['qualities', 'positions', 'agents', 'time_converged', 'site_converged'])
    # df_new_metadata.to_csv(new_metadata_file)


main()

