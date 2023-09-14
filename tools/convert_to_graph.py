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
from params import AGENT_SPEED

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

# nodesAt5 = [x for x,y in P.nodes(data=True) if y['at']==5]


def oneHotSiteVector(numsites, site):
    vec = [0]*numsites
    vec[site] = 1
    return vec


def get_current_state_long(astate, asite, apose, sitePoses):
    current_state = []#np.zeros((10, 5))
    explorers = 0
    observers = 0
    RO = []
    RA = []
    RD = []
    A = []
    D = []
    for site, state, pose in zip(asite, astate, apose):
        if state == 'EXPLORE':
            explorers += 1
        if state == 'REST':
            observers += 1
        if state == 'TRAVEL_HOME_TO_REST':
            RO.append(AGENT_SPEED/np.sqrt(pose[0]**2 + pose[1]**2))
        if state == 'TRAVEL_HOME_TO_DANCE':
            RD.append([oneHotSiteVector(len(sitePoses), site), AGENT_SPEED/np.sqrt(pose[0]**2 + pose[1]**2)])
        if state == 'TRAVEL_TO_SITE':
            RA.append([oneHotSiteVector(len(sitePoses), site), AGENT_SPEED/np.sqrt((pose[0]-sitePoses[0])**2 + (pose[1]-sitePoses[1])**2)])
        if state == ''

        # if state == 'DANCE':
        #     current_state[0, STATES[state]] += 1
        # else:
        #     if state == 'EXPLORE' or state == 'TRAVEL_HOME_TO_REST':
        #         continue

        #     elif state == 'REST':
        #         current_state[0, STATES[state]] += 1

        #     elif not(site is None):
        #         current_state[getSiteID(site),STATES[state]] += 1

        #     else:
        #         pdb.set_trace()
        #         # raise(Exception)
    
    return tuple(current_state.reshape(50))

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

def get_unique_IDs(fl):
    states_unique = fl.currentState.unique()
    dict_new = {}
    for i in range(len(states_unique)):
        dict_new[states_unique[i]] = i
    return dict_new


# def update_edge()

def main():
    folder = './results_for_graphs/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'fixed_metadata.csv'
    folder_graph = './graphs/'
    new_metadata_file = folder_graph + 'metadata.csv'
    # df_new_metadata = pd.DataFrame([], columns=['qualities', 'positions', 'agents'])
    meta_arr = []
    metadata = pd.read_csv(metadata_file)
    # pdb.set_trace()
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    for entry in df.iterrows():
        graph = nx.Graph()
        for fileName in entry[1][4]:
            fl = pd.read_csv(folder + fileName)
            fl.agent_states = fl.agent_states.apply(literal_eval)
            fl.agent_sites = fl.agent_sites.apply(literal_eval)
            fl.agent_positions = fl.agent_positions.apply(literal_eval)
            fl['currentState'] = fl.apply(lambda x: get_current_state(x.agent_states, x.agent_sites, x.agent_positions, entry[1][1]), axis=1)
            IDLookup = get_unique_IDs(fl)
            for nodePos, nodeID in IDLookup.items():
                # pdb.set_trace()
                graph.add_node(nodeID, x=nodePos)
            fl['stateIDs'] = fl.apply(lambda x: IDLookup[x.currentState], axis=1)
            # prev_state = copy.deepcopy(fl.currentState.values[0])
            prev_state_id = 0
            for csid in fl.stateIDs.values:             
            # for id, (cs, csid) in enumerate(zip(fl.currentState.values, fl.stateIDs.values)):
                if graph.has_edge(prev_state_id, csid):
                    graph[prev_state_id][csid]['weight'] += 1.0
                else:
                    graph.add_edge(prev_state_id, csid, weight=1.0)
                # prev_state = copy.deepcopy(cs)
                prev_state_id = copy.deepcopy(csid)
        
        fname = str(entry[1][0]) + str(entry[1][1]) + str(entry[1][2]) + '.pickle'
        meta_arr.append([entry[1][0], entry[1][1], entry[1][2]])
        # pd.concat([df_new_metadata, pd.DataFrame([entry[1][0], entry[1][1], entry[1][2]], columns=['qualities', 'positions', 'agents'])], axis=0, ignore_index=True)
        # pdb.set_trace()
        ''' PUNEET: TODO: TEST'''
        fil =  open(folder_graph+fname, 'wb')
        pickle.dump(graph, fil)   
        fil.close() 
        ''' TEMP BREAK'''
        break
        # fil2 = open(new_metadata_file, 'wb')
    df_new_metadata = pd.DataFrame(meta_arr, columns=['qualities', 'positions', 'agents'])
    df_new_metadata.to_csv(new_metadata_file)


main()

