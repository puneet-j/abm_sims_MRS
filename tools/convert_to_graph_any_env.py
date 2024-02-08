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
# from itertools import compress

STATES = {'RECRUIT':0, 'ASSESS':1, 'TRAVEL_HOME_TO_RECRUIT':2, 'TRAVEL_SITE':3, 
          'OBSERVE':4, 'EXPLORE':5, 'TRAVEL_HOME_TO_OBSERVE':6}

MAX_DIST=ENVIRONMENT_BOUNDARY_X[-1]

def getSiteID(site):
    if site == 'H':
        return -1
    else:
        try:
           return site
        except:
            pdb.set_trace()


def get_current_state(astates, asites, aposes, sposes, squals):
    try:
        current_state_main = []
        for id, (site, state, pos) in enumerate(sorted(zip(asites, astates, aposes), key = lambda x: (squals, ))):
            current_state = []
            if not(site is None):
                whichSite = getSiteID(site)
                current_state.append(1.0*pos[0]/MAX_DIST)
                current_state.append(1.0*pos[1]/MAX_DIST)
                current_state.append(1.0*STATES[state])
                # current_state.append(1.0*STATES[state]/len(STATES))
                if whichSite==-1:
                    current_state.append([0.0,0.0])
                    current_state.append(0.0)
                else:
                    current_state.append(1.0*sposes[whichSite][0]/MAX_DIST)
                    current_state.append(1.0*sposes[whichSite][1]/MAX_DIST)
                    current_state.append(1.0*squals[whichSite])
            
            else:
                current_state.append(1.0*pos[0]/MAX_DIST)
                current_state.append(1.0*pos[1]/MAX_DIST)
                current_state.append(1.0*STATES[state])
                # current_state.append(1.0*STATES[state]/len(STATES))
                current_state.append(1.0)
                current_state.append(1.0)
                current_state.append(0.0)
            current_state_main.append(current_state)
        current_state_main = sorted(current_state_main, key = lambda x: (1.0 - x[4], x[2], 1.0 - x[0]**2 + x[1]**2))
        retVal = []
        for cstate in current_state_main:
            for c in cstate:
                retVal.append(c)
    except Exception as e:
        print(e)
        pdb.set_trace()  
    # pdb.set_trace()
 
    # pdb.set_trace()
    return tuple(retVal)

def round_function(x, d):
    new = []
    for r in x:
        new.append(np.round(r,decimals=d))
    # pdb.set_trace()
    return tuple(new)

def get_unique_IDs(fl, dict_old, nodeSize):
    states_unique = np.unique(fl.currentState.apply(lambda x: round_function(x, 3)))

    for i in range(len(states_unique)):
        if states_unique[i] in dict_old:
            nodeSize[states_unique[i]] += 1.0
        else:
            # pdb.set_trace()
            dict_old[states_unique[i]] = len(dict_old)
            nodeSize[states_unique[i]] = 1.0
    return dict_old, nodeSize


def get_edges_success_time(fl, IDLookup, get_edges_with, success_dict, time_dict, success, time_conv):
    fl['stateIDs'] = fl.apply(lambda x: IDLookup[round_function(x.currentState, 3)], axis=1)
    # fl['currentState'] = fl['currentState'].apply(lambda x: list(x))
    # tempstate =  fl.apply(lambda x: round_function(x.currentState, 3), axis=1)
    # for st in tempstate:
    #     if st == (0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 0, 0, 1, 0.001, 0, 0, 1, 0.001, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 0, 1, 0, 0.016, 0.0, 0, 0, 1, 0.98, 0.2, 0, 1, 0, 0.016, 0.0, 0, 0, 1, 0.98, 0.6):
    #         pdb.set_trace()
    for id, (csid, time_val) in enumerate(zip(fl.stateIDs.values, fl.time.values)):
        if csid in time_dict:
            time_dict[csid].append(time_conv)#([[time_conv - time_val], [time_val]])
        else:
            time_dict[csid] = [time_conv] #[[time_conv - time_val], [time_val]]

        if csid in success_dict:
            success_dict[csid].append(success)
        else:
            success_dict[csid] = [success]
        
        if id+1 == len(fl.stateIDs.values):
            break
        if csid in get_edges_with:
            get_edges_with[csid].append(fl.stateIDs.values[id+1])
        else:
            get_edges_with[csid] = [fl.stateIDs.values[id+1]]
        # if csid == 88:
        #     pdb.set_trace()


    return get_edges_with, success_dict, time_dict

def main():
    folder = './graphsage_results/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'metadata.csv'
    folder_graph = './graphs/graphsage_graph/'
    new_metadata_file = folder_graph + 'metadata.csv'
    meta_arr = []
    metadata = pd.read_csv(metadata_file) 
    metadata.site_qualities=metadata.site_qualities.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    graph_metaFile = 'graphMetadata.csv'
    for some_id, entry in enumerate(df.iterrows()):
        if entry[1].iloc[2] != 10: # or some_id==0:
            continue
        print(entry)
        graph = nx.Graph()
        IDLookup = dict()
        nodeSize = dict()
        has_edges_with = dict()
        success_dict = dict()
        time_dict = dict()
        for fileName, site_conv, time_conv in zip(entry[1].iloc[3], entry[1].iloc[5], entry[1].iloc[6]):
            ''' TEMP BREAK'''
            # pdb.set_trace()
            quals = entry[1].iloc[0]
            fl = pd.read_csv(folder + fileName)
            fl.agent_states = fl.agent_states.apply(literal_eval)
            fl.agent_sites = fl.agent_sites.apply(literal_eval)
            fl.agent_positions = fl.agent_positions.apply(literal_eval)
            # pdb.set_trace()
            success_now = site_conv/max(quals) if site_conv else 0.0 #1 if site_conv == max(quals) else 0
            
            # pdb.set_trace()
            fl['currentState'] = fl.apply(lambda x: get_current_state(x.agent_states, x.agent_sites, x.agent_positions, entry[1].iloc[1], entry[1].iloc[0]), axis=1)
            IDLookup, nodeSize = get_unique_IDs(fl, IDLookup, nodeSize)

            has_edges_with, success_dict, time_dict = get_edges_success_time(fl, IDLookup, has_edges_with, success_dict, time_dict, success_now, time_conv)
        

        nodeMetaArr = []
        # for id, successVal in success_dict.items():
        #     pdb.set_trace()
        #     success_dict[id] = np.nanmean()
        #     if np.isnan(successVal):
        #         success_dict[id] = 0
        #     time_dict[id] = np.nanmean(time_dict[id])*1.0/TIME_LIMIT
            

        ''' ADD NODES, NODE SIZES, and EDGES WITH WEIGHTS'''
        for nodePos, nodeID in IDLookup.items():
            graph.add_node(nodeID, x=nodePos, sz=nodeSize[nodePos])#, success=np.mean(success_dict[nodeID]), time=np.mean(time_dict[nodeID][0]), time_now=np.mean(time_dict[nodeID][1]))
            nodeMetaArr.append([nodeID, np.nanmean(success_dict[nodeID]), np.mean(time_dict[nodeID])*1.0/TIME_LIMIT])

        for node,value in has_edges_with.items():
            for edge_to in value:
                if graph.has_edge(node, edge_to):
                    graph[node][edge_to]['weight'] += 1.0
                else:
                    graph.add_edge(node, edge_to, weight=1.0)

        # for succ, time, pos, id in zip(IDLookup.items())
        # pdb.set_trace()
        fname = str(entry[1][0]) + str(entry[1][1]) + str(entry[1][2]) + '.pickle'
        # pdb.set_trace()
        meta_arr.append([entry[1].iloc[0], entry[1].iloc[1], entry[1].iloc[2], entry[1].iloc[6], list(np.nan_to_num(entry[1].iloc[5], nan=0.0, posinf=1.0, neginf=0.0))])
        ''' PUNEET: TODO: TEST'''
        fil =  open(folder_graph+fname, 'wb')
        pickle.dump(graph, fil)   
        fil.close() 
        ''' TEMP BREAK'''
        try:
            df_metaArr = pd.DataFrame(nodeMetaArr, columns=['nodeID', 'mean_success', 'mean_conv_time'])
            df_metaArr.to_csv(folder_graph + graph_metaFile)
        except:
            pdb.set_trace()
        # break
        # fil2 = open(new_metadata_file, 'wb')
    df_new_metadata = pd.DataFrame(meta_arr, columns=['qualities', 'positions', 'agents', 'time_converged', 'site_converged'])
    df_new_metadata.to_csv(new_metadata_file)
    


main()

