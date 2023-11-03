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

STATES = {'OBSERVE':0, 'RECRUIT':1, 'ASSESS':2, 'TRAVEL_SITE':3, 'TRAVEL_HOME_TO_RECRUIT':4}

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
    vec = [0]*(numsites+1)
    # if site==None:
    #     vec[0] = 1
    vec[site+1] = 1
    return vec

# OUR ONE HOT VECTOR FOR SITES: [Hub, Site 1, Site 2]

def get_assessors_or_recruiters(site, astate, asite, val, siteQuals):
    agents_at_site = [True if site == a else False for a in asite]
    states = list(compress(astate,agents_at_site))
    ags = sum([1 if a==val else 0 for a in states]) 
    return [oneHotSiteVector(len(siteQuals), site), siteQuals[site], ags*1.0/len(astate)]

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


    # if len(flat_arr)!=203:
    # pdb.set_trace()
    # else:
    #     print(len(flat_arr))
    return flat_arr

def fix_non_full(TO, TA, TR, A, R, sq, nA):
    l = len(sq)
    lenTO = len(TO)
    lenTA = len(TA)
    lenTR = len(TR)
    lenA = len(A)
    lenR = len(R)
    for i in range(0, nA - lenTO):
        TO.append([TIME_LIMIT/TIME_LIMIT])

    for i in range(0, nA - lenTA):
        TA.append([oneHotSiteVector(len(sq), -1), TIME_LIMIT/TIME_LIMIT])

    for i in range(0, nA - lenTR):
        TR.append([oneHotSiteVector(len(sq), -1), TIME_LIMIT/TIME_LIMIT])

    for i in range(0, l - lenA):
        A.append([oneHotSiteVector(len(sq), lenA+i), sq[lenA-1+i], 0])

    for i in range(0, l - lenR):
        R.append([oneHotSiteVector(len(sq), lenR+i), sq[lenR-1+i], 0])
    
    return TO, TA, TR, A, R 

def get_current_state_long(astate, asite, apose, sitePoses, siteQuals, numAg):
    current_state = []#np.zeros((10, 5))
    explorers = 0
    observers = 0
    TO = []
    TA = []
    TR = []
    A = []
    R = []
    # pdb.set_trace()
    # sitePoses=get_qual_from_string(sitePoses)
    # siteQuals=get_qual_from_string(siteQuals)
    for site, state, pose in zip(asite, astate, apose):
        if state == 'EXPLORE':
            explorers += 1
        if state == 'OBSERVE':
            observers += 1
        if state == 'TRAVEL_HOME_TO_OBSERVE':
            TO.append(sqrt(pose[0]**2 + pose[1]**2)/TIME_LIMIT)
        if state == 'TRAVEL_HOME_TO_RECRUIT':
            TR.append([oneHotSiteVector(len(sitePoses), site), sqrt(pose[0]**2 + pose[1]**2)/TIME_LIMIT])
        if state == 'TRAVEL_SITE':
            # pdb.set_trace()
            TA.append([oneHotSiteVector(len(sitePoses), site), sqrt((pose[0]-sitePoses[site][0])**2 + (pose[1]-sitePoses[site][1])**2)/TIME_LIMIT])
    for site in range(len(siteQuals)):
        R.append(get_assessors_or_recruiters(site, astate, asite, 'RECRUIT', siteQuals))
        A.append(get_assessors_or_recruiters(site, astate, asite, 'ASSESS', siteQuals))
    # R.append([oneHotSiteVector(len(sitePoses), site), siteQuals[site], get_assessors_or_recruiters(site, astate, asite, 'DANCE')*1.0/numAg])
    # A.append([oneHotSiteVector(len(sitePoses), site), siteQuals[site], get_assessors_or_recruiters(site, astate, asite, 'ASSESS')*1.0/numAg])
    TO, TA, TR, A, R = fix_non_full(TO, TA, TR, A, R, siteQuals, numAg)
    # pdb.set_trace()
    current_state = get_flattened_state([COMMIT_THRESHOLD, explorers*1.0/numAg, observers*1.0/numAg, TO, TA, TR, A, R])
    # print(np.shape(current_state))
    # pdb.set_trace()
    return tuple(current_state)#.reshape(np.shape(current_state)[0]*np.shape(current_state)[1]*np.shape(current_state)[2]))


def get_current_state(astate, asite):
    current_state = np.zeros((10, 5))
    for site, state in zip(asite, astate):
        if state == 'RECRUIT':
            current_state[0, STATES[state]] += 1
        else:
            if state == 'EXPLORE' or state == 'TRAVEL_HOME_TO_OBSERVE':
                continue

            elif state == 'OBSERVE':
                current_state[0, STATES[state]] += 1

            elif not(site is None):
                current_state[getSiteID(site),STATES[state]] += 1

            else:
                pdb.set_trace()
                # raise(Exception)
    
    return tuple(current_state.reshape(50))

def round_function(x, d):
    new = []
    for r in x:
        new.append(np.round(r,decimals=d))
    # pdb.set_trace()
    return tuple(new)

def get_unique_IDs(fl, dict_old, nodeSize):
    states_unique = np.unique(fl.currentState.apply(lambda x: round_function(x, 2)))
    # pdb.set_trace()
    # dict_new = copy.deepcopy(dict_old)
    for i in range(len(states_unique)):
        if states_unique[i] in dict_old:
                nodeSize[states_unique[i]] += 1.0
        else:
            # pdb.set_trace()
            dict_old[states_unique[i]] = len(dict_old)
            nodeSize[states_unique[i]] = 1.0
    return dict_old, nodeSize

# def get_qual_from_string(st):_TO_ASSESS
#     new = st.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","").split()
#     pdb.set_trace()
# # def update_edge()

def get_edges(fl, IDLookup, get_edges_with):
    fl['stateIDs'] = fl.apply(lambda x: IDLookup[round_function(x.currentState, 2)], axis=1)
    fl['currentState'] = fl['currentState'].apply(lambda x: list(x))
    # pdb.set_trace()
    # prev_state = copy.deepcopy(fl.currentState.values[0])
    # prev_state_id = 0
    for id, csid in enumerate(fl.stateIDs.values):
        if id+1 == len(fl.stateIDs.values):
            break
        if csid in get_edges_with:
            get_edges_with[csid].append(fl.stateIDs.values[id+1])
        else:
            get_edges_with[csid] = [fl.stateIDs.values[id+1]]

    return get_edges_with

def main():
    folder = './test_results/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'metadata.csv'
    folder_graph = './graphs/random_test2/'
    new_metadata_file = folder_graph + 'metadata.csv'
    # df_new_metadata = pd.DataFrame([], columns=['qualities', 'positions', 'agents'])
    meta_arr = []
    metadata = pd.read_csv(metadata_file) 
    metadata.site_qualities=metadata.site_qualities.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(literal_eval)
    # pdb.set_trace()
    metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    # pdb.set_trace()
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    
    for some_id, entry in enumerate(df.iterrows()):
        if some_id == 0:
            continue
        print(entry)
        graph = nx.Graph()
        IDLookup = dict()
        nodeSize = dict()
        has_edges_with = dict()
        for fileName in entry[1][3]:
            ''' TEMP BREAK'''
            # break
            # print(fileName)

            fl = pd.read_csv(folder + fileName)
            fl.agent_states = fl.agent_states.apply(literal_eval)
            fl.agent_sites = fl.agent_sites.apply(literal_eval)
            fl.agent_positions = fl.agent_positions.apply(literal_eval)
            # pdb.set_trace()
            fl['currentState'] = fl.apply(lambda x: get_current_state_long(x.agent_states, x.agent_sites, x.agent_positions, entry[1][1], entry[1][0], int(entry[1][2])), axis=1)
            IDLookup, nodeSize = get_unique_IDs(fl, IDLookup, nodeSize)
            # print(len(IDLookup))
            # print(len(nodeSize))
            # get_edges(IDLo)
            # if fileName=='1695410459226961.csv':
            #     pdb.set_trace()
            
            has_edges_with = get_edges(fl, IDLookup, has_edges_with)
            # print(len(has_edges_with))

        # pdb.set_trace()        
        ''' ADD NODES, NODE SIZES, and EDGES WITH WEIGHTS'''
        for nodePos, nodeID in IDLookup.items():
            # pdb.set_trace()
            graph.add_node(nodeID, x=nodePos, sz=nodeSize[nodePos])

        # for nS, nodeID in nodeSize.items():
        #     graph.nodes[nodeID]['size'] = nS

        for node,value in has_edges_with.items():
            for edge_to in value:
                if graph.has_edge(node, edge_to):
                    graph[node][edge_to]['weight'] += 1.0
                else:
                    graph.add_edge(node, edge_to, weight=1.0)

        # pdb.set_trace()
        fname = str(entry[1][0]) + str(entry[1][1]) + str(entry[1][2]) + '.pickle'
        # pdb.set_trace()
        meta_arr.append([entry[1][0], entry[1][1], entry[1][2], entry[1][6], list(np.nan_to_num(entry[1][5], nan=0.0, posinf=1.0, neginf=0.0))])
        ''' PUNEET: TODO: TEST'''
        fil =  open(folder_graph+fname, 'wb')
        pickle.dump(graph, fil)   
        fil.close() 
        ''' TEMP BREAK'''
        break
        # fil2 = open(new_metadata_file, 'wb')
    df_new_metadata = pd.DataFrame(meta_arr, columns=['qualities', 'positions', 'agents', 'time_converged', 'site_converged'])
    df_new_metadata.to_csv(new_metadata_file)


main()

