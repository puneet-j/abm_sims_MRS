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
    # lenA = len(A)
    # lenR = len(R)
    for i in range(0, nA - lenTO):
        TO.append([TIME_LIMIT/TIME_LIMIT])

    for i in range(0, nA - lenTA):
        TA.append([oneHotSiteVector(len(sq), -1), TIME_LIMIT/TIME_LIMIT])

    for i in range(0, nA - lenTR):
        TR.append([oneHotSiteVector(len(sq), -1), TIME_LIMIT/TIME_LIMIT])

    # for i in range(0, l - lenA):
    #     A.append([oneHotSiteVector(len(sq), lenA+i), sq[lenA-1+i], 0])

    # for i in range(0, l - lenR):
    #     R.append([oneHotSiteVector(len(sq), lenR+i), sq[lenR-1+i], 0])
    
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

def get_unique_IDs(fl):
    states_unique = np.unique(fl.currentState.apply(lambda x: round_function(x, 2)))
    # pdb.set_trace()
    dict_new = {}
    for i in range(len(states_unique)):
        dict_new[states_unique[i]] = i
    return dict_new

# def get_converged_nodes(graph):
#     for n in graph.nodes['x']:
#         if n

# def get_qual_from_string(st):_TO_ASSESS
#     new = st.replace("(","").replace(")","").replace("[","").replace("]","").replace(",","").split()
#     pdb.set_trace()
# # def update_edge()

def get_start_state(which_state, q):
    q0 = q[0]
    q1 = q[1]
    if which_state == 0:
        return (0.5, 0.0, 1.0)
    elif which_state == 1:
        return (0.5, 0.5, 0.5)
    elif which_state == 2:
        return (0.5, 0.0, 0.9)
        # return     [0.5, 0.0, 1.0, 
        #             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        #             1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 
        #             1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0,
        #             0,1,0,q0,0.0, 1,0,0, q1, 0.0,
        #             0,1,0,q0,0.0, 1,0,0, q1, 0.0]
def main():
    folder = './graphs/new/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.pickle')]
    files = np.sort(files)
    metadata_file = folder + 'metadata.csv'
    metadata = pd.read_csv(metadata_file)
    # pdb.set_trace()
    metadata.qualities=metadata.qualities.apply(literal_eval)
    # metadata.site_positions=metadata.site_positions.apply(literal_eval)
    # metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    # df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())


    '''
    # [COMMIT_THRESHOLD, explorers*1.0/numAg, observers*1.0/numAg, TO, TA, TR, A, R]

    [0.5, 0.0, 1.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0,
    1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0, 1,1,0,1.0,
    0,1,0,q0,0.0, 1,0,0, q1, 0.0,
    0,1,0,q0,0.0, 1,0,0, q1, 0.0]

    {'x': (0.5, 0.0, 0.0, 
           0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
           0, 0, 1, 0.0, 0, 0, 1, 0.0, 0, 0, 1, 0.0, 0, 0, 1, 0.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 
           0, 0, 1, 0.0, 0, 0, 1, 0.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 1, 0, 0, 1.0, 
           0, 1, 0, 0.01, 0.0, 0, 0, 1, 0.55, 0.0, 
           0, 1, 0, 0.01, 0.0, 0, 0, 1, 0.55, 0.0)}
    '''


    for id, row in enumerate(metadata.iterrows()):
        if id == 0:
            continue 
        else:
            # pdb.set_trace()
            fname = str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
            fil =  open(folder+fname, 'rb')
            graph = pickle.load(fil)
            fil.close()
            qual = row[1][1]
            node_to_color = tuple(get_start_state(0,qual))
            id_to_color = [tuple((z, y['x'][:3],node_to_color)) for z,y in graph.nodes(data=True) if y['x'][:3] == node_to_color]
            all_ids_start = [tuple((z, y['x'][:3])) for z,y in graph.nodes(data=True)]# if y['x'][:3] == node_to_color]
            colors = 'r'#['r' for i in range(len(graph.nodes))]
            nx.set_node_attributes(graph, colors, 'colors')
            try:
                graph.nodes[id_to_color[0][0]]['colors'] = 'k'
            except:
                # print(Exception)
                pdb.set_trace()
            # id_to_color = [tuple((y['x'][:3],node_to_color)) for z,y in graph.nodes(data=True)]
            # pdb.set_trace()
            id_of_goal = [tuple((z, y['x'][-1], y['x'][-6], y['x'][0])) for z,y in graph.nodes(data=True) if (y['x'][-1] >= y['x'][0] or y['x'][-6] >= y['x'][0])]
            all_ids_goal = [tuple((z, y['x'][-1], y['x'][-6], y['x'][0])) for z,y in graph.nodes(data=True)]# if (y['x'][-1] >= y['x'][0] or y['x'][-6] >= y['x'][0])]

            # id_of_goal = [tuple((z, y['x'][-1], y['x'][-6], y['x'][0])) for z,y in graph.nodes(data=True) if (y['x'][-1] >= 0.5 or y['x'][-6] >= 0.5)]

            for node in id_of_goal:
                graph.nodes[node[0]]['colors'] = 'b'
            

            pdb.set_trace()

            fil =  open(folder+fname, 'wb')
            pickle.dump(graph, fil)   
            fil.close()
            ''' TEMP BREAK'''
            # break


main()

