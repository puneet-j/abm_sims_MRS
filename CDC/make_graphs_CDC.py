import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
import time
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
from concurrent.futures import ThreadPoolExecutor, as_completed

STATES = {'RECRUIT':0.0/6.0, 'ASSESS':1.0/6.0, 'TRAVEL_HOME_TO_RECRUIT':2.0/6.0, 'TRAVEL_SITE':3.0/6.0, 
          'OBSERVE':4.0/6.0, 'EXPLORE':5.0/6.0, 'TRAVEL_HOME_TO_OBSERVE':6.0/6.0}

MAX_DIST=ENVIRONMENT_BOUNDARY_X[-1]

def getSiteID(site):
    if site == 'H':
        return -1
    else:
        try:
           return site
        except:
            pdb.set_trace()

def round_function(x, d):
    new = []
    for r in x:
        new.append(np.round(r,decimals=d))
    # pdb.set_trace()
    return tuple(new)

def get_unique_IDs(fl, dict_old, nodeSize):
    states_unique = np.unique(fl.currentState)

    for i in range(len(states_unique)):
        dict_old[states_unique[i]] = i #len(dict_old)
    return dict_old#, nodeSize


def get_edges_success_time(fl, IDLookup, get_edges_with, success_dict, time_dict, success, time_conv):
    fl['stateIDs'] = fl.apply(lambda x: IDLookup[x.currentState], axis=1)
    for id, csid in enumerate(fl.stateIDs.values):
        if id+1 == len(fl.stateIDs.values):
            break
        if csid in get_edges_with:
            get_edges_with[csid].append(fl.stateIDs.values[id+1])
        else:
            get_edges_with[csid] = [fl.stateIDs.values[id+1]]



    return get_edges_with#, success_dict, time_dict

def node_to_color_black(node):
    if node[3] == 0.0:
        return True
    else:
        return False

def node_to_color_green(node, quals):
    # pdb.set_trace()
    nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    q = 0
    # pdb.set_trace()
    for d in dancers:
        if d[1] == np.max(quals):
            q += 1
    
    if q > COMMIT_THRESHOLD*nA:
        return True
    else:
        return False
    
def node_to_color_red(node, quals):
    # pdb.set_trace()
    nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    q = 0
    # pdb.set_trace()
    for d in dancers:
        if d[1] == np.min(quals):
            q += 1
    if q > COMMIT_THRESHOLD*nA:
        return True
    else:
        return False

# def get_full_qual(q):
#     while len(q) < 4:
#         q.append(0.0)
#     return q

def dancers_at_hub(node, quals):
    # pdb.set_trace()
    arr = [0]*4#len(quals)#[0, 0, 0, 0]
    danc = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    qsorted = sorted(quals, reverse=True)
    qdict = dict()
    for i, q in enumerate(qsorted):
        qdict[q] = i
    # qsorted = get_full_qual(qsorted)
    for d in danc:
        arr[qdict[d[1]]] += 1
        # if d[1] == qsorted[0]:
        #     arr[0] += 1
        # elif d[1] == qsorted[1]:
        #     arr[1] += 1
        # elif d[1] == qsorted[2]:
        #     arr[2] += 1
        # elif d[1] == qsorted[3]:
        #     arr[3] += 1   
    # pdb.set_trace()
    return arr

def process_file(fileName, site_conv, time_conv, entry, folder, folder_graph, graph_metaFile):


    graph = nx.Graph()
    IDLookup = dict()
    nodeSize = dict()
    has_edges_with = dict()
    success_dict = dict()
    time_dict = dict()
    ''' TEMP BREAK'''
    # pdb.set_trace()
    quals = entry[1].iloc[0]
    if np.max(quals) - np.min(quals) < 0.3 or np.min(quals) < 0.2 or np.max(quals) - np.min(quals) > 0.5 :
        return "Skipped", fileName
    fl = pd.read_csv(folder + fileName)

    fl.agent_states = fl.agent_states.apply(literal_eval)
    fl.agent_sites = fl.agent_sites.apply(literal_eval)
    fl.agent_positions = fl.agent_positions.apply(literal_eval)
    fl['currentState'] = fl.node.apply(literal_eval)
    success_now =  0.0 if np.isnan(site_conv) else site_conv/max(quals) #1 if site_conv == max(quals) else 0

    try:
        IDLookup = get_unique_IDs(fl, IDLookup, nodeSize)
    except Exception as e:
        print(e)
        pdb.set_trace()

    try:
        has_edges_with = get_edges_success_time(fl, IDLookup, has_edges_with, success_dict, time_dict, success_now, time_conv)
    except Exception as e:
        print(e)
        pdb.set_trace()

    nodeMetaArr = []
    qrounded = [np.round(q, decimals=3) for q in quals]

    
    try:
        ''' ADD NODES, NODE SIZES, and EDGES WITH WEIGHTS'''
        for nodePos, nodeID in IDLookup.items():
            graph.add_node(nodeID, x=nodePos)#, sz=nodeSize[nodePos])#, success=np.mean(success_dict[nodeID]), time=np.mean(time_dict[nodeID][0]), time_now=np.mean(time_dict[nodeID][1]))

        for node,value in has_edges_with.items():
            for edge_to in value:
                if graph.has_edge(node, edge_to):
                    graph[node][edge_to]['weight'] += 1.0
                else:
                    graph.add_edge(node, edge_to, weight=1.0)
    except Exception as e:
        print(e)
        pdb.set_trace()    
    # prev, now = now,time.time()*1000.0
    # print("time now 8: ", now - prev)
    
    # pdb.set_trace()
    colors = 'b'
    nx.set_node_attributes(graph, colors, 'colors')
    nx.set_node_attributes(graph, qrounded, 'quals')
    nx.set_node_attributes(graph, entry[1].iloc[1], 'poses')
    nx.set_node_attributes(graph, success_now, 'success')
    nx.set_node_attributes(graph, time_conv, 'times_conved')
    nx.set_node_attributes(graph, 0, 'global_info')
    
    try:
        for node_dance in graph.nodes(data=True):
            graph.nodes[node_dance[0]]['global_info'] = dancers_at_hub(node_dance[1]['x'], qrounded)
    except Exception as e:
        print(e)
        pdb.set_trace() 

    # node_to_color_black = tuple([0.0, 1.0])
    id_to_color = [z for z,y in graph.nodes(data=True) if node_to_color_black(y['x'])]
    for node_c in id_to_color:
            graph.nodes[node_c]['colors'] = 'k'

    id_of_goal1 = [z for z,y in graph.nodes(data=True) if node_to_color_green(y['x'], qrounded)]
    id_of_goal2 = [z for z,y in graph.nodes(data=True) if node_to_color_red(y['x'], qrounded)]
    # pdb.set_trace()
    for node in id_of_goal1:
        graph.nodes[node]['colors'] = 'g'
    for node in id_of_goal2:
        graph.nodes[node]['colors'] = 'r'  
    # for edge in 


    if len(id_of_goal2) > 0:
        if len(id_of_goal1) > 0:
            print("PROBLEM PROBLEM PROBLEM!!!!!!!!")
            pdb.set_trace()
        print(len(graph.nodes), len(id_to_color), len(id_of_goal1), len(id_of_goal2), success_now)
    
    
    newfname =  str(entry[1][0]) + str(entry[1][1]) + str(entry[1][2])
    fname = newfname + '_' + fileName + '_noAgentPos_single_sim' + '.pickle'

    ''' PUNEET: TODO: TEST'''
    fil =  open(folder_graph+fname, 'wb')
    pickle.dump(graph, fil)   
    fil.close() 
    ''' TEMP BREAK'''



    return "Processed", fileName

def main():
    folder = './graphsage_results/CDC/multiple_agent_env_results/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'metadata.csv'
    folder_graph = './CDC/allenvsmore/'
    new_metadata_file = folder_graph + 'metadata.csv'
    graph_metaFile = 'graphMetadata.csv'
    meta_arr = []
    metadata = pd.read_csv(metadata_file) 
    metadata.site_qualities=metadata.site_qualities.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(literal_eval)
    metadata.site_qualities=metadata.site_qualities.apply(lambda x: tuple(x))
    metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    # pdb.set_trace()
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    
    for some_id, entry in enumerate(df.iterrows()):
        # if entry[1].iloc[2] != 10: # or some_id==0:
        #     continue
        print(entry)
        files_to_process = [(fileName, site_conv, time_conv, entry, folder, folder_graph, graph_metaFile) 
                            for fileName, site_conv, time_conv in zip(entry[1].iloc[3], entry[1].iloc[5], entry[1].iloc[6])]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file, *file_info) for file_info in files_to_process]
            for future in as_completed(futures):
                try:
                    result, fileName = future.result()
                    print(f"{result}: {fileName}")
                except Exception as exc:
                    print(f"File generated an exception: {exc}")
        
        # for fileinfo in files_to_process:
        #     process_file(fileinfo[0], fileinfo[1], fileinfo[2], fileinfo[3], fileinfo[4], fileinfo[5], fileinfo[6])
            # break
        # break
if __name__ == "__main__":
    main()

