import networkx as nx
import numpy as np
import scipy.sparse as sp
import os
import time
from collections import defaultdict
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
import csv


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

def node_to_color_black(node):
    if node[3] == 0.0:
        return True
    else:
        return False

def node_to_color_green(node, quals):
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    q = 0
    # pdb.set_trace()
    for d in dancers:
        if d[1] == np.max(quals):
            q += 1
    
    if q > 5:
        return True
    else:
        return False
    
def node_to_color_red(node, quals):
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    q = 0
    # pdb.set_trace()
    for d in dancers:
        if d[1] == np.min(quals):
            q += 1
    if q > 5:
        return True
    else:
        return False

def get_current_state(astates, asites, aposes, sposes, squals):
    try:
        current_state_main = []
        for id, (site, state, pos) in enumerate(sorted(zip(asites, astates, aposes), key = lambda x: (squals, ))):
            current_state = []
            if not(site is None):
                whichSite = getSiteID(site)
                # current_state.append(1.0*pos[0]/MAX_DIST)
                # current_state.append(1.0*pos[1]/MAX_DIST)
                current_state.append(1.0*STATES[state])
                # current_state.append(1.0*STATES[state]/len(STATES))
                if whichSite==-1:
                    current_state.append(0.0)
                    current_state.append(0.0)
                    current_state.append(0.0)
                else:
                    current_state.append(1.0*sposes[whichSite][0]/MAX_DIST)
                    current_state.append(1.0*sposes[whichSite][1]/MAX_DIST)
                    current_state.append(1.0*squals[whichSite])
            
            else:
                # current_state.append(1.0*pos[0]/MAX_DIST)
                # current_state.append(1.0*pos[1]/MAX_DIST)
                current_state.append(1.0*STATES[state])
                # current_state.append(1.0*STATES[state]/len(STATES))
                current_state.append(1.0)
                current_state.append(1.0)
                current_state.append(0.0)
            current_state_main.append(current_state)
        current_state_main = sorted(current_state_main, key = lambda x: (1.0 - x[3], x[0], x[1]**2 + x[2]**2))
        retVal = []
        for cstate in current_state_main:
            for c in cstate:
                retVal.append(c)
    except Exception as e:
        print(e)
        pdb.set_trace()  
    # pdb.set_trace()
 
    # pdb.set_trace()
    return tuple(round_function(retVal,3))

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


    

def parse_row(r, quals, poses):
    
    agent_states = literal_eval(r[4])
    agent_sites = literal_eval(r[-1])
    agent_positions = literal_eval(r[2])
    # pdb.set_trace()
    node = get_current_state(agent_states, agent_sites, agent_positions, poses, quals)
    return node

def process_file(fileName, site_conv, time_conv, entry, folder, folder_graph, graph_metaFile):

# While line = F.readline():
# 	p_line = parse(line)	

# 	node = tuple(p_line(4), p_line(6), …. ) // some function

# 	//if node not in N:
# 	//	N[node] = indCounter
# 	//	indCounter++

	
# 	if PrevNode != nil:
# 		G[PrevNode][node]++

# // convert G to tensor
    quals = entry[1].iloc[0]
    poses = entry[1].iloc[1]
    prev = time.time()*1000.0
    success_now =  0.0 if np.isnan(site_conv) else site_conv/max(quals)
    # print("time now 1: 0")
    G = defaultdict(defaultdict)
    prev_node = None
    # now = time.time()*1000.0
    # print("time now 1: ", now - prev)
    with open(folder+fileName, newline='\n') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)
        filereader = csv.reader(csvfile, delimiter=',')
        if has_header:
            next(filereader)
        # prev, now = now, time.time()*1000.0
        # print("time now 2: ", now - prev)
        
        for row in filereader:
            
            node = str(parse_row(row, quals, poses))
            #pdb.set_trace()
            if node not in G:
                G[node] = defaultdict(int)

            if prev_node is not None:
                G[prev_node][node] += 1
                
            prev_node = node
        #print(G)
        # pdb.set_trace()
        # prev, now = now, time.time()*1000.0
        # print("time now 3: ", now - prev)
        # newfname =  str(entry[1].iloc[0]) + str(entry[1].iloc[1]) + str(entry[1].iloc[2])
        # fname = newfname + '_' + fileName + '_noAgentPos_single_sim' + '.pickle'
        # fil =  open(folder_graph+fname, 'wb')
        # pickle.dump(G, fil)   
        # fil.close() 
        # prev, now = now, time.time()*1000.0
        # print("time now 4: ", now - prev)
        return G #"Processed", fileName


def main():
    folder = './graphsage_results/CDC/'
    files = os.listdir(folder)
    files = [file for file in files if file.endswith('.csv') and file.startswith('1')]
    files = np.sort(files)
    # data_files = []
    metadata_file = folder + 'metadata.csv'
    folder_graph = './graphs/CDC/testaks/'
    # new_metadata_file = folder_graph + 'metadata.csv'
    graph_metaFile = 'graphMetadata.csv'
    # meta_arr = []
    metadata = pd.read_csv(metadata_file) 
    metadata.site_qualities=metadata.site_qualities.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(literal_eval)
    metadata.site_positions=metadata.site_positions.apply(lambda x: tuple([tuple(a) for a in x]))
    df = metadata.groupby(by=['site_qualities', 'site_positions', 'num_agents'], as_index=False).agg(lambda x: x.tolist())
    
    for some_id, entry in enumerate(df.iterrows()):
        # if entry[1].iloc[2] != 10: # or some_id==0:
        #     continue
        #prev = time.time()*1000.0
        print("time now 1: 0")
        print(entry)
        list_of_dicts = []
        files_to_process = [(fileName, site_conv, time_conv, entry, folder, folder_graph, graph_metaFile) 
                            for fileName, site_conv, time_conv in zip(entry[1].iloc[3], entry[1].iloc[5], entry[1].iloc[6])]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file, *file_info) for file_info in files_to_process]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    list_of_dicts.append(result)
                    # print(f"{result}: {fileName}")
                except Exception as exc:
                    print(f"File generated an exception: {exc}")
        # for fileinfo in files_to_process:
        #     result = process_file(fileinfo[0], fileinfo[1], fileinfo[2], fileinfo[3], fileinfo[4], fileinfo[5], fileinfo[6])
        #     list_of_dicts.append(result)
        #     break
        # break
        #prev, now = now, time.time()*1000.0
        #print("time now 2: ", now - prev)
        newfname =  str(entry[1].iloc[0]) + str(entry[1].iloc[1]) + str(entry[1].iloc[2])
        fname = newfname + '_' + '_noAgentPos_single_sim' + '.pickle'
        fil =  open(folder_graph+fname, 'wb')
        pickle.dump(list_of_dicts, fil)   
        fil.close() 
        #now = time.time()*1000.0
        #print("time now 3: ", now - prev)

        # for fileinfo in files_to_process:
        #     result = process_file(fileinfo[0], fileinfo[1], fileinfo[2], fileinfo[3], fileinfo[4], fileinfo[5], fileinfo[6])
        #     break
        # break

if __name__ == "__main__":
    main()

