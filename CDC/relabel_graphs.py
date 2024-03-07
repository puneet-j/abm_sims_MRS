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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch 
import shutil

def label_file(fl, folder_now, folder_to_save):
        # fil =  open(folder_now+fl, 'rb')
        # G = pickle.load(fil)
        # fil.close()
        # pdb.set_trace()
        vals = fl.split(')(')[0].strip('(').split(',')
        quals = [float(q) for q in vals]
        if np.max(quals) - np.min(quals) > 0.5 or np.min(quals) < 0.2:
             return "Skipped", fl
        else:
             shutil.copyfile(folder_now+fl, folder_to_save+fl)
             return "Processed", fl
        # global_info = torch.tensor([G.nodes[node]['global_info'] for node in G.nodes], dtype=torch.float)
        # node_features = torch.tensor([G.nodes[node]['x'] for node in G.nodes], dtype=torch.float)
        # # pdb.set_trace()
        # # For edge features, assuming 'weight' attribute exists for each edge
        # # Creating a tensor for edge indices and another for edge features
        # edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        # edge_features = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)


        # df = pd.read_csv(flcsv, header=0)
        # pdb.set_trace()
        # nx.set_node_attributes(graph, colors, 'colors')

        # try:
        #     for node_c in id_to_color:
        #         graph.nodes[node_c[0]]['colors'] = 'k'
        # except:
        #     # print(Exception)
        #     pdb.set_trace()


        #     fil =  open(folder_graph+'_'+file, 'wb')
        #     pickle.dump(graph, fil)   
        #     fil.close()
        #     ''' TEMP BREAK'''
        #     # break

def main():
 
 
    folder_graph = './graphs/CDC/lotsOfInfo/'
    folder_graph_2 = './graphs/CDC/lotsOfInfo/notAllSuccess/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle')]# and file.startswith('1')]
    # filescsv = [file for file in files if file.endswith('.csv')]
    # filescsv.remove('metadata.csv')    

    # dists = 
    # sq = 
    files_to_process = [(file, folder_graph, folder_graph_2) 
                            for file in zip(files)]
    for file in files:
        label_file(file, folder_graph, folder_graph_2)
    # with ThreadPoolExecutor(max_workers=40) as executor:
    #     futures = [executor.submit(label_file, *file_info) for file_info in files_to_process]
    #     for future in as_completed(futures):
    #         try:
    #             result, fileName = future.result()
    #             print(f"{result}: {fileName}")
    #         except Exception as exc:
    #             print(f"File generated an exception: {exc}")




main()

