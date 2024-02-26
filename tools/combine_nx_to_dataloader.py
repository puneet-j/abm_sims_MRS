import torch
import networkx as nx
import numpy as np
import pdb 
import pandas as pd 
from ast import literal_eval


def get_SFH(arr):
    nonoriented = np.sum([1 if a > 3 else 0 for a in arr[0::4]])
    if np.sum([1 if a == 0 else 0 for a in arr[0::4]]) > 3:
        if np.sort(np.unique(arr[3::4]))[-1] != 0:
            bestsite = arr[3::4].count(np.sort(np.unique(arr[3::4]))[-1])
        # pdb.set_trace()
        if len(np.sort(np.unique(arr[3::4]))) > 1:
            if np.sort(np.unique(arr[3::4]))[-2] != 0:
                badsite = arr[3::4].count(np.sort(np.unique(arr[3::4]))[-2])
            else:
                badsite = 0
    # pdb.set_trace()
    if nonoriented == 10:
        return 'H'
    elif bestsite >= 3:
        return  'S'   
    elif badsite >= 3:
        # print('F')
        return 'F'
    else:
        return 'I'
    


# Example usage
graph_list = []
folder_graph = './graphs/CDC/'
metadata_file = folder_graph + 'metadata.csv'
succTimeFiles = folder_graph + 'graphMetadata.csv'
import os
import pickle
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
# filescsv = [file for file in files if file.endswith('.csv')]
# filescsv.remove('metadata.csv')
for file in files:
    # 
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    graph_list.append(G)


# met = pd.read_csv(folder_graph+'metadata.csv', header=0)
# met.qualities = met.qualities.apply(literal_eval)
# reverse_dict = {}
# qual_dict = {}
# # pdb.set_trace()
# for me, id in zip(met.qualities.values, met['Unnamed: 0']):
#     for m in me:
#         qual_dict[m] = id
#         if id in reverse_dict:
#             reverse_dict[id] = [m]
#         else:
#             reverse_dict[id].append(m)

# arr = []
# for filecsv in filescsv[:-1]:
#     fil = pd.read_csv(folder_graph+filecsv, header=0)
#     try:
#         fil.nodeID = fil.nodeID.apply(literal_eval)
#         [arr.append([nid, ms, mt]) for 
#             nid,ms, mt in zip(list(fil.nodeID.values), list(fil.mean_success.values), list(fil.mean_conv_time.values))]
#     except Exception as e:
#         pdb.set_trace()

# maindf = pd.DataFrame(arr, columns=['nodeID', 'meanSuccess', 'meanTime'])
# maindf = maindf.groupby(['nodeID'], as_index=False).mean()
# # pdb.set_trace()
# maindf['SFHI'] = maindf.nodeID.apply(lambda x: get_SFH(x))
# pdb.set_trace()
# maindf.to_csv('metadata_graphsage_nodes.csv')



# pdb.set_trace()

# dataset = GraphDataset(graph_list)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# pdb.set_trace()
# Save the graph list for later use, similar to the previous example
torch.save(graph_list, 'small_graphs.pth')

# # Example of iterating over the DataLoader in a training loop
# for node_features, edge_index, edge_features in dataloader:
#     # Use these in your GAE model
#     pass