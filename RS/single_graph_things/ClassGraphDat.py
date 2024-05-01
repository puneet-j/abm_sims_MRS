import torch
import networkx as nx 
from torch.utils.data import Dataset, DataLoader
import pickle
import pdb 
import numpy as np
STATES_LIST = ['RECRUIT', 'ASSESS', 'TRAVEL_HOME_TO_RECRUIT', 'TRAVEL_SITE', 'OBSERVE', 'EXPLORE', 'TRAVEL_HOME_TO_OBSERVE']
STATES = {'RECRUIT':0.0/6.0, 'ASSESS':1.0/6.0, 'TRAVEL_HOME_TO_RECRUIT':2.0/6.0, 'TRAVEL_SITE':3.0/6.0, 
          'OBSERVE':4.0/6.0, 'EXPLORE':5.0/6.0, 'TRAVEL_HOME_TO_OBSERVE':6.0/6.0}
# def get_complete_global():
#     return
class GraphDataset(Dataset):
    def __init__(self, folder, file_paths):
        self.file_paths_edges = file_paths
        self.folder = folder
        # files = ['alledges', 'allsucc', 'alltime', 'edgesList', 'nodeIDs']

    def __len__(self):
        return len(self.file_paths)

    def get_complete_global(self, a, l):
        arr = [0]*4
        nA = np.ceil(l/4.0)
        for i in range(0,len(a)):
            # pdb.set_trace()
            arr[i] = a[i]/nA
        # print(arr)
        # pdb.set_trace()
        return arr
    # CLIST = ["Slow/Success", "Fast/Success", "Slow/Failure", "Fast/Failure"]
    # 
    # def get_class(self, cl):
    #     x = [0]*4
    #     if cl == "Slow/Success":
    #         x[0] = 1
    #     elif cl == "Fast/Success":
    #         x[1] = 1
    #     elif cl == "Slow/Failure":
    #         x[2] = 1
    #     elif cl == "Fast/Failure":
    #         x[3] = 1
    #     return x
    # def oneHotToState(self, st):
    #     newx = []
    #     i=0
    #     while i < 10:
    #         state = np.round(STATES[STATES_LIST[st[10*i:10*i+7].index(1)]], 3)
    #         newx.append(state)
    #         for j in range(10*i+7, 10*i+10):
    #             newx.append(st[j])
    #         i += 1
    #     return newx

    def __getitem__(self, idx):
        # with open(self.folder + self.file_paths[idx], 'rb') as f:
        #     G = pickle.load(f)
        
        # files = ['alledges', 'allsucc', 'alltime', 'edgesList', 'nodeIDs']
        # w
        # print(G.nodes[0]['class'])
        classes = torch.tensor([G.nodes[node]['classes'].index(1) for node in G.nodes], dtype=torch.long)
        # global_info = torch.tensor([self.get_complete_global(G.nodes[node]['global_info'], len(G.nodes[node]['x'])) for node in G.nodes], dtype=torch.float)
        # node_features = torch.tensor([self.oneHotToState(G.nodes[node]['x']) for node in G.nodes], dtype=torch.float)
        node_features = torch.tensor([G.nodes[node]['x'] for node in G.nodes], dtype=torch.float)

        # pdb.set_trace()
        # For edge features, assuming 'weight' attribute exists for each edge
        # Creating a tensor for edge indices and another for edge features
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        # edge_features = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)

        # Adjacency matrix (optional if you use edge_index and edge_features directly)
        # adj_matrix = nx.to_numpy_matrix(G)
        # adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float)

        # Return node features, edge index, and edge features
        return node_features, edge_index, classes
        # return torch.cat((global_info, node_features), dim=1), edge_index#, edge_features
        # return torch.cat((global_info, node_features), dim=1), edge_index#, edge_features