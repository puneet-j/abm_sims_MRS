import torch
import networkx as nx 
from torch.utils.data import Dataset, DataLoader
import pickle
import pdb 
import numpy as np
import pandas as pd 
STATES_LIST = ['RECRUIT', 'ASSESS', 'TRAVEL_HOME_TO_RECRUIT', 'TRAVEL_SITE', 'OBSERVE', 'EXPLORE', 'TRAVEL_HOME_TO_OBSERVE']
STATES = {'RECRUIT':0.0/6.0, 'ASSESS':1.0/6.0, 'TRAVEL_HOME_TO_RECRUIT':2.0/6.0, 'TRAVEL_SITE':3.0/6.0, 
          'OBSERVE':4.0/6.0, 'EXPLORE':5.0/6.0, 'TRAVEL_HOME_TO_OBSERVE':6.0/6.0}
TIME_LIMIT = 2500
SUCC_LIMIT = 0.95
# def get_complete_global():
#     return
class GraphDataset(Dataset):
    def __init__(self, folder, file_paths, flag, timesucc):#, meta):
        self.file_paths = file_paths
        self.folder = folder
        self.flag = flag
        self.timesucc = timesucc
        # self.times_df = pd.read_csv(self.folder + 'times.csv')
        # self.dffiles = pd.read_csv(self.folder + 'files.csv')
        # self.meta = meta
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

    def get_timeclass(self, t):
        if t > TIME_LIMIT:
            return 0
        else:
            return 1
        
    def get_succclass(self, s):
        if s <= SUCC_LIMIT:
            return 0
        else:
            return 1
    
    def get_succtimeclass(self, s, t, fl):
        
        if s <= SUCC_LIMIT and t > TIME_LIMIT:
            return 0 # slow fail
        elif s > SUCC_LIMIT and t > TIME_LIMIT:
            return 1 # slow success
        elif s <= SUCC_LIMIT and t <= TIME_LIMIT:
            return 2 # fast fail
        elif s > SUCC_LIMIT and t <= TIME_LIMIT:
            return 3 # fast success
        else:
            print(s, t, fl)
            pdb.set_trace()
    # def get_succclass(self, s):
    #     if s < 0.25:
    #         return 0
    #     elif s >= 0.25 and s < 0.5:
    #         return 1
    #     elif s >= 0.5 and s < 0.75:
    #         return 2
    #     else:
    #         return 3
        
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
        with open(self.folder + self.file_paths[idx], 'rb') as f:
            G = pickle.load(f)
        
        # pdb.set_trace()
        # print(list(G[0][1]['weight']))
        # files = ['alledges', 'allsucc', 'alltime', 'edgesList', 'nodeIDs']
        # w
        # print(G.nodes[0]['class'])
        # pdb.set_trace()
        # classes = torch.tensor([self.meta[G.nodes[node]] for node in G.nodes], dtype=torch.long)
        # global_info = torch.tensor([self.get_complete_global(G.nodes[node]['global_info'], len(G.nodes[node]['x'])) for node in G.nodes], dtype=torch.float)
        # node_features = torch.tensor([self.oneHotToState(G.nodes[node]['x']) for node in G.nodes], dtype=torch.float)
        # for node in G.nodes:
        #     print(node)
        # pdb.set_trace()
        try:
            node_features = torch.tensor([node[1]['x'] for node in G.nodes(data=True)], dtype=torch.float)
        except:
            return self.file_paths[idx], self.file_paths[idx], self.file_paths[idx]
        # pdb.set_trace()
        # For edge features, assuming 'weight' attribute exists for each edge
        # Creating a tensor for edge indices and another for edge features
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        # print(G.nodes[0])
        # for i in self.meta: print(i); break
        # time_feature = torch.tensor([self.meta[node[1]['x']] for node in G.nodes(data=True)], dtype=torch.float)
        #time_feature = torch.tensor([node[1]['time'] for node in G.nodes(data=True)], dtype=torch.float)
        edge_features = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)
        # try:
        #     times = torch.tensor([self.get_timeclass(node[1]['AvgTime']) for node in G.nodes(data=True)], dtype=torch.long)
        # except Exception as e:
        #     print(e)
        #     print(self.folder)
        #     print(self.file_paths[idx])
        #     print(G.nodes[0])
        #     # pdb.set_trace()
        
        # # successes = torch.tensor([self.get_succclass(node[1]['AvgSucc']) for node in G.nodes(data=True)], dtype=torch.long)
        # if self.flag == 'train':
        #     if self.timesucc == 'time':
        #         # timesucc = torch.tensor([node[1]['AvgTime'] for node in G.nodes(data=True)], dtype=torch.float)
        #         # timesucc = torch.tensor([node[1]['times_conved']  for node in G.nodes(data=True)], dtype=torch.float)
        #         # try:
        #         # p
        #         # ref = self.dffiles.loc[self.file_paths[idx][:-7]+'.csv']['ref']
        #         # print(ref)
        #         # timesucc = torch.tensor(self.times_df[ref].tolist(), dtype = torch.float)
        #         timesucc = torch.tensor(self.times_df[self.file_paths[idx]].tolist(), dtype = torch.float)
        #         # except:
        #         #     pdb.set_trace()
        #     else:
        #         # timesucc = torch.tensor([node[1]['AvgSucc'] for node in G.nodes(data=True)], dtype=torch.float)
        #         timesucc = torch.tensor([node[1]['success'] for node in G.nodes(data=True)], dtype=torch.float)

        #         # pdb.set_trace()
        #         # print(timesucc)
        #         # exit()
        #     times = torch.empty((1,1), dtype=torch.float) #torch.tensor([self.get_timeclass(node[1]['AvgTime']) for node in G.nodes(data=True)], dtype=torch.long)
        #     successes = torch.empty((1,1), dtype=torch.float) #torch.tensor([self.get_succclass(node[1]['AvgSucc']) for node in G.nodes(data=True)], dtype=torch.long)
        # else:
        #     if self.timesucc == 'time':
        #         # timesucc = torch.tensor([node[1]['AvgTime'] for node in G.nodes(data=True)], dtype=torch.float)
        #         # timesucc = torch.tensor([node[1]['times_conved'] for node in G.nodes(data=True)], dtype=torch.float)
        #         timesucc = torch.tensor(self.times_df[self.file_paths[idx]].tolist(), dtype = torch.float)
        #         # ref = self.dffiles.loc[self.file_paths[idx][:-7]+'.csv']['ref']
        #         # print(ref)
        #         # timesucc = torch.tensor(self.times_df[ref].tolist(), dtype = torch.float)
        #     else:
        #         # timesucc = torch.tensor([node[1]['AvgSucc'] for node in G.nodes(data=True)], dtype=torch.float)
        #         timesucc = torch.tensor([node[1]['success'] for node in G.nodes(data=True)], dtype=torch.float)            # pdb.set_trace()
        #     # timesucc = torch.tensor([(node[1]['AvgSucc'], node[1]['AvgTime']) for node in G.nodes(data=True)], dtype=torch.float)
        #     # timesucc = torch.tensor([(node[1]['success'], node[1]['times_conved'] - node[1]['time']) for node in G.nodes(data=True)], dtype=torch.float)
        #     # print(timesucc)
        #     # exit()
        #     # pdb.set_trace()
        #     times = torch.empty((1,1), dtype=torch.float) # torch.tensor([self.get_timeclass(node[1]['times_conved']) for node in G.nodes(data=True)], dtype=torch.long)
        #     successes =torch.empty((1,1), dtype=torch.float) # torch.tensor([self.get_succclass(node[1]['success']) for node in G.nodes(data=True)], dtype=torch.long)

        # # if self.flag == 'train':
        # #     timesucc = torch.tensor([self.get_succtimeclass(node[1]['AvgSucc'], node[1]['AvgTime'], f) for node in G.nodes(data=True)], dtype=torch.long)
        # #     times = torch.tensor([self.get_timeclass(node[1]['AvgTime']) for node in G.nodes(data=True)], dtype=torch.long)
        # #     successes = torch.tensor([self.get_succclass(node[1]['AvgSucc']) for node in G.nodes(data=True)], dtype=torch.long)
        # # else:
        # #     timesucc = torch.tensor([self.get_succtimeclass(node[1]['success'], node[1]['times_conved'], f) for node in G.nodes(data=True)], dtype=torch.long)
        # #     times = torch.tensor([self.get_timeclass(node[1]['times_conved']) for node in G.nodes(data=True)], dtype=torch.long)
        # #     successes = torch.tensor([self.get_succclass(node[1]['success']) for node in G.nodes(data=True)], dtype=torch.long)


        # quals = [node[1]['quals'] for node in G.nodes(data=True)]
        # nA = [node[1]['agents'] for node in G.nodes(data=True)]
        # # Adjacency matrix (optional if you use edge_index and edge_features directly)
        # # adj_matrix = nx.to_numpy_matrix(G)
        # # adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float)

        # # Return node features, edge index, and edge features
        return node_features, edge_index, edge_features#, times, successes, quals, nA, timesucc #, time_feature#, classes
        # return torch.cat((global_info, node_features), dim=1), edge_index#, edge_features
        # return torch.cat((global_info, node_features), dim=1), edge_index#, edge_features