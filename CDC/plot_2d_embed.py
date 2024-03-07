
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv, GraphConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd 
import pickle
import pdb
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, subgraph, k_hop_subgraph
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'tools'))
from graphDat import GraphDataset
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import ConvexHull

torch.manual_seed(42)

class GraphEncoderWithResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoderWithResidual, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, out_channels)
        self.lin = nn.Linear(hidden_channels * 2, out_channels)
        # Linear transformation to match dimensions for residual connection
        self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = self.conv3(x, edge_index)
        x = self.lin(x)
        # Applying shortcut and adding it to the output of conv3
        identity = self.shortcut(identity)
        x += identity  # Element-wise addition
        return x
    
encoder_dict = torch.load('./CDC/LinearLayernoAllGood_global_LotsInfo_Colored_Res_30ep_0p01_encoder_state_dict3.pth')
# decoder_dict = torch.load('./CDC/decoder_state_dict3.pth')
in_channels = 42
hidden_channels = 10
out_channels = 3
GAE = GraphEncoderWithResidual(in_channels, hidden_channels, out_channels)
GAE.load_state_dict(encoder_dict)
# GAE.decoder.load_state_dict(decoder_dict)


folder_graph = './graphs/CDC/lotsOfInfo/notAllSuccess/'
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
arr = dict()
arr1 = []
arr2 = []
for file in files[:-1]:
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    # pdb.set_trace()
    for node in G.nodes(data=True):
        
        # pdb.set_trace()
        if node[1]['x'] in arr:
            continue
        else:
            arr2.append([node[0], node[1]['x'], node[1]['colors'], node[1]['quals'], node[1]['poses'], node[1]['global_info']])
            # pdb.set_trace()
            arr1.append(list(node[1]['x'])+list(node[1]['global_info']))
            # arr.append([list(node[1]['x'])+[a/10.0 for a in node[1]['global_info']]])
            arr[node[1]['x']] = node[1]['colors']

emb = []
arrtosave = []
for i in arr1: 
    # pdb.set_trace()
    emb.append(GAE(torch.tensor([i], dtype=torch.float), torch.empty((2,0), dtype=torch.int64)).detach().tolist()[0])
    arrtosave.append(i)
df = pd.DataFrame(emb, columns=['x', 'y'])
df.to_csv('./CDC/LinearLayernotAllSuccess_GlobalTrained_AllInfoPlot_emb_trained_on_lots_info.csv')

# temp = []
# for i in arr.values():
#     temp.append([i])

df2 = pd.DataFrame(arr2, columns=['id', 'x', 'colors','quals', 'poses', 'global_info'])
df2.to_csv('./CDC/LinearLayernotAllSuccess_GlobalTrained_AllInfoPlot_arr_trained_on_lots_info.csv')
