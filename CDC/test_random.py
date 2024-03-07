
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
# folder_graph = './graphs/CDC/'

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels * 2, out_channels)  # Final Layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

# class GraphDecoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphDecoder, self).__init__()
#         # Assuming the encoded features are to be decoded back to original feature size
#         self.conv1 = SAGEConv(out_channels, hidden_channels )
#         self.conv2 = SAGEConv(hidden_channels , hidden_channels * 2)  # Mimic encoder complexity
#         self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
#         self.conv4 = SAGEConv(hidden_channels, in_channels)  # Additional hidden layer to output size

#     def forward(self, z, edge_index):
#         # pdb.set_trace()
#         z = F.relu(self.conv1(z, edge_index))
#         z = F.relu(self.conv2(z, edge_index))
#         z = F.relu(self.conv3(z, edge_index))
#         z = self.conv4(z, edge_index)
#         return z

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        # self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x_masked = x #* mask
        z = self.encoder(x_masked, edge_index)
        # x_reconstructed = self.decoder(z, edge_index)
        return z # x_reconstructed, z
    


encoder_dict = torch.load('./CDC/30ep_0p01_encoder_state_dict3.pth')
# decoder_dict = torch.load('./CDC/decoder_state_dict3.pth')
in_channels = 40
hidden_channels = 64
out_channels = 3
GAE = MaskedGraphAutoencoder(in_channels, hidden_channels, out_channels)
GAE.encoder.load_state_dict(encoder_dict)
# GAE.decoder.load_state_dict(decoder_dict)


folder_graph = './graphs/CDC/test3/'
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
arr = set()
for file in files[:-1]:
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    # pdb.set_trace()
    for node in G.nodes(data=True):
        # pdb.set_trace()
        try:
            arr.add(node[1]['x'])
        except:
            continue
        # if node[1]['x'] in arr:
        #     continue
        # else:
        #     arr[node[1]['x']] = node[1]['colors']

emb = []
for i in arr: 
    # pdb.set_trace()
    emb.append(GAE.encoder(torch.tensor([i], dtype=torch.float), torch.empty((2,0), dtype=torch.int64)).detach().tolist())


np.save('./CDC/RANDOM_nodes_to_plot.npy', arr)
np.save('./CDC/RANDOM_emb.npy', emb)

# arr = np.load('30ep_0p01_nodes_to_plot.npy', allow_pickle=True).item()
# emb = np.load('30ep_0p01_emb.npy', allow_pickle=True)
# pdb.set_trace()
x = []
y = []
z = []
c = []
a = []
# pdb.set_trace()

# pdb.set_trace()
for e,i in zip(emb,arr):
    # pdb.set_trace()
    if i != 'b':
        x.append(e[0][0])
        y.append(e[0][1])
        z.append(e[0][2])
        # c.append(i)
        # if i == 'r' or i == 'g':
        #     print('success or fail')
        # a.append(1.0)

# nodes_to_plot_blue_edges = []
# for e,i in zip(emb,arr.values()):
#     if i == 'b':
#         nodes_to_plot_blue_edges.append(tuple([e[0][0],e[0][1],e[0][2]]))
# blueHull = ConvexHull(np.array(nodes_to_plot_blue_edges))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x, y, z, marker = '.')#, color=c)#, alpha = a)#, marker = '.', color=c)
# for simplex in blueHull.simplices:
#     try:
#         simplex = np.append(simplex, simplex[0])  # Close the loop
#         simplex_points = [nodes_to_plot_blue_edges[n] for n in simplex]
#         xs = [a[0] for a in simplex_points]
#         ys = [a[1] for a in simplex_points]
#         zs = [a[2] for a in simplex_points]
#         ax.plot(xs, ys, zs, 'b-', alpha = 0.1)
#     except Exception as e:
#         print(e)
#         pdb.set_trace()
    
plt.show()
