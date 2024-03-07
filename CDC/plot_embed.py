
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
        # Linear transformation to match dimensions for residual connection
        self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # Applying shortcut and adding it to the output of conv3
        identity = self.shortcut(identity)
        x += identity  # Element-wise addition
        return x
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
    

encoder_dict = torch.load('./CDC/noAllGood_global_LotsInfo_Colored_Res_30ep_0p01_encoder_state_dict2.pth')
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
df = pd.DataFrame(emb, columns=['x', 'y', 'z'])
df.to_csv('./CDC/notAllSuccess_GlobalTrained_AllInfoPlot_emb_trained_on_lots_info.csv')

# temp = []
# for i in arr.values():
#     temp.append([i])

df2 = pd.DataFrame(arr2, columns=['id', 'x', 'colors','quals', 'poses', 'global_info'])
df2.to_csv('./CDC/notAllSuccess_GlobalTrained_AllInfoPlot_arr_trained_on_lots_info.csv')

# df2 = pd.DataFrame(temp, columns=['color'])#, 'x', 'quals', 'poses', 'global_info'])
# df2.to_csv('./CDC/Colored_Res_30ep_0p01_arr.csv')

# pdb.set_trace()
# np.save('./CDC/Colored_Res_30ep_0p01_nodes_to_plot.npy', arr)
# np.save('./CDC/Colored_Res_30ep_0p01_emb.npy', emb)

# arr = np.load('./CDC/30ep_0p01_nodes_to_plot.npy', allow_pickle=True).item()
# emb = np.load('./CDC/30ep_0p01_emb.npy', allow_pickle=True)
# pdb.set_trace()
# x = []
# y = []
# z = []
# c = []
# a = []

# for i in arr.values():

# # pdb.set_trace()

# # pdb.set_trace()
# for e,i in zip(emb,arr.values()):
#     # pdb.set_trace()
#     # if i != 'b':
#         x.append(e[0])
#         y.append(e[1])
#         z.append(e[2])
#         c.append(i)
#         if i == 'b':
#              a.append(0.1)
#         else:
#              a.append(1.0)
#         # if i == 'r' or i == 'g':
#         #     print('success or fail')
#         # a.append(1.0)

# # nodes_to_plot_blue_edges = []
# # for e,i in zip(emb,arr.values()):
# #     if i == 'b':
# #         nodes_to_plot_blue_edges.append(tuple([e[0][0],e[0][1],e[0][2]]))
# # blueHull = ConvexHull(np.array(nodes_to_plot_blue_edges))

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter3D(x, y, z, marker = '.', color=c, alpha = a)#, marker = '.', color=c)
# # for simplex in blueHull.simplices:
# #     try:
# #         simplex = np.append(simplex, simplex[0])  # Close the loop
# #         simplex_points = [nodes_to_plot_blue_edges[n] for n in simplex]
# #         xs = [a[0] for a in simplex_points]
# #         ys = [a[1] for a in simplex_points]
# #         zs = [a[2] for a in simplex_points]
# #         ax.plot(xs, ys, zs, 'b-', alpha = 0.1)
# #     except Exception as e:
# #         print(e)
# #         pdb.set_trace()
    
# plt.show()
