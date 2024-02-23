
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd 
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
import os
import pickle 
# from tools.combine_nx_to_dataloader import GraphDataset
torch.manual_seed(42)
from mpl_toolkits import mplot3d

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
    
class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels )
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, out_channels)  # Final Layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        # Assuming the encoded features are to be decoded back to original feature size
        self.conv1 = SAGEConv(out_channels, hidden_channels )
        self.conv2 = SAGEConv(hidden_channels , hidden_channels )  # Mimic encoder complexity
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, in_channels)  # Additional hidden layer to output size

    def forward(self, z, edge_index):
        # pdb.set_trace()
        z = F.relu(self.conv1(z, edge_index))
        z = F.relu(self.conv2(z, edge_index))
        # z = F.relu(self.conv3(z, edge_index))
        z = self.conv4(z, edge_index)
        return z

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x_masked = x #* mask
        z = self.encoder(x_masked, edge_index)
        x_reconstructed = self.decoder(z, edge_index)
        return x_reconstructed, z
    
encoder_dict = torch.load('Bigencoder_state_dict3.pth')
decoder_dict = torch.load('Bigdecoder_state_dict3.pth')
in_channels = 40
hidden_channels = 64
out_channels = 3
GAE = MaskedGraphAutoencoder(in_channels, hidden_channels, out_channels)
GAE.encoder.load_state_dict(encoder_dict)
GAE.decoder.load_state_dict(decoder_dict)


folder_graph = './graphs/graphsage_graph/'
# metadata_file = folder_graph + 'metadata.csv'
# succTimeFiles = folder_graph + 'graphMetadata.csv'
files = os.listdir(folder_graph)
# files = [file for file in files if file.endswith('.pickle')]
files = [file for file in files if file.endswith('.pickle')]
# filescsv.remove('metadata.csv')
arr = set()
for file in files[:-1]:
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    for node in G.nodes(data=True):
        # pdb.set_trace()
        arr.add(node[1]['x'])
        # try:
        #     # fil.nodeID = fil.nodeID.apply(literal_eval)
        #     [arr.append([nid, ms, mt]) for 
        #         nid,ms, mt in zip(list(fil.nodeID.values), list(fil.mean_success.values), list(fil.mean_conv_time.values))]
        # except Exception as e:
        #     pdb.set_trace()

# maindf = pd.DataFrame(arr, columns=['nodeID', 'meanSuccess', 'meanTime'])
# maindf = maindf.groupby(['nodeID'], as_index=False).mean()
# pdb.set_trace()
# maindf['SFHI'] = maindf.nodeID.apply(lambda x: get_SFH(x))

# df = pd.read_csv('metadata_graphsage_nodes.csv')
# maindf.nodeID = maindf.nodeID.apply(literal_eval)
# inp = []
# col = []
# emb = []
# COLORS = {'H':'b', 'S':'g', 'F':'r', 'I':'b'}
# for node in maindf.iterrows():
#     # pdb.set_trace()
    
    
#     if COLORS[node[1][-1]] != 'g':
#         # print(COLORS[node[1][-1]] != 'g')
#         col.append(COLORS[node[1][-1]])
#         inp.append(torch.tensor([[node[1][1]]]))
emb = []
for i in arr: 
    # pdb.set_trace()
    emb.append(GAE.encoder(torch.tensor([i], dtype=torch.float), torch.empty((2,0), dtype=torch.int64)).detach().tolist())

x = []
y = []
z = []
c = []
a = []
for e in emb:
    # pdb.set_trace()
# for e, cl in zip(emb,col):
    x.append(e[0][0])
    y.append(e[0][1])
    z.append(e[0][2])
    # c.append(cl)
    # a.append(1.0 if cl == 'r' else 0.1)


# val  = [x,y,z,c,a]

# # np.save('pltarray.npy', np.array(val))



# # val = np.load('pltarray.npy')
# # pdb.set_trace()
# x = [float(v) for v in val[0]]
# y = [float(v) for v in val[1]]
# z = [float(v) for v in val[2]]
# c = [v for v in val[3]]
# a = [float(v) for v in val[4]]
# # a = val[4]
# ss = []
# for col in c:
#     # a.append(0.9 if col=='r' else 0.1)
#     ss.append(10 if col=='r' else 2)
# print(s)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x, y, z, marker = '.')# alpha = a, marker = '.', color=c)
plt.show()
