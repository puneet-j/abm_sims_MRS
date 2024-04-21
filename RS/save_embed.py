
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
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from graphDat import GraphDataset
from params import ENVIRONMENT_BOUNDARY_X, COMMIT_THRESHOLD
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import ConvexHull
MAX_DIST=ENVIRONMENT_BOUNDARY_X[-1]

torch.manual_seed(42)

class GraphEncoderWithResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoderWithResidual, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels*4)
        self.conv2 = SAGEConv(hidden_channels*4, hidden_channels*2)
        self.conv3 = SAGEConv(hidden_channels*2, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.lin2 = nn.Linear(int(hidden_channels/2), out_channels)
        # Linear transformation to match dimensions for residual connection
        self.shortcut = nn.Linear(in_channels, out_channels)
        # self.sm = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = F.dropout(F.relu(self.lin1(x)), p=0.2)
        x = self.lin2(x)
        # Applying shortcut and adding it to the output of conv3
        identity = self.shortcut(identity)
        x += identity  # Element-wise addition
        # x = self.sm(x)
        return x

# class GraphEncoderWithResidual(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphEncoderWithResidual, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels*2)
#         self.conv2 = SAGEConv(hidden_channels*2, hidden_channels)
#         # self.conv3 = SAGEConv(hidden_channels, out_channels)
#         self.lin = nn.Linear(hidden_channels, out_channels)
#         # Linear transformation to match dimensions for residual connection
#         self.shortcut = nn.Linear(in_channels, out_channels)
#         # self.sm = nn.Softmax(dim=1)

#     def forward(self, x, edge_index):
#         identity = x
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         # x = self.conv3(x, edge_index)
#         x = self.lin(x)
#         # Applying shortcut and adding it to the output of conv3
#         identity = self.shortcut(identity)
#         x += identity  # Element-wise addition
#         # print('before: ', x[0])
#         # print(np.shape(x))
#         # x = self.sm(x)
#         # print('after: ', x[0])
#         return x



# class GraphEncoderWithResidual(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphEncoderWithResidual, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels*2)
#         self.conv2 = SAGEConv(hidden_channels*2, hidden_channels)
#         self.conv3 = SAGEConv(hidden_channels, out_channels)
#         # self.lin = nn.Linear(hidden_channels, out_channels)
#         # Linear transformation to match dimensions for residual connection
#         self.shortcut = nn.Linear(in_channels, out_channels)
#         # self.sm = nn.Softmax(dim=1)

#     def forward(self, x, edge_index):
#         identity = x
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         x = self.conv3(x, edge_index)
#         # x = self.lin(x)
#         # Applying shortcut and adding it to the output of conv3
#         identity = self.shortcut(identity)
#         x += identity  # Element-wise addition
#         # print('before: ', x[0])
#         # print(np.shape(x))
#         # x = self.sm(x)
#         # print('after: ', x[0])
#         return x
    
    
def node_to_color_black(node):
    if node[3] == 0.0:
        return True
    else:
        return False

def node_to_color_green(node, quals):
    # pdb.set_trace()
    nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    qs = [0]*4
    # pdb.set_trace()
    for d in dancers:
        ii = quals.index(d[1])
        qs[ii] += 1

    if np.max(qs) > COMMIT_THRESHOLD*nA:
        id = np.argmax(qs)
        if quals[id] == np.max(quals):
            return True
        else:
            return False
    else:
        return False
    
def node_to_color_red(node, quals):
    # pdb.set_trace()
    nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    qs = [0]*4
    # pdb.set_trace()
    for d in dancers:
        ii = quals.index(d[1])
        qs[ii] += 1

    if np.max(qs) > COMMIT_THRESHOLD*nA:
        id = np.argmax(qs)
        if quals[id] != np.max(quals):
            return True
        else:
            return False
    else:
        return False

# def node_to_color_black(node):
#     if node[9] == 0.0:
#         return True
#     else:
#         return False

# def node_to_color_green(node, quals):
#     # pdb.set_trace()
#     nA = np.sum([1 for _ in node[0::10]])
#     dancers = [(1, q) for a, q in zip(node[0::10], node[9::10]) if a == 1]
#     qs = [0]*4
#     # pdb.set_trace()
#     for d in dancers:
#         ii = quals.index(d[1])
#         qs[ii] += 1

#     if np.max(qs) > COMMIT_THRESHOLD*nA:
#         id = np.argmax(qs)
#         if quals[id] == np.max(quals):
#             return True
#         else:
#             return False
#     else:
#         return False
    
# def node_to_color_red(node, quals):
#     # pdb.set_trace()
#     nA = np.sum([1 for _ in node[0::10]])
#     dancers = [(1, q) for a, q in zip(node[0::10], node[9::10]) if a == 1]
#     qs = [0]*4
#     # pdb.set_trace()
#     for d in dancers:
#         ii = quals.index(d[1])
#         qs[ii] += 1

#     if np.max(qs) > COMMIT_THRESHOLD*nA:
#         id = np.argmax(qs)
#         if quals[id] != np.max(quals):
#             return True
#         else:
#             return False
#     else:
#         return False
    
def get_color(node, quals):
    red = node_to_color_red(node, quals)
    green = node_to_color_green(node, quals)
    black = node_to_color_black(node)
    if red is True:
        return 'r'
    elif green is True:
        return 'g'
    elif black is True:
        return 'k'
    else:
        return 'b'

# def get_full_qual(q):
#     while len(q) < 4:
#         q.append(0.0)
#     return q

    

def get_complete_poses(a):
    arr = [[MAX_DIST]*2]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = list(a[i])
    # pdb.set_trace()
    return arr

def get_complete_quals(a):
    arr = [[0]*2]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = a[i]
    # pdb.set_trace()
    return arr

def get_complete_global(a):
    arr = [0]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = a[i]
    # pdb.set_trace()
    return arr

folder_graph = './RS/finished_trials/'
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
arr = dict()
arr1 = []
arrpos = []
arrqual = []
arrcol = []
arr_global_info = []
arr_num_agents = []
# metadata = pd.read_csv('./graphsage_results/CDC/multiple_agent_env_results/metadata.csv')
for file in files[:-1]:
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    # pdb.set_trace()
    nA = file.split(')')[-1][:2]
    if nA[-1] == '_':
        num_agents = int(file.split(')')[-1][0])
    else:
        num_agents = int(file.split(')')[-1][:2])

    for node in G.nodes(data=True):
        # pdb.set_trace()
        if node[1]['x'] in arr:
            continue
        else:
            # pdb.set_trace()
            new_color = get_color(node[1]['x'], node[1]['quals'])
            arrpos.append(get_complete_poses(node[1]['poses']))
            arrqual.append(get_complete_quals(node[1]['quals']))
            arrcol.append(new_color)
            arr_global_info.append(get_complete_global(node[1]['global_info']))
            arr_num_agents.append(num_agents)
            # arr2.append([node[0], node[1]['x'], node[1]['colors'], node[1]['quals'], node[1]['poses'], node[1]['global_info'], num_agents])
            # pdb.set_trace()
            arr1.append(list(node[1]['x']))#+list(node[1]['global_info']))
            # arr1.append(list(node[1]['x'])+list(node[1]['global_info']))
            # arr.append([list(node[1]['x'])+[a for a in node[1]['global_info']]])
            arr[node[1]['x']] = node[1]['colors']

# pdb.set_trace()
encoder_dict = torch.load('./RS/Finished_Large_LinNew_encoder_state_dict3.pth')
in_channels = 40
hidden_channels = 100
out_channels = 3
GAE = GraphEncoderWithResidual(in_channels, hidden_channels, out_channels)
GAE.load_state_dict(encoder_dict)
emb = []
arrtosave = []
# for i in arr1: 
for i in arr1: 
    # pdb.set_trace()
    emb.append(GAE(torch.tensor([i], dtype=torch.float), torch.empty((2,0), dtype=torch.int64)).detach().tolist()[0])
    arrtosave.append(i)
# pdb.set_trace()
df = pd.DataFrame(emb, columns=['x', 'y', 'z'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_emb.csv')

df = pd.DataFrame(arrpos, columns=['pose1', 'pose2', 'pose3', 'pose4'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_poses.csv')

df = pd.DataFrame(arrqual, columns=['qual1', 'qual2', 'qual3', 'qual4'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_quals.csv')


df = pd.DataFrame(arr_global_info, columns=['1', '2', '3', '4'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_global_info.csv')

df = pd.DataFrame(arrcol, columns=['colors'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_colors.csv')

df = pd.DataFrame(arr_num_agents, columns=['num_agents'])
df.to_csv('./RS/Finished_Large_LinNew_ALLENVS_AGENTS_num_agents.csv')

# pdb.set_trace()

# temp = []
# for i in arr.values():
#     temp.append([i])

# df2 = pd.DataFrame(arr2, columns=['id', 'x', 'colors','quals', 'poses', 'global_info'])
# df2.to_csv('./CDC/ALLENVS_AGENTS_arr.csv')

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
