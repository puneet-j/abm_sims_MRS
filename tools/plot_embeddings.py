import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pandas as pd
import pdb
from ast import literal_eval
import copy
import pickle
from sklearn.preprocessing import StandardScaler
from myclasses import GraphDataset
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
# https://towardsdatascience.com/how-to-create-a-graph-neural-network-in-python-61fd9b83b54e
import torch
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
# plt.ion()   # interactive mode
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
import os.path as osp
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import plotly.express as px
from collections import Counter
from torch_geometric.utils import to_networkx

folder = './graphs/new/'
print(os.listdir('.'))
files = os.listdir(folder)
files = [file for file in files if file.endswith('.pickle')]
files = np.sort(files)
metadata_file = folder + 'metadata.csv'
OUTPUT_DIM = 128
metadata = pd.read_csv(metadata_file)

@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index).cpu()

    # clf = LogisticRegression()
    # clf.fit(out[data.train_mask], data.y[data.train_mask])

    # val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    # test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    # return val_acc, test_acc
    return out

for row in metadata.iterrows():
    # pdb.set_trace()
    fname = str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
    num_neighbors_sampling = [5] * 2
    train_percent = 0.8
    data = GraphDataset(folder + fname,  train_percent, num_neighbors_sampling).pyg_graph
    # break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = data.to(device, 'x', 'edge_index')
    # print('using device', device)
    model = GraphSAGE(
        in_channels=len(data.x[0]),
        hidden_channels=128,
        num_layers=2,
        out_channels=OUTPUT_DIM,
    ).to(device)

    # model = GraphSAGE()
    model.load_state_dict(torch.load('./models/' + str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pt'))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


   

    embeddings = test()
    # pdb.set_trace()
    print(len(embeddings))
    tsne = TSNE(2)
    two_d = tsne.fit_transform(embeddings)
    c = Counter(zip(two_d[:,0],two_d[:,1]))
    weights = [100 + 0.01*c[(i,j)] for i,j in zip(two_d[:,0],two_d[:,1])]
    # pdb.set_trace()
    fig1 = px.scatter(x=two_d[:,0], y=two_d[:,1], size= weights)
    fig1.write_html('./figures/'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.html')
    plt.figure()
    plt.scatter(two_d[:, 0], two_d[:, 1], s=weights, alpha=0.4)
    # plt.savefig('./figures/'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.fig')
    plt.savefig('./figures/'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.png')
    
    print(len(data.x.cpu()))
    tsne2 = TSNE(2)
    two_d2 = tsne.fit_transform(data.x.cpu())
    c = Counter(zip(two_d2[:,0],two_d2[:,1]))
    weights = [100 + 0.01*c[(i,j)] for i,j in zip(two_d2[:,0],two_d2[:,1])]
    fig2 = px.scatter(x=two_d2[:,0], y=two_d2[:,1], size=weights)
    fig2.write_html('./figures/original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.html')
    plt.figure()
    plt.scatter(two_d2[:, 0], two_d2[:, 1],  s=weights, alpha=0.4)
    # plt.savefig('./figures/original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.fig')
    plt.savefig('./figures/original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.png')
    pdb.set_trace()
