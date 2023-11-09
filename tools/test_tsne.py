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

folder = './graphs/random_test2/'
print(os.listdir('.'))
files = os.listdir(folder)
files = [file for file in files if file.endswith('.pickle')]
files = np.sort(files)
metadata_file = folder + 'metadata.csv'
OUTPUT_DIM = 128
metadata = pd.read_csv(metadata_file)

# @torch.no_grad()
# def test():
#     model.eval()
#     out = model(data.x, data.edge_index).cpu()

#     # clf = LogisticRegression()
#     # clf.fit(out[data.train_mask], data.y[data.train_mask])

#     # val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
#     # test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

#     # return val_acc, test_acc
#     return out
import random
import torch
import pickle
import pdb 
from torch_geometric.utils.convert import from_networkx
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
matplotlib.use("Agg") 
import random 
from torch_geometric.loader import NeighborLoader
import numpy as np
# from sim import params
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from params import *

counter = 0
for row in metadata.iterrows():
    if counter == 10:
        # counter += 1
        continue
        
    else:        
        # pdb.set_trace()
        fname = folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
        fil =  open(fname, 'rb')
        dat = pickle.load(fil)
        fil.close()
        num_neighbors_sampling = [5] * 2
        train_percent = 0.8

        draw = False
        print('1')
        
        
        print('2')
        # k = 1000
        # sampled_nodes = random.sample(list(dat.nodes), k)
        # dat = dat.subgraph(sampled_nodes)
        # pdb.set_trace()
        sampled_graph = dat
        # pos = nx.spring_layout(sampled_graph)#, pos=positions)
        # pos = nx.nx_pydot.graphviz_layout(sampled_graph)
        ''' NEATO'''
        fil_pos = open('poses_test.pickle', 'rb')
        pos = pickle.load(fil_pos)
        fil_pos.close()
        print('3')
        sizes_ = [10.0*a for a in nx.get_node_attributes(sampled_graph, 'sz').values()]
        positions = [a for a in nx.get_node_attributes(sampled_graph, 'x').values()]
        colors_ = [a for a in nx.get_node_attributes(sampled_graph, 'colors').values()]
        alphas = [0.2 if a=='r' else 1.0 for a in colors_]
        sizes_ = [s*0.1 if r=='r' else s for s,r in zip(sizes_, colors_)]
        sizes_ = [s*10.0 if r=='g' else s for s,r in zip(sizes_, colors_)]
        colors_plotly = ['rgb(255, 0, 0)' if r=='r' else 'rgb(0, 0, 255)' if r=='b' else 'rgb(0, 0, 0)' for r in colors_]
        # plt.figure(0)
        # pdb.set_trace()
        if draw:
            plt.figure()
            try:
                nx.draw_networkx_nodes(sampled_graph, pos, node_size=sizes_, node_color=colors_, alpha=alphas) #pos=nx.spring_layout(sampled_graph),
            except:
                pdb.set_trace()
            for edge in sampled_graph.edges(data='weight'):
                nx.draw_networkx_edges(sampled_graph, pos, edgelist=[edge], width=0.2*edge[2])
            plt.savefig("test_random2.png")
            print(sampled_graph.number_of_edges())
        
        # pdb.set_trace()
        data = GraphDataset(fname,  train_percent, num_neighbors_sampling).pyg_graph
 
        # print(len(data))
        tsne2 = TSNE(2)
        two_d2 = tsne2.fit_transform(data.x)
        # # pdb.set_trace()
        c = Counter(zip(two_d2[:,0],two_d2[:,1]))
        weights = [100 + 0.01*c[(i,j)] for i,j in zip(two_d2[:,0],two_d2[:,1])]
        fig = px.scatter(x=two_d2[:, 0], y=two_d2[:, 1],  size=sizes_, opacity=alphas)#, color=colors_plotly)
        fig.write_html('original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.html')
        '''
        x=tuple(data.edge_index.numpy())
        start = x[0]
        end = x[1]
        edges = [tuple((a,b,c)) for a,b,c in zip(start,end, data.weight.numpy())]
        nodes = two_d2
        # try:
        lines = [tuple((nodes[e[0]], nodes[e[1]], e[2])) for e in edges]

        lines2 = []
        for line in lines:
            if np.all([l1==l2 for l1,l2 in zip(line[0],line[1])]):
                continue
            else:
                lenline = np.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)
                if lenline > 1:
                    # print(line)
                    lines2.append(np.array([line[0][0], line[1][0], line[0][1], line[1][1], line[2]]))

                # print('len line:', np.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2))
        print(len(lines2))

        for line in lines2:
            # pdb.set_trace()
            fig.add_trace(px.line(line[:2], line[2:4], width=10+line[4]).data[0])

        fig.write_html('with_edges_original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.html')
        # plt.savefig("with_edges.png") '''
        # pdb.set_trace()
