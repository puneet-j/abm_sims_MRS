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
# warnings.filterwarnings("ignore")
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
from bokeh.plotting import figure, from_networkx, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, save, output_file
from mayavi import mlab


folder = './graphs/random_test2/'
print(os.listdir('.'))
files = os.listdir(folder)
files = [file for file in files if file.endswith('.pickle')]
files = np.sort(files)
metadata_file = folder + 'metadata.csv'
OUTPUT_DIM = 128
metadata = pd.read_csv(metadata_file)


import random
import torch
import pickle
import pdb 
from torch_geometric.utils.convert import from_networkx
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
matplotlib.use("MacOSX") 
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

        # positions_ = nx.spring_layout(sampled_graph)  

        sizes_ = [10.0*a for a in nx.get_node_attributes(sampled_graph, 'sz').values()]
        positions_ = [a for a in nx.get_node_attributes(sampled_graph, 'x').values()]
        colors_ = [a for a in nx.get_node_attributes(sampled_graph, 'colors').values()]
        alphas_ = [0.2 if color=='r' else 1.0 for color in colors_]
        sizes_ = [size * 0.1 if color == 'r' else size * 10.0 if color == 'g' else size for size, color in zip(sizes_, colors_)]

        colors_plotly = ['red' if color=='r' else 'blue' if color=='b' else 'green' if color=='g' else 'black' for color in colors_]
        # plt.figure(0)
        # pdb.set_trace()
        if draw:
            plt.figure()
            try:
                nx.draw_networkx_nodes(sampled_graph, pos, node_size=sizes_, node_color=colors_, alpha=alphas_) #pos=nx.spring_layout(sampled_graph),
            except:
                pdb.set_trace()
            for edge in sampled_graph.edges(data='weight'):
                nx.draw_networkx_edges(sampled_graph, pos, edgelist=[edge], width=0.2*edge[2])
            plt.savefig("test_random2.png")
            print(sampled_graph.number_of_edges())

        # pdb.set_trace()
        data = GraphDataset(fname,  train_percent, num_neighbors_sampling).pyg_graph
        print('data loaded')




        # print(len(data))
        tsne2 = TSNE(n_components=2)
        two_d2 = tsne2.fit_transform(data.x)

        tsne3 = TSNE(n_components=3)
        d3Trans = tsne3.fit_transform(data.x)

        dict_twod = {'x': list(two_d2[:,0]), 'y': list(two_d2[:,1]), 'colors': colors_plotly}
        dict_3d = {'x': list(d3Trans[:,0]), 'y': list(d3Trans[:,1]), 'z': list(d3Trans[:,2]), 'colors': colors_plotly}

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot
        scatter = ax.scatter(dict_3d['x'], dict_3d['y'], dict_3d['z'], c=dict_3d['colors'], alpha=alphas_, s=sizes_)

        # Labeling axes
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # plt.axis('off')

        # Show plot
        # plt.show()
        # plt.pause(10)
        # pdb.set_trace()
        nodes_to_plot_edges = []
        # plot all edges between blue nodes. 
        for k, (x,y,z,c) in enumerate(zip(dict_3d['x'],dict_3d['y'],dict_3d['z'], dict_3d['colors'])):
            if c == 'blue':
                nodes_to_plot_edges.append(tuple([x,y,z]))
        # pdb.set_trace()

        edges_to_plot = set()
        for i in range(len(nodes_to_plot_edges)):
            for j in range(i+1, len(nodes_to_plot_edges)):
                edges_to_plot.add(tuple([nodes_to_plot_edges[i], nodes_to_plot_edges[j]]))


        for point in edges_to_plot:
            # for point2 in edges_to_plot[point1]:
                ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], [point[0][2], point[1][2]], 'k--', alpha=0.1)
        plt.show()
        
        '''
        # this plots all edges..?
        x=tuple(data.edge_index.numpy())
        start = x[0]
        end = x[1]
        edges = [tuple((a,b,c)) for a,b,c in zip(start,end, data.weight.numpy())]
        nodes = d3Trans
        pdb.set_trace()
        # try:
        lines = [[[nodes[e[0]][0], nodes[e[1]][0]],
                        [nodes[e[0]][1], nodes[e[1]][1]],
                           [nodes[e[0]][2], nodes[e[1]][2]]] for e in edges]
        # filter out lines by length
        # lines2 = []
        # for line in lines:
        #     if np.all([l1==l2 for l1,l2 in zip(line[0],line[1])]):
        #         continue
        #     else:
        #         lenline = np.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)
        #         if lenline > 1:
        #             # print(line)
        #             lines2.append(np.array([line[0][0], line[1][0], line[0][1], line[1][1], line[2]]))

                # print('len line:', np.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2))
        # print(len(lines2))

        for line in lines:
            # pdb.set_trace()
            ax.plot(line[0], line[1], line[2], color='g', alpha=0.1)
            # fig.add_trace(plt.plot(line[:2], line[2:4], width=10+line[4]).data[0])

        # fig.write_html('with_edges_original_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3])+'.html')
        plt.show()
        plt.savefig("with_edges_3d.png") '''
        # pdb.set_trace()
