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
from sklearn.cluster import HDBSCAN
import seaborn as sns

folder = './graphs/new_test/'
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

def generate_distinct_rgb_colors(n):
    """
    Generate an array of n distinct RGB colors with values in the [0, 1] range.

    Args:
        n (int): The number of distinct RGB colors to generate.

    Returns:
        list: A list of n distinct RGB color tuples with values in the [0, 1] range.
    """
    colors = []
    for i in range(n):
        hue = i / n
        rgb_color = hsv_to_rgb(hue, 1, 1)
        normalized_color = tuple(channel / 255.0 for channel in rgb_color)
        colors.append(normalized_color)
    return colors

def hsv_to_rgb(h, s, v):
    """
    Convert HSV (Hue, Saturation, Value) color to RGB color.

    Args:
        h (float): Hue value in the range [0, 1].
        s (float): Saturation value in the range [0, 1].
        v (float): Value (brightness) in the range [0, 1].

    Returns:
        tuple: A tuple representing the RGB color.
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return (int(r * 255), int(g * 255), int(b * 255))

counter = 0
for row in metadata.iterrows():
    if counter == 100:
        # counter += 1
        continue
        # pass
    else:        
        # pdb.set_trace()
        fname = folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
        fil =  open(fname, 'rb')
        dat = pickle.load(fil)
        fil.close()
        num_neighbors_sampling = [5] * 2
        train_percent = 0.8

        draw = False
        # print('1')


        # k = 1000
        # sampled_nodes = random.sample(list(dat.nodes), k)
        # dat = dat.subgraph(sampled_nodes)
        # pdb.set_trace()
        sampled_graph = dat
        # pos = nx.spring_layout(sampled_graph)#, pos=positions)
        # pos = nx.nx_pydot.graphviz_layout(sampled_graph)
        ''' NEATO'''
        # fil_pos = open('poses_test.pickle', 'rb')
        # pos = pickle.load(fil_pos)
        # fil_pos.close()

        # positions_ = nx.spring_layout(sampled_graph)  

        sizes_ = [10.0*a for a in nx.get_node_attributes(sampled_graph, 'sz').values()]
        positions_ = [a for a in nx.get_node_attributes(sampled_graph, 'x').values()]
        colors_ = [a for a in nx.get_node_attributes(sampled_graph, 'colors').values()]
        times = [t for t in nx.get_node_attributes(sampled_graph, 'time').values()]
        successes = [success for success in nx.get_node_attributes(sampled_graph, 'success').values()]
        alphas_ = [0.2 if color=='r' else 1.0 for color in colors_]
        sizes_ = [size * 0.1 if color == 'r' else size * 10.0 if color == 'g' else size for size, color in zip(sizes_, colors_)]
        colors_plotly = colors_
        sum_red = np.sum(1 if color=='r' else 0 for color in colors_)
        if sum_red == 0:
             continue
        # pdb.set_trace()
        # colors_plotly = ['red' if color=='r' else 'blue' if color=='b' else 'green' if color=='g' else 'black' for color in colors_]
        # plt.figure(0)
        # pdb.set_trace()
        # if draw:
        #     plt.figure()
        #     try:
        #         nx.draw_networkx_nodes(sampled_graph, pos, node_size=sizes_, node_color=colors_, alpha=alphas_) #pos=nx.spring_layout(sampled_graph),
        #     except:
        #         pdb.set_trace()
        #     for edge in sampled_graph.edges(data='weight'):
        #         nx.draw_networkx_edges(sampled_graph, pos, edgelist=[edge], width=0.2*edge[2])
        #     plt.savefig("test_random2.png")
        #     print(sampled_graph.number_of_edges())

        # pdb.set_trace()

        data = GraphDataset(dat,  train_percent, num_neighbors_sampling).pyg_graph
        # data = GraphDataset(fname,  train_percent, num_neighbors_sampling).pyg_graph
        print('data loaded')




        # print(len(data))
        # tsne2 = TSNE(n_components=2)
        # two_d2 = tsne2.fit_transform(data.x)

        # tsne3 = TSNE(n_components=3, n_jobs=20)
        # d3Trans = tsne3.fit_transform(data.x)

        # fil_tsne3 = open(folder +'single_tsne_for_dbscan.pickle', 'wb')
        # pickle.dump(d3Trans,fil_tsne3)
        # fil_tsne3.close()
                
        fil_tsne3 = open(folder +'single_tsne_for_dbscan.pickle', 'rb')
        d3Trans = pickle.load(fil_tsne3)
        fil_tsne3.close()

        # dict_twod = {'x': list(two_d2[:,0]), 'y': list(two_d2[:,1]), 'colors': colors_plotly}
        dict_3d = {'x': list(d3Trans[:,0]), 'y': list(d3Trans[:,1]), 'z': list(d3Trans[:,2]), 'colors': colors_plotly, 'times': times, 'successes':successes}
        dict_3d_PoI = dict_3d
        
        
        # colors = [
        #             (0, 0, 1),       # Blue
        #             (0, 0.5, 0),     # Green
        #             (1, 0, 0),       # Red
        #             (0, 0.75, 0.75), # Cyan
        #             (1, 0, 1),       # Magenta
        #             (1, 1, 0),       # Yellow
        #             (0, 0, 0),       # Black
        #             (1, 1, 1),       # White
        #             (0.5, 0.5, 0.5), # Gray
        #             (0.8, 0.8, 0.8), # Light Gray
        #             (0, 0, 0.5),     # Navy Blue
        #             (0.53, 0.81, 0.98), # Sky Blue
        #             (0, 0.98, 0.6),  # Lime Green
        #             (0.94, 0.5, 0.5), # Light Coral
        #             (1, 0.84, 0),     # Gold
        #             (0.5, 0, 0.5),   # Purple
        #             (1, 0.75, 0.8),   # Pink
        #             (1, 0.65, 0),     # Orange
        #             (0.59, 0.27, 0.08), # Brown
        #             (0.5, 0.5, 0),    # Olive
        #             (0, 0.5, 0.5),    # Teal
        #             (0.86, 0.44, 0.58), # Orchid
        #             (0.18, 0.55, 0.34), # Sea Green
        #             (0.12, 0.56, 1),   # Dodger Blue
        #             (0.18, 0.31, 0.31), # Dark Slate Gray
        #             (0.29, 0, 0.51),   # Indigo
                    # ]
        
        list_of_succ_fail = []
        cluster_centroids = []
        cluster_times = []
        cluster_time_mean = []
        cluster_time_variance = []


        # data_2d = pd.DataFrame(dict_twod)
        # dbscan = DBSCAN(eps=5, min_samples=50)
        # labels = dbscan.fit_predict(data_2d[['x', 'y']])

        # data_2d['cluster'] = labels
        # sns.set(style="whitegrid")
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x='x', y='y', hue='cluster', palette='viridis', data=data_2d, s=100, legend='full')

        # plt.title('DBSCAN Clustering Results')
        # plt.show()

        data_3d = pd.DataFrame(dict_3d)
        # pdb.set_trace()
        # Perform DBSCAN clustering
        dbscan = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=5.0)

        labels_3d = dbscan.fit_predict(data_3d[['x', 'y', 'z']])
        colors = generate_distinct_rgb_colors(len(np.unique(labels_3d)))
        nodes_to_plot_blue_edges = []
        nodes_to_plot_green_edges = []   
        nodes_to_plot_black_edges = []   
        nodes_to_plot_red_edges = []
        nodes_to_plot_cyan_edges = []

        data_3d['cluster'] = labels_3d

        marker = {'b':'.', 'k':'x', 'c': 'x', 'r': '^', 'g': 's'}
        for k, (x,y,z,c, cluster) in enumerate(zip(dict_3d['x'],dict_3d['y'],dict_3d['z'], dict_3d['colors'], data_3d['cluster'].values)): 
            # for k, (x,y,z,c) in enumerate(zip(dict_3d['x'],dict_3d['y'],dict_3d['z'], dict_3d['colors'])):
                if c == 'b':
                    nodes_to_plot_blue_edges.append([x,y,z,colors[cluster],marker[c]])
                if c == 'g':
                    nodes_to_plot_green_edges.append([x,y,z,colors[cluster],marker[c]])
                if c == 'k':
                    nodes_to_plot_black_edges.append([x,y,z,colors[cluster],marker[c]])
                elif c == 'r':
                    nodes_to_plot_red_edges.append([x,y,z,colors[cluster],marker[c]])
                elif c == 'c':
                    nodes_to_plot_cyan_edges.append([x,y,z,colors[cluster],marker[c]])

        nodes_to_plot_blue_edges = np.array(nodes_to_plot_blue_edges)
        nodes_to_plot_green_edges = np.array(nodes_to_plot_green_edges)
        nodes_to_plot_black_edges = np.array(nodes_to_plot_black_edges)    
        nodes_to_plot_red_edges = np.array(nodes_to_plot_red_edges)
        nodes_to_plot_cyan_edges = np.array(nodes_to_plot_cyan_edges)
        

        # marker_styles = {'red': 'o', 'blue': '^', 'green': 's', 'black': 'x'}
        # label_names = {'red': 'intermediate', 'blue': 'success', 'green': 'failure', 'black': 'start'}
        # cmap = matplotlib.cm.get_cmap('Spectral')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        pdb.set_trace()
        # ax.scatter(nodes_to_plot_blue_edges[:,0],nodes_to_plot_blue_edges[:,1],nodes_to_plot_blue_edges[:,2], c= nodes_to_plot_blue_edges[:,3], marker=nodes_to_plot_blue_edges[0,4])
        ax.scatter3D(nodes_to_plot_black_edges[:,0].tolist(),nodes_to_plot_black_edges[:,1].tolist(),nodes_to_plot_black_edges[:,2].tolist(), marker=nodes_to_plot_black_edges[0,4], c=nodes_to_plot_black_edges[:,3].tolist())
        ax.scatter3D(nodes_to_plot_green_edges[:,0].tolist(),nodes_to_plot_green_edges[:,1].tolist(),nodes_to_plot_green_edges[:,2].tolist(), marker=nodes_to_plot_green_edges[0,4], c=nodes_to_plot_green_edges[:,3].tolist())
        ax.scatter3D(nodes_to_plot_red_edges[:,0].tolist(),nodes_to_plot_red_edges[:,1].tolist(),nodes_to_plot_red_edges[:,2].tolist(), marker=nodes_to_plot_red_edges[0,4], c=nodes_to_plot_red_edges[:,3].tolist())
        ax.scatter3D(nodes_to_plot_cyan_edges[:,0].tolist(),nodes_to_plot_cyan_edges[:,1].tolist(),nodes_to_plot_cyan_edges[:,2].tolist(), marker=nodes_to_plot_cyan_edges[0,4], c=nodes_to_plot_cyan_edges[:,3].tolist())


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('DBSCAN Clustering Results (3D)')
        plt.show()
