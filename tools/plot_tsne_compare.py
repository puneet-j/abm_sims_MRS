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

fil_concat = open(folder +'all_combined_graphs_poses.pickle', 'rb')
dat = pickle.load(fil_concat)
fil_concat.close()

fil_concat = open(folder +'all_combined_graphs_colors.pickle', 'rb')
allCols = pickle.load(fil_concat)
fil_concat.close()

fil_concat = open(folder +'all_combined_graphs_succttc.pickle', 'rb')
succttc = pickle.load(fil_concat)
fil_concat.close()

colors_plotly = allCols
succ = succttc[0]
ttc = succttc[1]

# print('data loaded')
# # pdb.set_trace()
# # dat = dat[:500000,:]

for perp in range(1,11):
    values_to_plot = 10*perp
    tsne3 = TSNE(n_components=3, verbose=1, perplexity=10*values_to_plot, n_iter=5000,n_jobs=-1)
    d3Trans = tsne3.fit_transform(dat)
    print('tsne done, perp: ', values_to_plot)


# fil_tsne2 = open(folder +'combined_tsne2.pickle', 'wb')
# pickle.dump(d2Trans,fil_tsne2)
# fil_tsne2.close()

    fil_tsne3 = open(folder +'combined_tsne3_'+str(values_to_plot*10)+'.pickle', 'wb')
    pickle.dump(d3Trans,fil_tsne3)
    fil_tsne3.close()

exit()

arr = ['5', '10', '15', '20', '25', '30']
for values_to_plot in arr:
    fil_tsne3 = open(folder +'combined_tsne3_'+values_to_plot+'.pickle', 'rb')
    d3Trans = pickle.load(fil_tsne3)
    fil_tsne3.close()

    # dict_twod = {'x': list(two_d2[:,0]), 'y': list(two_d2[:,1]), 'colors': colors_plotly}
    dict_3d = {'x': list(d3Trans[:,0]), 'y': list(d3Trans[:,1]), 'z': list(d3Trans[:,2]), 'colors': colors_plotly}


    # dict_3d_PoI = {key: value for key, value in dict_3d.items() if dict_3d['colors'] != 'red'}
    dict_3d_PoI = {'x':[], 'y':[], 'z':[], 'colors':[]}
    # dict_3d_PoI
    for x,y,z,color in zip(dict_3d['x'],dict_3d['y'],dict_3d['z'],dict_3d['colors']):
        # pdb.set_trace()
        if color != 'r':
            dict_3d_PoI['x'].append(x)
            dict_3d_PoI['y'].append(y)
            dict_3d_PoI['z'].append(z)
            dict_3d_PoI['colors'].append(color)
    # dict_3d_PoI['colors'] = [value for value in dict_3d['colors'] if value != 'red']
    # pdb.set_trace()

    dict_3d_PoI = dict_3d

    # from mpl_toolkits.mplot3d import Axes3D

    print('dicts loaded')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    # scatter = ax.scatter(dict_3d_PoI['x'], dict_3d_PoI['y'], dict_3d_PoI['z'], c=dict_3d_PoI['colors'])#, alpha=alphas_, s=sizes_)
    # scatter = ax.scatter(dict_3d['x'], dict_3d['y'], dict_3d['z'], c=dict_3d['colors'], alpha=alphas_, s=sizes_)

    # Labeling axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # print('scatter plot done')

    nodes_to_plot_blue_edges = []
    nodes_to_plot_green_edges = []   
    nodes_to_plot_black_edges = []   
    nodes_to_plot_red_edges = []
    nodes_to_plot_cyan_edges = []

    for k, (x,y,z,c) in enumerate(zip(dict_3d_PoI['x'],dict_3d_PoI['y'],dict_3d_PoI['z'], dict_3d_PoI['colors'])): 
    # for k, (x,y,z,c) in enumerate(zip(dict_3d['x'],dict_3d['y'],dict_3d['z'], dict_3d['colors'])):
        if c == 'b':
            nodes_to_plot_blue_edges.append(tuple([x,y,z]))
        if c == 'g':
            nodes_to_plot_green_edges.append(tuple([x,y,z]))
        if c == 'k':
            nodes_to_plot_black_edges.append(tuple([x,y,z]))
        elif c == 'r':
            nodes_to_plot_red_edges.append(tuple([x,y,z]))
        elif c == 'c':
            nodes_to_plot_cyan_edges.append(tuple([x,y,z]))

    nodes_to_plot_blue_edges = np.array(nodes_to_plot_blue_edges)
    nodes_to_plot_green_edges = np.array(nodes_to_plot_green_edges)
    nodes_to_plot_black_edges = np.array(nodes_to_plot_black_edges)    
    nodes_to_plot_red_edges = np.array(nodes_to_plot_red_edges)
    nodes_to_plot_cyan_edges = np.array(nodes_to_plot_cyan_edges)

    from scipy.spatial import ConvexHull
    # pdb.set_trace()
    blueHull = ConvexHull(nodes_to_plot_blue_edges)
    greenHull = ConvexHull(nodes_to_plot_green_edges)
    blackHull = ConvexHull(nodes_to_plot_black_edges)
    redHull = ConvexHull(nodes_to_plot_red_edges)
    cyanHull = ConvexHull(nodes_to_plot_cyan_edges)


    for simplex in blueHull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        simplex_points = nodes_to_plot_blue_edges[simplex]
        ax.plot(nodes_to_plot_blue_edges[simplex, 0], nodes_to_plot_blue_edges[simplex, 1], nodes_to_plot_blue_edges[simplex, 2], 'b-', alpha = 0.1)

    for simplex in greenHull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        simplex_points = nodes_to_plot_green_edges[simplex]
        ax.plot(nodes_to_plot_green_edges[simplex, 0], nodes_to_plot_green_edges[simplex, 1], nodes_to_plot_green_edges[simplex, 2], 'g-', alpha = 0.3)


    for simplex in blackHull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        simplex_points = nodes_to_plot_black_edges[simplex]
        ax.plot(nodes_to_plot_black_edges[simplex, 0], nodes_to_plot_black_edges[simplex, 1], nodes_to_plot_black_edges[simplex, 2], 'k-', alpha = 0.5)

    for simplex in cyanHull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        simplex_points = nodes_to_plot_cyan_edges[simplex]
        ax.plot(nodes_to_plot_cyan_edges[simplex, 0], nodes_to_plot_cyan_edges[simplex, 1], nodes_to_plot_cyan_edges[simplex, 2], 'c-', alpha = 0.8)

    for simplex in redHull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        simplex_points = nodes_to_plot_red_edges[simplex]
        ax.plot(nodes_to_plot_red_edges[simplex, 0], nodes_to_plot_red_edges[simplex, 1], nodes_to_plot_red_edges[simplex, 2], 'r-', alpha = 0.3)

    # plt.axis('off')

    # pdb.set_trace()

    # plot all edges between blue nodes. 


    ax.set_aspect('equal', adjustable='box')

    plt.savefig('combined_3d_perp='+values_to_plot+'.png')

    plt.show()
# exit()
