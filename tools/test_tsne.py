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

counter = 0
combined_data = np.array([])
# Allcols = []
for row in metadata.iterrows():
    if counter == 10:
        # counter += 1
        pass

    else:        
        # pdb.set_trace()
        try:
            fname = folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
            fil =  open(fname, 'rb')
            dat = pickle.load(fil)
            fil.close()
        except:
            continue
        num_neighbors_sampling = [5] * 2
        train_percent = 0.8

        # print('1')

        # k = 1000
        # sampled_nodes = random.sample(list(dat.nodes), k)
        # dat = dat.subgraph(sampled_nodes)
        # pdb.set_trace()
        sampled_graph = dat

        # positions_ = nx.spring_layout(sampled_graph)  
        # green_counter = np.sum()
        sizes_ = [10.0*a for a in nx.get_node_attributes(sampled_graph, 'sz').values()]
        positions_ = [a for a in nx.get_node_attributes(sampled_graph, 'x').values()]
        colors_ = [a for a in nx.get_node_attributes(sampled_graph, 'colors').values()]
        alphas_ = [0.1 if color=='b' else 1.0 for color in colors_]
        sizes_ = [size * 0.1 if color == 'b' else size * 10.0 for size, color in zip(sizes_, colors_)]
        sum_green = np.sum(1 if color=='g' else 0 for color in colors_)
        sum_black = np.sum(1 if color=='k' else 0 for color in colors_)
        sum_red = np.sum(1 if color=='r' else 0 for color in colors_)
        successes = [a for a in nx.get_node_attributes(sampled_graph, 'success').values()]
        convtimes = [a for a in nx.get_node_attributes(sampled_graph, 'time').values()]
        # print('sum of green: ', sum_green)
        # print('sum of black: ', sum_black)

        # if sum_black < 3:
        #      pdb.set_trace()
        if sum_red == 0:
            #  pass
             continue
            #  pass
        # colors_plotly = ['red' if color=='r' else 'blue' if color=='b' else 'green' if color=='g' else 'black' for color in colors_]
        
        # plt.figure(0)
        # pdb.set_trace()
        draw = False

        if draw:
    
            pos = nx.spring_layout(sampled_graph)#, pos=positions)
            # # pos = nx.nx_agraph.graphviz_layout(sampled_graph)
            # ''' NEATO'''
            # fil_pos = open('poses_test.pickle', 'rb')
            # # pos = pickle.load(fil_pos)
            # # fil_pos.close()
            fil_pos = open(folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_poses.pickle', 'wb')
            pickle.dump(pos,fil_pos)
            # # pos = pickle.load(fil_pos)
            fil_pos.close()
            plt.figure()
            try:
                nx.draw_networkx_nodes(sampled_graph, pos, node_size=sizes_, node_color=colors_, alpha=alphas_) #pos=nx.spring_layout(sampled_graph),
            except:
                pdb.set_trace()
            for edge in sampled_graph.edges(data='weight'):
                nx.draw_networkx_edges(sampled_graph, pos, edgelist=[edge], width=0.2*edge[2])
            plt.savefig(str(row[1][1]) + str(row[1][2]) + str(row[1][3]) +'.png')
            print(sampled_graph.number_of_edges())

        # pdb.set_trace()

        data = GraphDataset(dat,  train_percent, num_neighbors_sampling).pyg_graph
        # pdb.set_trace()
        needed = np.squeeze(np.concatenate([data.x.numpy(),data.time_now.numpy().reshape(len(data.time_now), 1)], axis=1))
        if len(combined_data)<1:# is None:
            combined_data = needed
            Allcols = colors_
            # pdb.set_trace()
            AllSucc = successes
            AllTTC = convtimes
        else:
            combined_data = np.concatenate([combined_data, needed], axis=0)
            print(len(combined_data), 'dat len')
            # pdb.set_trace()
            Allcols += colors_
            AllSucc += successes
            AllTTC += convtimes
            print(len(Allcols), 'col len')
            print(len(AllSucc), 'succ len')
            print(len(AllTTC), 'ttc len')
        # pdb.set_trace()
        print('data loaded')
# pdb.set_trace()
fil_concat = open(folder +'all_combined_graphs_poses.pickle', 'wb')
pickle.dump(combined_data,fil_concat)
fil_concat.close()
fil_concat2 = open(folder +'all_combined_graphs_colors.pickle', 'wb')
pickle.dump(Allcols,fil_concat2)
fil_concat2.close()
fil_concat3 = open(folder +'all_combined_graphs_succttc.pickle', 'wb')
pickle.dump([AllSucc, AllTTC],fil_concat3)
fil_concat3.close()

print('file_saved')

'''
        # Should we not do the embedding again for each simulation???
        # test: compare embedding with all data vs embedding for a few holdouts - are structures consistent?
        # We can predict for newer environments if they are consistent. 
        # enough structure to learn mapping to apply to brand new environments.

        # print(len(data))

        # tsne2 = TSNE(n_components=2)
        # d2Trans = tsne2.fit_transform(data.x)

        # tsne3 = TSNE(n_components=3)
        # d3Trans = tsne3.fit_transform(data.x)
        
        
        # pdb.set_trace()
        # tsne4 = TSNE(n_components=4)
        # d4Trans = tsne4.fit_transform(data.x)


        # fil_tsne2 = open(folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_tsne2.pickle', 'wb')
        # pickle.dump(d2Trans,fil_tsne2)
        # fil_tsne2.close()

        # fil_tsne3 = open(folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_tsne3.pickle', 'wb')
        # pickle.dump(d3Trans,fil_tsne3)
        # fil_tsne3.close()

        # fil_tsne4 = open(folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_tsne4.pickle', 'wb')
        # pickle.dump(d4Trans,fil_tsne4)
        # fil_tsne4.close()

        fil_tsne3 = open(folder +'_'+str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_tsne3.pickle', 'rb')
        d3Trans = pickle.load(fil_tsne3)
        fil_tsne3.close()

        print('tsne done')

        # dict_twod = {'x': list(two_d2[:,0]), 'y': list(two_d2[:,1]), 'colors': colors_plotly}
        dict_3d = {'x': list(d3Trans[:,0]), 'y': list(d3Trans[:,1]), 'z': list(d3Trans[:,2]), 'colors': colors_plotly}

        # dict_3d_PoI = {key: value for key, value in dict_3d.items() if dict_3d['colors'] != 'red'}
        dict_3d_PoI = {'x':[], 'y':[], 'z':[], 'colors':[]}
        # dict_3d_PoI
        for x,y,z,color in zip(dict_3d['x'],dict_3d['y'],dict_3d['z'],dict_3d['colors']):
            # pdb.set_trace()
            if color != 'red':
                dict_3d_PoI['x'].append(x)
                dict_3d_PoI['y'].append(y)
                dict_3d_PoI['z'].append(z)
                dict_3d_PoI['colors'].append(color)
        # dict_3d_PoI['colors'] = [value for value in dict_3d['colors'] if value != 'red']
        # pdb.set_trace()

        from mpl_toolkits.mplot3d import Axes3D

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

        print('scatter plot done')

        nodes_to_plot_edges = []
        nodes_to_plot_green_edges = []   
        nodes_to_plot_black_edges = []   
        for k, (x,y,z,c) in enumerate(zip(dict_3d_PoI['x'],dict_3d_PoI['y'],dict_3d_PoI['z'], dict_3d_PoI['colors'])): 
        # for k, (x,y,z,c) in enumerate(zip(dict_3d['x'],dict_3d['y'],dict_3d['z'], dict_3d['colors'])):
            if c == 'blue':
                nodes_to_plot_edges.append(tuple([x,y,z]))
            if c == 'green':
                nodes_to_plot_green_edges.append(tuple([x,y,z]))
            if c == 'black':
                nodes_to_plot_black_edges.append(tuple([x,y,z]))

        # edges_to_plot = set()
        # for i in range(len(nodes_to_plot_edges)):
        #     for j in range(i+1, len(nodes_to_plot_edges)):
        #         edges_to_plot.add(tuple([nodes_to_plot_edges[i], nodes_to_plot_edges[j]]))


        # edges_to_plot_green = set()
        # for i in range(len(nodes_to_plot_green_edges)):
        #     for j in range(i+1, len(nodes_to_plot_green_edges)):
        #         edges_to_plot_green.add(tuple([nodes_to_plot_green_edges[i], nodes_to_plot_green_edges[j]]))


        # edges_to_plot_black = set()
        # for i in range(len(nodes_to_plot_black_edges)):
        #     for j in range(i+1, len(nodes_to_plot_black_edges)):
        #         edges_to_plot_black.add(tuple([nodes_to_plot_black_edges[i], nodes_to_plot_black_edges[j]]))

        # for point in edges_to_plot:
        #     # for point2 in edges_to_plot[point1]:
        #         ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], [point[0][2], point[1][2]], 'b--', alpha=0.1)
        
        # for point in edges_to_plot_green:
        #     # for point2 in edges_to_plot[point1]:
        #         ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], [point[0][2], point[1][2]], 'g--', alpha=0.1)

        # for point in edges_to_plot_black:
        #     # for point2 in edges_to_plot[point1]:
        #         ax.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]], [point[0][2], point[1][2]], 'k--', alpha=0.5)
        #     

        
        
        nodes_to_plot_edges = np.array(nodes_to_plot_edges)
        nodes_to_plot_green_edges = np.array(nodes_to_plot_green_edges)
        nodes_to_plot_black_edges = np.array(nodes_to_plot_black_edges)    

        from scipy.spatial import ConvexHull

        blueHull = ConvexHull(nodes_to_plot_edges)
        greenHull = ConvexHull(nodes_to_plot_green_edges)
        blackHull = ConvexHull(nodes_to_plot_black_edges)

        for simplex in blueHull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close the loop
            simplex_points = nodes_to_plot_edges[simplex]
            ax.plot(nodes_to_plot_edges[simplex, 0], nodes_to_plot_edges[simplex, 1], nodes_to_plot_edges[simplex, 2], 'b-', alpha = 0.1)

        for simplex in greenHull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close the loop
            simplex_points = nodes_to_plot_green_edges[simplex]
            ax.plot(nodes_to_plot_green_edges[simplex, 0], nodes_to_plot_green_edges[simplex, 1], nodes_to_plot_green_edges[simplex, 2], 'g-', alpha = 0.2)

        for simplex in blackHull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close the loop
            simplex_points = nodes_to_plot_black_edges[simplex]
            ax.plot(nodes_to_plot_black_edges[simplex, 0], nodes_to_plot_black_edges[simplex, 1], nodes_to_plot_black_edges[simplex, 2], 'k-', alpha = 0.5)



        # plt.axis('off')

        # pdb.set_trace()

        # plot all edges between blue nodes. 

        
        ax.set_aspect('equal', adjustable='box')
        
        plt.savefig(str(row[1][1]) + str(row[1][2]) + str(row[1][3]) +'_3d.png')

        plt.show()
        # exit()

'''