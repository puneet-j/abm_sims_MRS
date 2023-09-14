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
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from params import *

class GraphDataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, file_name, train, neighbors_list):
            'Initialization'
            fil =  open(file_name, 'rb')
            self.dat = pickle.load(fil)   
            fil.close()
            draw = False
            shorten_graph_for_test = True
            if shorten_graph_for_test:
                  k = 100
                  sampled_nodes = random.sample(list(self.dat.nodes), k)
                  dat = self.dat.subgraph(sampled_nodes)
            else:
                  dat = self.dat
            
            self.len_of_data = dat.number_of_nodes()
            train_num = int(np.ceil(self.len_of_data*train))
            # test_num = self.len_of_data - train_mask
            # pdb.set_trace()

            idx = [i for i in range(self.len_of_data)]
            
            pyg_graph = from_networkx(dat)
            # pdb.set_trace()

            np.random.shuffle(idx)
            self.train_idx = idx[:train_num]
            self.test_idx = idx[train_num:]
            # train_mask = torch.full_like(pyg_graph.x, False, dtype=bool)
            # train_mask[idx[:train_num]] = True
            # test_mask = torch.full_like(pyg_graph.y, False, dtype=bool)
            # test_mask[idx[train_num:]] = True
            self.sampled_graph = NeighborLoader(pyg_graph,
                  # Sample 30 neighbors for each node for 2 iterations
                  num_neighbors=neighbors_list,
                  # Use a batch size of 128 for sampling training nodes
                  batch_size=32,
                  input_nodes=self.train_idx,)
            self.all = NeighborLoader(pyg_graph, num_neighbors=[-1])
            # ,
            #       # Sample 30 neighbors for each node for 2 iterations
            #       num_neighbors=neighbors_list,
            #       # Use a batch size of 128 for sampling training nodes
            #       batch_size=32,)
            # # pdb.set_trace()

            if draw:
                  # k = 100
                  # sampled_nodes = random.sample(list(dat.nodes), k)
                  sampled_graph = dat
                  pos = nx.spring_layout(sampled_graph, seed=0)
                  nx.draw_networkx(sampled_graph, pos, with_labels=False, node_size = 50.0) #pos=nx.spring_layout(sampled_graph),
                  for edge in sampled_graph.edges(data='weight'):
                        nx.draw_networkx_edges(sampled_graph, pos, edgelist=[edge], width=edge[2])
                  plt.savefig("filename4.png")
                  print(sampled_graph.number_of_edges())
                  # pdb.set_trace()
                  exit()

      def __len__(self):
            'Denotes the total number of samples'
            return self.len_of_data

      def __getitem__(self, index):
            return self.sampled_graph # [index]
            'Generates one sample of data'
            # Select sample
            # ID = self.len_of_data[index]
# 
            # Load data and get label
            # X = torch.load('data/' + ID + '.pt')
            # y = self.labels[ID]

            # return X, y