import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import networkx as nx


folder_graph = './graphs/graphsage_graph/'
metadata_file = folder_graph + 'metadata.csv'
metadata = pd.read_csv(metadata_file)

for id, row in enumerate(metadata.iterrows()):
    fname = str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '_CLEAN.pickle'
    try:
        fil =  open(folder_graph+fname, 'rb')
        graph = pickle.load(fil)
        fil.close()
    except:
        continue

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, z_dim):
        super(GraphAutoencoder, self).__init__()
        # Encoder layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, z_dim)

        # Decoder layers
        # For simplicity, we'll use a linear layer for decoding
        self.decoder = torch.nn.Linear(z_dim, num_node_features)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z):
        z = self.decoder(z)
        return torch.sigmoid(z)  # Using sigmoid for the adjacency matrix reconstruction

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z)
