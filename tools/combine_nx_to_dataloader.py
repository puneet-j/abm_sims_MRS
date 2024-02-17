import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        G = self.graph_list[idx]

        # Assuming node features 'x' are stored in each node
        # This will create a list of node feature tensors
        node_features = torch.tensor([G.nodes[node]['x'] for node in G.nodes], dtype=torch.float)

        # For edge features, assuming 'weight' attribute exists for each edge
        # Creating a tensor for edge indices and another for edge features
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_features = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float)

        # Adjacency matrix (optional if you use edge_index and edge_features directly)
        # adj_matrix = nx.to_numpy_matrix(G)
        # adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float)

        # Return node features, edge index, and edge features
        return node_features, edge_index, edge_features

# Example usage
graph_list = []
folder_graph = './graphs/graphsage_graph/single_graphs/'
metadata_file = folder_graph + 'metadata.csv'
succTimeFiles = folder_graph + 'graphMetadata.csv'
import os
import pickle
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
for file in files:
    fil =  open(folder_graph+file, 'rb')
    G = pickle.load(fil)
    fil.close()
    graph_list.append(G)

# dataset = GraphDataset(graph_list)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Save the graph list for later use, similar to the previous example
torch.save(graph_list, 'graph_list_with_features.pth')

# # Example of iterating over the DataLoader in a training loop
# for node_features, edge_index, edge_features in dataloader:
#     # Use these in your GAE model
#     pass