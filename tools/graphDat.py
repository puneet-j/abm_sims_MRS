import torch
import networkx as nx 
from torch.utils.data import Dataset, DataLoader
import pickle

class GraphDataset(Dataset):
    def __init__(self, folder, file_paths):
        self.file_paths = file_paths
        self.folder = folder 
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.folder + self.file_paths[idx], 'rb') as f:
            G = pickle.load( f)
        # Transform the graph as needed, e.g., to PyTorch geometric data format
        # return graph

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