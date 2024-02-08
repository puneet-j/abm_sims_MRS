import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd 
import pickle
import pdb
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, subgraph
import matplotlib.pyplot as plt

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, 2 * out_channels)
        self.conv2 = SAGEConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphDecoder(nn.Module):
    def __init__(self):
        super(GraphDecoder, self).__init__()

    def forward(self, z):
        adj = torch.matmul(z, z.t())
        return adj

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, out_channels)
        self.decoder = GraphDecoder()

    def forward(self, x, edge_index, mask):
        # Apply mask to input features
        x_masked = x * mask
        z = self.encoder(x_masked, edge_index)
        adj_reconstructed = self.decoder(z)
        return adj_reconstructed

def networkx_to_pyg_graph(G, node_features, edge_features):
    # Convert a networkx graph to a PyTorch Geometric Data object
    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.tensor([G.nodes[node][node_features] for node in G.nodes], dtype=torch.float)
    
    if edge_features is not None:
        edge_attr = torch.tensor([G.edges[edge][edge_features] for edge in G.edges], dtype=torch.float)
    else:
        edge_attr = None
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def generate_mask(x, mask_rate=0.1):
    # Generate a mask for input features
    mask = torch.FloatTensor(x.size()).uniform_() > mask_rate
    return mask.float()

def generate_batch_mask(x, batch, mask_rate=0.1):
    # Generate a mask for batched input features
    mask = torch.rand(x.size(0), 1) > mask_rate
    return mask.float().to(x.device)


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        mask = generate_batch_mask(data.x, data.batch)  # Adjusted for subgraph batching
        adj_reconstructed = model(data.x, data.edge_index, mask)
        
        adj_label = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(0)
        adj_label = torch.triu(adj_label, diagonal=1)  # Consider only upper triangle to avoid redundancy
        
        loss = F.binary_cross_entropy_with_logits(adj_reconstructed, adj_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)



def create_subgraphs(data, num_subgraphs=10):
    # Split the graph into num_subgraphs subgraphs
    nodes = np.arange(data.num_nodes)
    subgraph_nodes = np.array_split(nodes, num_subgraphs)

    subgraphs = []
    for node_subset in subgraph_nodes:
        # Create subgraph for the current subset of nodes
        try:
            sg_edge_index, sg_edge_attr = subgraph(torch.from_numpy(node_subset), data.edge_index, edge_attr=data.edge_attr, relabel_nodes=False)
        except:
            # pdb.set_trace()
            plt.figure(1)
            plt.plot(node_subset)
            plt.figure(2)
            plt.plot(data.edge_index[0,:], data.edge_index[1,:])
            plt.figure(3)
            plt.plot(data.edge_attr)
            plt.show()
            pdb.set_trace()
        sg_x = data.x[node_subset]
        sg_data = Data(x=sg_x, edge_index=sg_edge_index, edge_attr=sg_edge_attr)
        subgraphs.append(sg_data)
    
    return subgraphs

# Example usage
if __name__ == "__main__":
    # Create a networkx graph
    folder_graph = './graphs/graphsage_graph/'
    metadata_file = folder_graph + 'metadata.csv'
    succTimeFiles = folder_graph + 'graphMetadata.csv'
    succTime = pd.read_csv(succTimeFiles)
    metFile = pd.read_csv(metadata_file)
    for id, row in enumerate(metFile.iterrows()):
        fname = str(row[1].iloc[1]) + str(row[1].iloc[2]) + str(row[1].iloc[3]) + '.pickle'
        try:
            fil =  open(folder_graph+fname, 'rb')
            G = pickle.load(fil)
            fil.close()
        except:
            continue
    # pdb.set_trace()
    inC = len(G.nodes[0]['x'])
    # Convert networkx graph to PyTorch Geometric graph
    pyg_graph = networkx_to_pyg_graph(G, 'x', 'weight')
    # pdb.set_trace()
    subgraphs = create_subgraphs(pyg_graph, num_subgraphs=200)
    data_loader = DataLoader(subgraphs, batch_size=1, shuffle=True) 
    
    model = MaskedGraphAutoencoder(in_channels=inC, out_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 10):  # Number of epochs
        loss = train(model, data_loader, optimizer)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
