import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GraphConv #,  SAGEConv,
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import pandas as pd 
import pickle
import pdb
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, subgraph, k_hop_subgraph
import matplotlib.pyplot as plt
from combine_nx_to_dataloader import GraphDataset
from dgl.nn import SAGEConv, EdgeWeightNorm
import dgl

'''NOTE: Vigynesh, Puneet: Implement garbage collection'''

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

def generate_mask(x, mask_rate=0.1, device=None):
    # Generate a mask for input features
    mask = torch.FloatTensor(x.size(), device=device).uniform_() > mask_rate
    return mask.float().to(x.device)

def generate_batch_mask(x, batch, mask_rate=0.01, device=None):
    # Generate a mask for batched input features
    mask = torch.rand(x.size(0), 1, device=device) > mask_rate
    return mask.float().to(x.device)


def create_subgraphs(data, num_subgraphs=100, num_hops=2, random_seed=42):
    """
    Generate subgraphs using k-hop subgraph around randomly selected nodes.

    Parameters:
    - data: PyTorch Geometric Data object of the original graph.
    - num_subgraphs: Number of subgraphs to generate.
    - num_hops: Number of hops to consider for the subgraph.
    - random_seed: Seed for reproducibility.
    
    Returns:
    - A list of PyTorch Geometric Data objects representing the subgraphs.
    """
    torch.manual_seed(random_seed)  # For reproducible node selection
    subgraphs = []

    for _ in range(num_subgraphs):
        # Randomly select a node as the center for the subgraph
        node_idx = torch.randint(0, data.num_nodes, (1,)).item()
        # Get the nodes and edges for the k-hop subgraph around the selected node
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )
        
        # Extract the features and labels for the nodes in the subgraph
        subgraph_x = data.x[subgraph_nodes]
        
        # Optionally handle edge attributes if your graph includes them
        subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
        
        # Create a new Data object for the subgraph
        sg_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr)
        subgraphs.append(sg_data)

    return subgraphs


def get_test_train(graph, num_test_nodes=5, random_seed=42):
    # torch.manual_seed(random_seed)
    np.random_seed(random_seed)
    test_idx = np.random.randint(0, graph.num_nodes, (num_test_nodes,))
    testG = nx.Graph()
    testG.add_nodes_from(graph.nodes([test_idx]))
    graph.remove_nodes_from(test_idx)
    return graph, testG

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, 'pool')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2, 'pool')
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2, 'pool')  # Additional hidden layer
        self.conv3 = SAGEConv(hidden_channels * 2, out_channels, 'pool')  # Final Layer

    def forward(self, x, feat, edge_weight):
        try:
            h = F.relu(self.conv1(x, feat, edge_weight=edge_weight))
            h = F.relu(self.conv2(x, h, edge_weight=edge_weight))
            h = F.relu(self.conv3(x, h, edge_weight=edge_weight))
            # x = self.conv3(x, edge_index)
            return h
        except Exception as e:
            print(e)
            pdb.set_trace()

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        # Assuming the encoded features are to be decoded back to original feature size
        self.conv1 = SAGEConv(out_channels, hidden_channels * 2, 'pool')
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels * 2, 'pool')  # Mimic encoder complexity
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels, 'pool')  # Additional hidden layer
        self.conv3 = SAGEConv(hidden_channels * 2, in_channels, 'pool')  # Additional hidden layer to output size

    def forward(self, z, feat, edge_weight):
        hd = F.relu(self.conv1(z, feat, edge_weight=edge_weight))
        hd = F.relu(self.conv2(z, hd, edge_weight=edge_weight))
        hd = F.relu(self.conv3(z, hd, edge_weight=edge_weight))
        # z = self.conv4(z, edge_index)
        adj_rec = torch.matmul(hd, hd.t())
        return hd, adj_rec

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, graph, feat, edge_weight):
        # x_masked = x #* mask
        z = self.encoder(graph, feat, edge_weight)
        x_reconstructed, adj_rec = self.decoder(graph, z, edge_weight)
        return x_reconstructed, adj_rec


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        # pdb.set_trace()
        # mask = generate_batch_mask(data.x, data.batch, device=device).to(device)  # Adjusted for subgraph batching
        # mask = generate_mask(data.x, device=device).to(device)
        # reconstructed_x = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), mask=None)
        # print(data[0][0].shape, data[1][0].shape, data[2][0].shape)
        ew = torch.tensor([1.0/d for d in data[2][0]]).to(device)
        g = dgl.graph((data[1][0][0].to(device), data[1][0][1].to(device)))
        # ew = data[2][0].to(device)
        # norm = EdgeWeightNorm(norm='both')
        # ew_norm = norm(g, ew)
        ew_norm = ew
        feats = data[0][0].to(device)
        reconstructed_x, adj_rec = model(g, feats, edge_weight=ew_norm)
        original_adj = g.adj().to_dense()
        # pdb.set_trace()
        loss = loss_function(reconstructed_x, feats, original_adj, adj_rec) #dgl.graph((data[1][0][0].to(device), data[1][0][0].to(device))))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Assuming a loss function appropriate for node feature reconstruction, e.g., MSE for continuous features
def loss_function(reconstructed_x, original_x, orig_adj, adj):
    return F.mse_loss(reconstructed_x, original_x) + F.cross_entropy(orig_adj, adj)


# Example usage
if __name__ == "__main__":
    # data = 
    # for node_features, edge_features in dataloader:

    # pdb.set_trace()
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    folder_graph = './graphs/graphsage_graph/single_graphs/'
    fname = 'graph_list_with_features.pth'

    hC = 64
    inC = 40 #len(G.nodes[0]['x'])
    # print('dimension of nodes: ',inC)
    '''
    print(f"radius: {nx.radius(G)}")
    print(f"diameter: {nx.diameter(G)}")
    # print(f"eccentricity: {nx.eccentricity(G)}")
    print(f"center: {nx.center(G)}")
    # print(f"periphery: {nx.periphery(G)}")
    # print(f"density: {nx.density(G)}")
    print(f"max degree: {np.max(G.degree())}")
    '''
    # trainG, testG = get_test_train(G, num_test_nodes=0)
    # Convert networkx graph to PyTorch Geometric graph
    # trainPyG = networkx_to_pyg_graph(trainG, 'x', 'weight')
    # pdb.set_trace()
    # subgraphs = create_subgraphs(pyg_graph, num_subgraphs=1, num_hops=2)
    # pdb.set_trace()
    # testPyG = get_test_train(testG, 'x')
    # data_loader = DataLoader(subgraphs, batch_size=1, shuffle=True)
    # pdb.set_trace()
    graph_list = torch.load(fname)
    dataset = GraphDataset(graph_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    final_loss = []
    for outchannels in range(2,5):
        model = MaskedGraphAutoencoder(in_channels=inC, hidden_channels=hC, out_channels=outchannels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        losses = []
        for epoch in range(1, 15):  # Number of epochs
            loss = train(model, dataloader, optimizer)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            losses.append(loss)
        final_loss.append(losses)
    print(final_loss)
    np.save('tools/gaelossarray_with_adjacency.npy', final_loss) 
    # print(range(5,20))
        
