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
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, subgraph, k_hop_subgraph
import matplotlib.pyplot as plt

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels * 2, out_channels)  # Additional hidden layer

    def forward(self, x, edge_index):
        try:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            x = self.conv4(x, edge_index)
            return x
        except:
            pdb.set_trace()

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        # Assuming the encoded features are to be decoded back to original feature size
        self.conv1 = SAGEConv(out_channels, hidden_channels * 2)
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Mimic encoder complexity
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, in_channels)  # Additional hidden layer to output size

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = F.relu(self.conv2(z, edge_index))
        z = F.relu(self.conv3(z, edge_index))
        z = self.conv4(z, edge_index)
        return z

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index, mask):
        x_masked = x * mask
        z = self.encoder(x_masked, edge_index)
        x_reconstructed = self.decoder(z, edge_index)
        return x_reconstructed

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

def generate_batch_mask(x, batch, mask_rate=0.01, device=None):
    # Generate a mask for batched input features
    mask = torch.rand(x.size(0), 1, device=device) > mask_rate
    return mask.float().to(x.device)


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        mask = generate_batch_mask(data.x, data.batch, device=device).to(device)  # Adjusted for subgraph batching
        
        reconstructed_x = model(data.x.to(device), data.edge_index.to(device), mask)
        # pdb.set_trace()
        loss = loss_function(reconstructed_x, data.x.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Assuming a loss function appropriate for node feature reconstruction, e.g., MSE for continuous features
def loss_function(reconstructed_x, original_x):
    return F.mse_loss(reconstructed_x, original_x)

# def create_subgraphs(data, num_subgraphs=10):
#     # Split the graph into num_subgraphs subgraphs
#     # pdb.set_trace()
#     nodes = np.arange(data.num_nodes)#np.random.choice(data.num_nodes,)
#     subgraph_nodes = np.array_split(nodes, num_subgraphs)

#     subgraphs = []
#     for node_subset in subgraph_nodes:
#         # Create subgraph for the current subset of nodes
#         try:
#             sg_edge_index, sg_edge_attr = k_hop_subgraph(torch.from_numpy(node_subset), data.edge_index, edge_attr=data.edge_attr, relabel_nodes=False)
#         except:
#             # pdb.set_trace()
#             plt.figure(1)
#             plt.plot(node_subset)
#             plt.figure(2)
#             plt.plot(data.edge_index[0,:], data.edge_index[1,:])
#             plt.figure(3)
#             plt.plot(data.edge_attr)
#             plt.show()
#             pdb.set_trace()
#         sg_x = data.x[node_subset]
#         sg_data = Data(x=sg_x, edge_index=sg_edge_index, edge_attr=sg_edge_attr)
#         subgraphs.append(sg_data)
    
#     return subgraphs

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


# Example usage
if __name__ == "__main__":
    # Create a networkx graph
    folder_graph = './graphs/graphsage_graph/'
    metadata_file = folder_graph + 'metadata.csv'
    succTimeFiles = folder_graph + 'graphMetadata.csv'
    succTime = pd.read_csv(succTimeFiles)
    metFile = pd.read_csv(metadata_file)
    for id, row in enumerate(metFile.iterrows()):
        fname = str(row[1].iloc[1]) + str(row[1].iloc[2]) + str(row[1].iloc[3]) + '_noAgentPos' + '.pickle'
        # pdb.set_trace()
        try:
            fil =  open(folder_graph+fname, 'rb')
            G = pickle.load(fil)
            fil.close()
        except:
            continue
    # pdb.set_trace()
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    hC = 64
    inC = len(G.nodes[0]['x'])
    '''
    print(f"radius: {nx.radius(G)}")
    print(f"diameter: {nx.diameter(G)}")
    # print(f"eccentricity: {nx.eccentricity(G)}")
    print(f"center: {nx.center(G)}")
    # print(f"periphery: {nx.periphery(G)}")
    # print(f"density: {nx.density(G)}")
    print(f"max degree: {np.max(G.degree())}")
    '''
    # Convert networkx graph to PyTorch Geometric graph
    pyg_graph = networkx_to_pyg_graph(G, 'x', 'weight')
    # pdb.set_trace()
    subgraphs = create_subgraphs(pyg_graph, num_subgraphs=100, num_hops=2)
    # pdb.set_trace()
    data_loader = DataLoader(subgraphs, batch_size=1, shuffle=True)
    # pdb.set_trace()
    final_loss = []
    for outchannels in range(5,20):
        model = MaskedGraphAutoencoder(in_channels=inC, hidden_channels=hC, out_channels=5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        for epoch in range(1, 30):  # Number of epochs
            loss = train(model, data_loader, optimizer)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
        final_loss.append(loss)
    print(final_loss) 
    print(range(5,20))
        