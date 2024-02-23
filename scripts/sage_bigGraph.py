import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv, GraphConv
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
from tools.graphDat import GraphDataset
torch.manual_seed(42)
import os 

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels )
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, out_channels)  # Final Layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        # Assuming the encoded features are to be decoded back to original feature size
        self.conv1 = SAGEConv(out_channels, hidden_channels )
        self.conv2 = SAGEConv(hidden_channels , hidden_channels )  # Mimic encoder complexity
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, in_channels)  # Additional hidden layer to output size

    def forward(self, z, edge_index):
        # pdb.set_trace()
        z = F.relu(self.conv1(z, edge_index))
        z = F.relu(self.conv2(z, edge_index))
        # z = F.relu(self.conv3(z, edge_index))
        z = self.conv4(z, edge_index)
        return z

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x_masked = x #* mask
        z = self.encoder(x_masked, edge_index)
        x_reconstructed = self.decoder(z, edge_index)
        return x_reconstructed, z
    

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    embeddings = []
    origs = []
    for data in data_loader:
        optimizer.zero_grad()
        # pdb.set_trace()
        reconstructed_x, embed = model(data[0][0], data[1][0])
        origs.append(data[0][0])
        # pdb.set_trace()
        loss = loss_function(reconstructed_x, data[0][0], embed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        embeddings.append(embed.detach())
    return total_loss / len(data_loader), embeddings, origs

# Assuming a loss function appropriate for node feature reconstruction, e.g., MSE for continuous features
def loss_function(reconstructed_x, original_x, emb):
    lambda_diversity = 0.3
    return F.mse_loss(reconstructed_x, original_x) + lambda_diversity * diversity_loss(emb)


def diversity_loss(embeddings):
    # Calculate pairwise Euclidean distances
    pairwise_distances = torch.pdist(embeddings, p=2)
    # Encourage distances to be greater than a threshold
    threshold  = 5
    diversity_penalty = torch.relu(threshold - pairwise_distances).mean()
    return diversity_penalty



# Example usage
if __name__ == "__main__":
    
    folder_graph = './graphs/graphsage_graph/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle')]

    hC = 64
    inC = 40 

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
    # trainG, testG = get_test_train(G, num_test_nodes=0)
    # trainPyG = networkx_to_pyg_graph(trainG, 'x', 'weight')
    # testPyG = get_test_train(testG, 'x')
    
    # graph_list = torch.load(fname)
    graph_list = []
    for f in files:
        fl = open(folder_graph+f, 'rb')
        graph_list.append(pickle.load(fl))
        fl.close()
    dataset = GraphDataset(graph_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    actuals = []
    final_loss = []
    embeddings = []
    for outchannels in range(2,5):
        model = MaskedGraphAutoencoder(in_channels=inC, hidden_channels=hC, out_channels=outchannels)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        losses = []
        embeds = []
        acts = []
        for epoch in range(1, 20):  # Number of epochs
            loss, embeds, orig = train(model, dataloader, optimizer)
            embeds.append(embeds)
            acts.append(orig)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            losses.append(loss)
        embeddings.append(embeds)
        final_loss.append(losses)
        actuals.append(acts)
        # Assuming encoder and decoder are your model's components
        encoder_state_dict = model.encoder.state_dict()
        decoder_state_dict = model.decoder.state_dict()

        # Save the state dictionaries
        torch.save(encoder_state_dict, 'Bigencoder_state_dict'+str(outchannels)+'.pth')
        torch.save(decoder_state_dict, 'Bigdecoder_state_dict'+str(outchannels)+'.pth')

    # print(final_loss)
    np.save('Biggaelossarray_sage_no_weights_3_4_with_embeds.npy', final_loss)
    # pdb.set_trace()
    fl = open('Bigembeddings_3_4.pickle', 'wb')
    pickle.dump(embeddings, fl)
    fl.close()

    fl = open('Bigactuals_3_4.pickle', 'wb')
    pickle.dump(actuals, fl)
    fl.close()
