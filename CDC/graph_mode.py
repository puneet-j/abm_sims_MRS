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
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'tools'))
from graphDat import GraphDataset
from sklearn.neighbors import kneighbors_graph
from torch.optim.lr_scheduler import ReduceLROnPlateau


torch.manual_seed(42)
folder_graph = './graphs/CDC/test3/'

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv4 = SAGEConv(hidden_channels, out_channels)  # Final Layer

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

# class GraphDecoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphDecoder, self).__init__()
#         # Assuming the encoded features are to be decoded back to original feature size
#         self.conv1 = SAGEConv(out_channels, hidden_channels )
#         self.conv2 = SAGEConv(hidden_channels , hidden_channels * 2)  # Mimic encoder complexity
#         self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
#         self.conv4 = SAGEConv(hidden_channels, in_channels)  # Additional hidden layer to output size

#     def forward(self, z, edge_index):
#         # pdb.set_trace()
#         z = F.relu(self.conv1(z, edge_index))
#         z = F.relu(self.conv2(z, edge_index))
#         z = F.relu(self.conv3(z, edge_index))
#         z = self.conv4(z, edge_index)
#         return z

class MaskedGraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MaskedGraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
        # self.decoder = GraphDecoder(in_channels, hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x_masked = x #* mask
        z = self.encoder(x_masked, edge_index)
        # x_reconstructed = self.decoder(z, edge_index)
        return z# x_reconstructed, z
    
# def compute_laplacian(adjacency_matrix):
#     # Degree matrix
#     degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
#     # Laplacian matrix
#     laplacian_matrix = degree_matrix - adjacency_matrix
#     return laplacian_matrix

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    embeddings = []
    origs = []
    for data in data_loader:
        optimizer.zero_grad()
        embed = model(data[0][0], data[1][0])
        origs.append(data[0][0])
        loss = cosine_loss(data[1][0], embed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        embeddings.append(embed.detach())
    return total_loss / len(data_loader), embeddings, origs

# Assuming a loss function appropriate for node feature reconstruction, e.g., MSE for continuous features
def loss_function(reconstructed_x, original_x):
    return F.mse_loss(reconstructed_x, original_x)


def cosine_loss(adj, emb):
    norms = torch.norm(emb, p=2, dim=1, keepdim=True)
    normalized_embeddings = emb / norms.clamp(min=1e-8)
    cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    target_adjacency = torch.zeros_like(cosine_similarity_matrix)
    target_adjacency[adj[0], adj[1]] = 1
    # Directly using logits; no need to apply sigmoid.
    return F.binary_cross_entropy_with_logits(cosine_similarity_matrix, target_adjacency)

# def cosine_loss(adj, emb):
#     norms = torch.norm(emb,p=2,dim=1, keepdim=True)
#     norms = torch.where(norms == 0, torch.tensor(1.0, device=emb.device), norms)
#     normalized_embeddings = emb / norms
#     cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
#     num_nodes = cosine_similarity_matrix.size(0)
#     target_adjacency = torch.zeros((num_nodes, num_nodes))#, device=cosine_similarity_matrix.device)
#     target_adjacency[adj[0], adj[1]] = 1
#     predicted_probabilities = torch.sigmoid(cosine_similarity_matrix)
#     return F.binary_cross_entropy(predicted_probabilities, target_adjacency)

# def lap_loss(lap, emb):
#     return torch.trace(torch.matmul(emb.t(), torch.matmul(lap, emb)))

# def diversity_loss(embeddings):
#     # Calculate pairwise Euclidean distances
#     pairwise_distances = torch.pdist(embeddings, p=2)
#     # Encourage distances to be greater than a threshold
#     threshold  = 1
#     diversity_penalty = torch.relu(threshold - pairwise_distances).mean()
#     return diversity_penalty



# Example usage
if __name__ == "__main__":

    
    # fname = 'small_graphs.pth'
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
 
    # graph_list = torch.load(folder_graph+fname)
    dataset = GraphDataset(folder_graph, files)
    # dataloader = 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    actuals = []
    final_loss = []
    embeddings = []
    for outchannels in range(2,4):
        model = MaskedGraphAutoencoder(in_channels=inC, hidden_channels=hC, out_channels=outchannels)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
        losses = []
        embeds = []
        acts = []
        for epoch in range(1, 30):  # Number of epochs
            loss, embeds, orig = train(model, dataloader, optimizer)
            embeds.append(embeds)
            acts.append(orig)
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
            losses.append(loss)
            # scheduler.step(loss)
        # epoch_loss = loss / len(dataloader)
        embeddings.append(embeds)
        final_loss.append(losses)
        actuals.append(acts)
        # Assuming encoder and decoder are your model's components
        encoder_state_dict = model.encoder.state_dict()
        # decoder_state_dict = model.decoder.state_dict()

        # Save the state dictionaries
        torch.save(encoder_state_dict, './CDC/encoder_state_dict'+str(outchannels)+'.pth')
        # torch.save(decoder_state_dict, './CDC/decoder_state_dict'+str(outchannels)+'.pth')

    # print(final_loss)
    np.save('./CDC/gaelossarray_sage_no_weights_3_5_with_embeds.npy', final_loss)
    # pdb.set_trace()
    fl = open('./CDC/embeddings_3_5.pickle', 'wb')
    pickle.dump(embeddings, fl)
    fl.close()

    fl = open('./CDC/actuals_3_5.pickle', 'wb')
    pickle.dump(actuals, fl)
    fl.close()

    np.save(files, './CDC/file_list.npy')
