import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv, GraphConv, GraphSAGE

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
folder_graph = './graphs/CDC/lotsOfInfo/notAllSuccess/'


class GraphEncoderWithResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoderWithResidual, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        self.conv3 = SAGEConv(hidden_channels * 2, out_channels)
        self.lin = nn.Linear(hidden_channels * 2, out_channels)
        # Linear transformation to match dimensions for residual connection
        self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = self.conv3(x, edge_index)
        x = self.lin(x)
        # Applying shortcut and adding it to the output of conv3
        identity = self.shortcut(identity)
        x += identity  # Element-wise addition
        return x
    

    
# class GraphEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GraphEncoder, self).__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
#         # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Additional hidden layer
#         self.conv4 = SAGEConv(hidden_channels * 2, out_channels)  # Final Layer

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         # x = F.relu(self.conv3(x, edge_index))
#         x = self.conv4(x, edge_index)
#         return x
    
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
    normalized_embeddings = emb / norms.clamp(min=1e-4)
    cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    target_adjacency = torch.zeros_like(cosine_similarity_matrix)
    target_adjacency[adj[0], adj[1]] = 1
    # Directly using logits; no need to apply sigmoid.
    return F.binary_cross_entropy_with_logits(cosine_similarity_matrix, target_adjacency)


# Example usage
if __name__ == "__main__":

    
    # fname = 'small_graphs.pth'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle')]
    hC = 10
    inC = 42

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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    actuals = []
    final_loss = []
    embeddings = []
    for outchannels in range(2,4):
        model = GraphEncoderWithResidual(in_channels=inC, hidden_channels=hC, out_channels=outchannels)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
        losses = []
        embeds = []
        acts = []
        for epoch in range(1, 20):  # Number of epochs
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
        encoder_state_dict = model.state_dict()
        # decoder_state_dict = model.decoder.state_dict()

        # Save the state dictionaries
        torch.save(encoder_state_dict, './CDC/LinearLayernoAllGood_global_LotsInfo_Colored_Res_30ep_0p01_encoder_state_dict'+str(outchannels)+'.pth')
        # torch.save(decoder_state_dict, './CDC/decoder_state_dict'+str(outchannels)+'.pth')


    # # print(final_loss)
    # np.save('./CDC/noAllGood_global_LotsInfo_Colored_Res_30ep_0p01_gaelossarray_sage_no_weights_3_5_with_embeds.npy', final_loss)
    # # pdb.set_trace()
    # fl = open('./CDC/noAllGood_global_LotsInfo_Colored_Res_30ep_0p01_embeddings_3_5.pickle', 'wb')
    # pickle.dump(embeddings, fl)
    # fl.close()

    # fl = open('./CDC/noAllGood_global_LotsInfo_Colored_Res_30ep_0p01_actuals_3_5.pickle', 'wb')
    # pickle.dump(actuals, fl)
    # fl.close()

    # np.save('./CDC/notAllGood_global_LotsInfo_Colored_Res_file_list.npy', files)
