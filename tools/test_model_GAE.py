import pickle
import numpy as np
import pandas as pd
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
from combine_nx_to_dataloader import GraphDataset

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels * 2)
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Additional hidden layer
        self.conv3 = SAGEConv(hidden_channels * 2, out_channels)  # Final Layer

    def forward(self, x, edge_index):
        try:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
            # x = self.conv3(x, edge_index)
            return x
        except Exception as e:
            print(e)
            pdb.set_trace()

class GraphDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        # Assuming the encoded features are to be decoded back to original feature size
        self.conv1 = SAGEConv(out_channels, hidden_channels * 2)
        self.conv2 = SAGEConv(hidden_channels * 2, hidden_channels * 2)  # Mimic encoder complexity
        # self.conv3 = SAGEConv(hidden_channels * 2, hidden_channels)  # Additional hidden layer
        self.conv3 = SAGEConv(hidden_channels * 2, in_channels)  # Additional hidden layer to output size

    def forward(self, z, edge_index):
        z = F.relu(self.conv1(z, edge_index))
        z = F.relu(self.conv2(z, edge_index))
        z = F.relu(self.conv3(z, edge_index))
        # z = self.conv4(z, edge_index)
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
    

'''

in_channels = 40
hidden_channels = 64
out_channels = 0


# Initialize the models / components with the same architecture
encoder = GraphEncoder(in_channels, hidden_channels, out_channels)
decoder = GraphDecoder(in_channels, hidden_channels, out_channels)
'''
# Load the state dictionaries
encoder = torch.load('encoder0.pth')
decoder = torch.load('decoder0.pth')

fil =  open('embeddings.pickle', 'rb')
embed = pickle.load(fil)   
fil.close() 
emb = embed[1]

# Update the models with the loaded state
# encoder.load_state_dict(encoder_state_dict)
# decoder.load_state_dict(decoder_state_dict)

pdb.set_trace()

result = decoder(emb, [])