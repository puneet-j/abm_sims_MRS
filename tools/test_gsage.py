import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pandas as pd
import pdb
from ast import literal_eval
import copy
import pickle
from sklearn.preprocessing import StandardScaler
from myclasses import GraphDataset
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
# https://towardsdatascience.com/how-to-create-a-graph-neural-network-in-python-61fd9b83b54e
import torch
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import GraphSAGE
plt.ion()   # interactive mode
import torch.nn.functional as F


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return h

    def fit(self, loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        self.train()
        for epoch in range(epochs+1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            for batch in loader:
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss.item()
                acc += accuracy(out[batch.train_mask].argmax(dim=1), batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                # val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                # val_acc += accuracy(out[batch.val_mask].argmax(dim=1), batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if epoch % 20 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {loss/len(loader):.3f} | Train Acc: {acc/len(loader)*100:>6.2f}%')# | Val Loss: {val_loss/len(train_loader):.2f} | Val Acc: {val_acc/len(train_loader)*100:.2f}%')

    # @torch.no_grad()
    # def test(self, data):
    #     self.eval()
    #     out = self(data.x, data.edge_index)
    #     acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    #     return acc
    
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def fit(loader):
    model.train()

    total_loss = 0
    data = loader
    # for data in loader:
        # pdb.set_trace()
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    total_loss += loss.item() * data.num_graphs
    loss.backward()
    optimizer.step()
    return total_loss / len(loader)

# @torch.no_grad()
# def test(loader):
#     model.eval()

#     data = next(iter(loader))
#     out = model(data.x.to(device), data.edge_index.to(device))
#     preds = (out > 0).float().cpu()

#     y, pred = data.y.numpy(), preds.numpy()
#     return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

# for epoch in range(301):
#     loss = fit(train_loader)
#     val_f1 = test(val_loader)
#     if epoch % 50 == 0:
#         print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Val F1-score: {val_f1:.4f}')

# print(f'Test F1-score: {test(test_loader):.4f}')

# def main():
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
# model.to(device)
# epochs = 100
# optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
# scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)
folder = './graphs/'
files = os.listdir(folder)
files = [file for file in files if file.endswith('.pickle')]
files = np.sort(files)
metadata_file = folder + 'metadata.csv'
OUTPUT_DIM = 10
metadata = pd.read_csv(metadata_file)

for row in metadata.iterrows():
    # pdb.set_trace()
    fname = str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
    num_neighbors_sampling = [5] * 2
    train_percent = 0.8
    data = GraphDataset(folder + fname,  train_percent, num_neighbors_sampling)
dataloader = data.sampled_graph.data #DataLoader(data, batch_size=4,shuffle=True, num_workers=0)
# pdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    dim_in=dataloader.num_features,
      dim_h=512,
        dim_out=OUTPUT_DIM).to(device)
    # in_channels=dataloader.num_features,
    # hidden_channels=512,
    # num_layers=2,
    # out_channels=OUTPUT_DIM,
    # ).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(30):
    loss = fit(dataloader)
    # val_f1 = test(val_loader)
    if epoch % 5 == 0:
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f}')#   | Val F1-score: {val_f1:.4f}')

# print(f'Test F1-score: {test(test_loader):.4f}')

# main()