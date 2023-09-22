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
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader, NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE
plt.ion()   # interactive mode
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
import os.path as osp
folder = './graphs/test/'
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
    data = GraphDataset(folder + fname,  train_percent, num_neighbors_sampling).pyg_graph


''' below is test code to see how data looks when in fixed format'''
# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data2 = dataset[0]
# pdb.set_trace()

train_loader = LinkNeighborLoader(
    data,
    batch_size=32,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[5, 5],
)
# pdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device, 'x', 'edge_index')
print('using device', device)
model = GraphSAGE(
    in_channels=len(data.x[0]),
    hidden_channels=64,
    num_layers=2,
    out_channels=32,
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# torch.backends.cuda.matmul.allow_tf32 = True

def train():
    model.train()
    print('starting training')
    total_loss = 0
    for batch in train_loader:
        # print('batch: ', batch)
        batch = batch.to(device)
        optimizer.zero_grad()
        # pdb.set_trace()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index).cpu()

    clf = LogisticRegression()
    clf.fit(out[data.train_mask], data.y[data.train_mask])

    val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
    test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

    return val_acc, test_acc

import time
times = []
for epoch in range(1, 51):
    print('started epoch', epoch)
    start = time.time()
    loss = train()
    # val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ')
        #   f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")