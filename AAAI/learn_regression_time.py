import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GATConv, SAGEConv, GraphConv, GraphSAGE, GCNConv, GATv2Conv
import torch
import networkx as nx 
from torch.utils.data import Dataset
import pickle
import pdb 
import numpy as np
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
from sklearn.neighbors import kneighbors_graph
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch_geometric.data import Data
# from newGraphDat import GraphDataset
# from GraphDatAll import GraphDataset
from GraphDatRegr import GraphDataset

from sklearn.manifold import TSNE
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# import numpy as np
import scipy.io
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# from graphDat import GraphDataset
# from ClassGraphDat import GraphDataset

torch.manual_seed(42)
 

def get_color(node, quals, na):
    red = node_to_color_red(node, quals, na)
    green = node_to_color_green(node, quals, na)
    black = node_to_color_black(node)
    if red is True:
        return 'r'
    elif green is True:
        return 'g'
    elif black is True:
        return 'k'
    else:
        return 'b'

# def get_full_qual(q):
#     while len(q) < 4:
#         q.append(0.0)
#     return q

    

def get_complete_poses(a):
    arr = [[MAX_DIST]*2]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = list(a[i])
    # pdb.set_trace()
    return arr

def get_complete_quals(a):
    arr = [0]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = a[i]
    # pdb.set_trace()
    return arr

def get_complete_global(a):
    arr = [0]*4
    for i in range(0,len(a)):
        # pdb.set_trace()
        arr[i] = a[i]
    # pdb.set_trace()
    return arr

def node_to_color_black(node):
    if node[3] == 0.0:
        return True
    else:
        return False

def node_to_color_green(node, quals, nA):
    # pdb.set_trace()
    # nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    qs = [0]*4
    # pdb.set_trace()
    for d in dancers:
        ii = quals.index(d[1])
        qs[ii] += 1

    if np.max(qs) > COMMIT_THRESHOLD*nA:
        id = np.argmax(qs)
        if quals[id] == np.max(quals):
            return True
        else:
            return False
    else:
        return False
    
def node_to_color_red(node, quals, nA):
    # pdb.set_trace()
    # nA = np.sum([1 for _ in node[0::4]])
    dancers = [(1, q) for a, q in zip(node[0::4], node[3::4]) if a == 0.0]
    qs = [0]*4
    # pdb.set_trace()
    for d in dancers:
        ii = quals.index(d[1])
        qs[ii] += 1

    if np.max(qs) > COMMIT_THRESHOLD*nA:
        id = np.argmax(qs)
        if quals[id] != np.max(quals):
            return True
        else:
            return False
    else:
        return False


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p, h = 8):
        super(GATNet, self).__init__()
        # Define the first GAT convolution layer
        self.conv1 = GATConv(in_channels, hidden_channels*4, heads=h, dropout=p)
        # Define the second GAT convolution layer
        self.conv2 = GATConv(hidden_channels*4 * h, hidden_channels*2, heads=h, concat=True, dropout=p)
        # Define the third GAT convolution layer
        self.conv3 = GATConv(hidden_channels*2 * h, out_channels, heads=1, concat=True, dropout=p)
        # Define a linear layer to refine the outputs to the desired size
        self.linear = nn.Linear(out_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.p = p
        # self.sigm = nn.Sigmoid()
        # self.ta = nn.Tanh()

    def forward(self, x, edge_index):
        init = x
        # x, edge_index = data.x, data.edge_index
        # Apply dropout to the input features and pass through the first GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the third GAT layer
        x = self.conv3(x, edge_index)#F.dropout(x, p=self.p, training=self.training)
        x = x + self.shortcut(init)
        
        # x = F.dropout(x, p=self.p, training=self.training)
        # x = F.elu(self.conv3(x, edge_index))
        # Apply the linear layer
        x = F.dropout(self.linear(x), p=self.p, training=self.training)
        embed = x
        x = self.linear2(x)
        
        # x[:,0] = self.sigm(x[:,0])
        # x[:,0] = self.ta(x[:,0] + 1)/2.0
        return x, embed

class MLPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p):
        super(MLPNet, self).__init__()
        # Define the first linear layer
        self.linear1 = nn.Linear(in_channels, hidden_channels * 4)
        # Define the second linear layer
        self.linear2 = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        # Define the third linear layer
        self.linear3 = nn.Linear(hidden_channels * 2, hidden_channels)
        # Define additional linear layers for refinement
        self.linear4 = nn.Linear(hidden_channels, out_channels)
        self.linear5 = nn.Linear(out_channels, out_channels)
        self.shortcut = nn.Linear(in_channels, hidden_channels)
        self.p = p

    def forward(self, x):
        init = x
        # Apply dropout to the input features and pass through the first linear layer
        x = F.elu(self.linear1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the second linear layer
        x = F.elu(self.linear2(x))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the third linear layer
        x = self.linear3(x)
        x = x + self.shortcut(init)

        x = F.dropout(self.linear4(x), p=self.p, training=self.training)
        embed = x
        x = self.linear5(x)

        return x, embed

def class_loss(reconstructed_x, original_x):
    # const = 200000.0 
    # arr = [1428, 1710, 84, 812]
    # arr = [106607, 49738, 3644, 77311]
    # arr = [46846, 77476, 3225, 84662]
    # arr = [113157, 238960, 25874, 288376]
    arr = [1.0, 1.0, 1.0, 1.0]
    # arr = [7360, 4321, 6313, 5469]
    # arr = [96700, 574801, 1770, 4105]
    # arr = [31638, 159420, 66832, 419486]
    # arr = [79848, 403710, 18672, 175146]
    # arr = [12210,7490,1296,2467]
    const = np.sum(arr)
    # arr = [const, const, const, const]
    weights = torch.log(torch.tensor([const/a for a in arr], dtype=torch.float))
    # weights = torch.tensor([const/(4.0*a) for a in arr], dtype=torch.float)

    return F.cross_entropy(reconstructed_x, original_x, weight=weights)  
  
# def new_class_loss(reconstructed_x, original_x):
#     return F.MSELoss(reconstructed_x, original_x)

# def regression_loss(reconstructed_x, original_x):
#     return F.MSELoss(reconstructed_x, original_x)

def cosine_loss(adj, emb, ew):
    # pdb.set_trace()
    norms = torch.norm(emb, p=2, dim=1, keepdim=True)
    normalized_embeddings = emb / norms.clamp(min=1e-4)
    cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    target_adjacency = torch.zeros_like(cosine_similarity_matrix)
    normalized_weights = ew / torch.norm(ew, p=2, keepdim=True)
    target_adjacency[adj[0], adj[1]] = normalized_weights
    # Directly using logits; no need to apply sigmoid.
    return F.binary_cross_entropy_with_logits(cosine_similarity_matrix, target_adjacency)

def total_loss_func(adj, emb, ew, reconstructed_x, original_x):
    cs = cosine_loss(adj, emb, ew)
    # cls = class_loss(reconstructed_x, original_x)
    reg_loss = nn.MSELoss()
    cls = reg_loss(reconstructed_x, original_x)
    # print(cls)
    # pdb.set_trace()
    # penalty1 = torch.mean(torch.relu(reconstructed_x[:, 0] - 1))
    # penalty0 = torch.mean(torch.relu(-reconstructed_x[:, 0]))  
    # cls = cls + (penalty1 + penalty0)*10
    alpha = 1.0
    return alpha * cls + (1.0 - alpha) * cs

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    embeddings = []
    origs = []
    # qs = []
    # agents = []
    # printlosses = []
    actuals = []
    classes = []
    for feat, edge, ew, _, _, _, _, tS in dataloader:
        optimizer.zero_grad()
        features = feat.squeeze_(0).to(device)
        edges = edge.squeeze_(0).to(device)
        origs.append(tS)
        cl, embed = model(features, edges)
        # cl, embed = model(features)#, edges)
        actuals.append(features.detach())
        # cl[:, 0] = torch.clamp(cl[:, 0], max=1.0)

        loss = total_loss_func(edges, embed, ew, cl.squeeze_(1), tS.squeeze_(0).to(device))

        # loss = total_loss_func(edges, embed, ew, cl, tS.squeeze_(0).to(device))
        # loss = regression_loss(cl, )
        # pdb.set_trace()
        # loss = cosine_loss(edges, embed, ew)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        embeddings.append(embed.detach())
        classes.append(cl.detach())
        # print(total_loss)
    return total_loss / len(data_loader), embeddings, origs, classes, actuals

# Copy code
def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    counter = 0 
    validation_embeddings = []
    validation_classes = []
    qs = []
    agents = []
    origs = []
    embeds = []
    act_feats = []
    # memory = torch.zeros((10,1))
    with torch.no_grad():
        for feat, edge, ew, _, _, q, nA, ts in data_loader:
            counter += 1
            features = feat.squeeze_(0).to(device)
            edges = edge.squeeze_(0).to(device)
            cl, embed = model(features, torch.empty((2,0), dtype=torch.int64))
            # cl, embed = model(features)#, torch.empty((2,0), dtype=torch.int64))

            validation_embeddings.append(embed.detach())
            qs.append(q)
            agents.append(nA)
            origs.append(ts.detach())
            embeds.append(embed.detach())
            validation_classes.append(cl.detach())
            act_feats.append(features.detach())
            # if counter < 5:
            #     print(embed)
            # loss = class_loss(cl, ts.squeeze_(0))
            # loss = total_loss_func(edges, embed, ew, cl, ts.squeeze_(0))
            loss = total_loss_func(edges, embed, ew, cl.squeeze_(1), ts.squeeze_(0))

            # loss = cosine_loss(edges, embed, ew)
            total_loss += loss.item()
    return total_loss / len(data_loader), validation_embeddings, origs, qs, agents, validation_classes, embeds, act_feats

# alidation_loss, ve, TSve, qs, ags


if __name__ == '__main__':
    device = "cpu"

    inC = 40

    nW = 8

    # folder_test = './oneGraphAll_test/'
    # files_test = os.listdir(folder_test)
    # files_test = [file for file in files_test if file.endswith('.pickle') and file.startswith('(')] #and (file not in files_test)

    folder_graph = './AAAI/data/1000_len_sims/graphs/train_means/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle') and file.startswith('(')] #and (file not in files_test)


    # fname = 'small_graphs.pth'
    valfolder = './AAAI/data/1000_len_sims/graphs/test_means/'
    files_val= os.listdir(valfolder)
    files_val = [file for file in files_val if file.endswith('.pickle') and file.startswith('(')]


    dataset = GraphDataset(folder_graph, files, 'train', 'time')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=nW)

    # dataset_test = GraphDataset(folder_test, files_test)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=nW)

    dataset_val = GraphDataset(valfolder, files_val, 'test', 'time')
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=nW)
# hidden count:  16 drops:  0.6 lr:  0.01 decays:  0.01
    # train_loss_end = []
    # test_loss_end = []
    for hC in [128]:#[8, 16]: #[8, 64]:#[8, 16, 32, 64]:#[8, 16, 32, 64]:
        for drops in [0.6]:#[0.4, 0.6]: #[0.2, 0.4, 0.6, 0.8]:#[0.2, 0.4, 0.6, 0.8]:
            for lrate in [0.01]:#[0.001, 0.01]:#[0.001, 0.05, 0.005, 0.01]: #[0.01, 0.05, 0.001, 0.005]:
                for decays in [5e-4]:#[5e-4, 0.01]:#[0.01, 0.0]: #[0]: #[5e-5, 0]:#[5e-4, 5e-5, 5e-3, 0.0]:
                    

                    print('hidden count: ', hC, 'drops: ', drops, 'lr: ', lrate, 'decays: ', decays)

                    actuals = []
                    final_loss = []
                    embeddings = []
                    outchannels = 1
                    # validation_loss = 100
                    # model = MLPNet(inC, hC, outchannels, drops).to(device)

                    model = GATNet(inC, hC, outchannels, drops, h=4).to(device)
                    optimizer = optim.AdamW(model.parameters(), lr=lrate, weight_decay=decays)
                    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
                    # losses = []
                    # embeds = []
                    # torch.autograd.set_detect_anomaly(True)
                    # acts = []
                    for epoch in range(1, 5):  # Number of epochs
                        # loss, embed, orig = train(model, dataloader, optimizer, device)
                        loss, embed, tsclass, pred_class, trainacts = train(model, dataloader, optimizer, device)
                        # embeds.append(embed)
                        # acts.append(orig)
                        print(f'Epoch {epoch}, Loss: {loss:.4f}')
                        if epoch % 4 == 0:
                            # In your main training loop:
                            # total_loss / len(data_loader), validation_embeddings, origs, qs, agents
                            validation_loss, ve, TSve, qs, ags, vc, emb, acts  = validate(model, dataloader_val, device)
                            # validation_loss, _, _, _, _  = validate(model, dataloader_test, device)
                            print(f'Validation Loss: {validation_loss:.4f}')
                    print('end train and val loss: ', loss, validation_loss)
    
    model.eval()
    encoder_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    torch.save(TSve, 'originals_time.pickle')
    torch.save(vc, 'model_out_time.pickle')
    torch.save(emb, 'embeddings_time.pickle')
    torch.save(acts, 'actuals_time.pickle')

    torch.save(tsclass, 'train_originals_time.pickle')
    torch.save(pred_class, 'train_model_out_time.pickle')
    torch.save(embed, 'train_embeddings_time.pickle')
    torch.save(trainacts, 'train_actuals_time.pickle')


    # torch.save(TSve, 'originals_means.pickle')
    # torch.save(vc, 'model_out_means.pickle')
    # torch.save(emb, 'embeddings_means.pickle')

    # CLIST =  ["Slow/Failure", "Slow/Success", "Fast/Failure", "Fast/Success"]
    
    
    # y_from_network = []
    
    # y = []


    # for vv, tsts in zip(vc, TSve):
    #     # print(vv, tsts)
    #     # break
    # # pdb.set_trace()
    #     for v in vv:
    #         y_from_network.append(v.tolist())
    #     y+=tsts[0].tolist()     
    # # pdb.set_trace()

    # y_pred_indices = np.argmax(y_from_network, axis=1)
    # cm = confusion_matrix(y, y_pred_indices)
    # # for y_ in y_from_network(v.)
    # # Plotting the confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=CLIST, yticklabels=CLIST)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.show()
    # f1_each_class = f1_score(y, y_pred_indices, average=None)
    # print("F1 Score for each class:", f1_each_class)

    # # Calculate macro-average F1 score (unweighted average across all classes)
    # f1_macro = f1_score(y, y_pred_indices, average='macro')
    # print("Macro-average F1 Score:", f1_macro)

    # # Calculate micro-average F1 score (weighted by support, considers overall precision and recall)
    # f1_micro = f1_score(y, y_pred_indices, average='micro')
    # print("Micro-average F1 Score:", f1_micro)

    # # Calculate weighted-average F1 score (weighted by the number of true instances for each class)
    # f1_weighted = f1_score(y, y_pred_indices, average='weighted')
    # print("Weighted-average F1 Score:", f1_weighted)


