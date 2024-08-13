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
from torch_geometric.data import Data, NeighborSampler
import networkx as nx
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
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
from torch_geometric.data import Data
# from newGraphDat import GraphDataset
# from GraphDatAll import GraphDataset
from GraphDatNodeWise import GraphDataset

from sklearn.manifold import TSNE
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# import numpy as np
import scipy.io
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
torch.manual_seed(42)
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p, h = 4):
        super(GATNet, self).__init__()
        # Define the first GAT convolution layer
        self.conv1 = GATConv(in_channels, hidden_channels*4, heads=h, dropout=p)
        # Define the second GAT convolution layer
        self.conv2 = GATConv(hidden_channels*4 * h, hidden_channels*2, heads=h, concat=True, dropout=p)
        # Define the third GAT convolution layer
        self.conv3 = GATConv(hidden_channels*2 * h, hidden_channels, heads=1, concat=True, dropout=p)
        # Define a linear layer to refine the outputs to the desired size
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.shortcut = nn.Linear(in_channels, hidden_channels)
        self.p = p
        # self.sigm = nn.Sigmoid()
        # self.ta = nn.Tanh()

    def forward(self, x, edge_index):
        init = x
        # x, edge_index = data.x, data.edge_index
        # Apply dropout to the input features and pass through the first GAT layer
        x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the third GAT layer
        x = self.conv3(x, edge_index)#F.dropout(x, p=self.p, training=self.training)
        x = x + self.shortcut(init)
        
        # x = F.dropout(x, p=self.p, training=self.training)
        # x = F.elu(self.conv3(x, edge_index))
        # Apply the linear layer
        x = F.dropout(self.linear(x), p=self.p, training=self.training)
        embed = x
        # x = F.relu(self.linear2(x))
        x = self.linear2(x)
        
        # x[:,0] = self.sigm(x[:,0])
        # x[:,0] = self.ta(x[:,0] + 1)/2.0
        return x, embed


def class_loss(reconstructed_x, original_x):
    arr = [1.0, 1.0, 1.0, 1.0]
    const = np.sum(arr)
    weights = torch.log(torch.tensor([const/a for a in arr], dtype=torch.float))
    return F.cross_entropy(reconstructed_x, original_x, weight=weights)

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
    alpha = 0.999999
    return alpha * cls + (1.0 - alpha) * cs

def round_function(x, d):
    new = []
    for r in x:
        new.append(np.round(r,decimals=d))
    # pdb.set_trace()
    return tuple(new)

def convert_to_undirected(directed_edges):
   undirected_edges = []
   # print(np.shape(directed_edges))
   # print(directed_edges.t().tolist())
   for u, v in directed_edges.t():
       undirected_edges.append([u, v])
       undirected_edges.append([v, u])
   # print(np.shape(undirected_edges))
   # Convert set back to list if needed
   return torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()


def convert_to_undirected_weights(directed_edge_weights):
   # print(np.shape(directed_edge_weights))
   d = directed_edge_weights.tolist()
   undirected_weights = d+d
   # print(np.shape(undirected_weights))
   # Convert set back to list if needed
   return torch.tensor(list(undirected_weights), dtype=torch.float)

def binarize_labels(labels, threshold):
    # return torch.tensor([1 if label > threshold else 0 for label in labels])

    binary_labels = []
    for l in labels:
        if torch.isnan(l):
            binary_labels.append(float('nan'))
            # print('got nan label')
        else:
            if l > threshold:
                binary_labels.append(1)
            else:
                binary_labels.append(0)

    return torch.tensor(binary_labels)

if __name__=='__main__':

    device = "cpu"

    inC = 54

    nW = 8

    folder_graph = './AAAI/data/lots_of_node_samples/train_new_random_sample_64/graphs/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle') and file.startswith('1')] #and (file not in files_test)


    dataset = GraphDataset(folder_graph, files, 'train', 'time')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=nW)

    nodelabels = pd.read_csv(folder_graph+'times.csv')
    # print(nodelabels['(0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0)'])

    node_labels_dict = dict()
    for n in nodelabels.columns:
        if n != 'Unnamed: 0':
            node_labels_dict[n] = nodelabels[n].tolist()

    # for n in node_labels_dict:
    #     print(n)
    #     break
    reg_loss = nn.MSELoss(reduction='mean')
    alpha = 1.0#0.000001#1.000#0.00001

    epochs = 5
    threshold = 400
    hC = 128
    drops = 0.0
    lrate = 0.001
    decays = 5e-4
    predictions_train = []
    actuals_train = []
    nodes_trained = []
    outchannels = 2
    model = GATNet(inC, hC, outchannels, drops, h=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lrate, weight_decay=decays)
    model.train()
    for epoch in range(0, epochs):
        allLoss = 0
        for feat, edge, ew in dataloader:
            
            features = feat.squeeze_(0)#.to(device)
            edges = convert_to_undirected(edge.squeeze_(0))#.to(device)
            edge_weights = convert_to_undirected_weights(ew.squeeze_(0))
            # edges = edge.squeeze_(0)#.to(device)
            # edge_weights = ew.squeeze_(0)
            # Create a tensor for labels, initially filled with NaN
            labels = torch.full((features.shape[0],), float('nan'), dtype=torch.float32)
            # print(np.shape(labels))
            # Assign labels to corresponding nodes based on features
            for i, feature_vector in enumerate(features):
                feature_tuple = str(round_function(feature_vector.tolist(), 3))
                # print(feature_tuple)
                if feature_tuple in node_labels_dict:
                    labels[i] = np.nanmean(node_labels_dict[feature_tuple])
                    # print(labels[i])
            
            binary_labels = binarize_labels(labels, threshold)

            data = Data(x=features, edge_index=edges, y=binary_labels, edge_attr=edge_weights)
            loader = NeighborLoader(
                        data,
                        num_neighbors=[4] * 2,
                        batch_size=32,
                        input_nodes=None
                    )#.to(device)

            loss_for_one_graph = 0.0
            for subdata in loader:
                subdata = subdata.to(device)
                # print(subdata)
                label_mask = ~torch.isnan(subdata.y) 
                if label_mask.any():
                    optimizer.zero_grad()
                    out, embed = model(subdata.x, subdata.edge_index)
            
                    mse_loss_val = F.cross_entropy(out[label_mask], subdata.y[label_mask].long())
                
                    predictions_train.append(torch.argmax(out[label_mask], dim=1).detach().numpy())
                    actuals_train.append(subdata.y[label_mask].detach().numpy())
                    nodes_trained.append(subdata.x[label_mask].detach().numpy())
                    mse_loss_val.backward()
                
                    # print('cos loss only: ', cos_loss_val)        
                    loss_for_one_graph += mse_loss_val.detach()
                    optimizer.step()
                    
            allLoss += loss_for_one_graph
        print('EPOCH ', epoch, ' : ', allLoss)
    model.eval()
    # Assuming encoder and decoder are your model's components
    encoder_state_dict = model.state_dict()
    # decoder_state_dict = model.decoder.state_dict()
    # Save the state dictionaries
    torch.save(encoder_state_dict, './AAAI/64_train_model_new_random_undirected_classification.pth')
    # break
    valfolder = './AAAI/data/lots_of_node_samples/test_new_random_sample_64/graphs/'
    files_val= os.listdir(valfolder)
    files_val = [file for file in files_val if file.endswith('.pickle') and file.startswith('1')]

    dataset_val = GraphDataset(valfolder, files_val, 'test', 'time')
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=nW)
    nodelabels = pd.read_csv(valfolder +'times.csv')

    node_labels_dict = dict()
    for n in nodelabels.columns:
        if n != 'Unnamed: 0':
            node_labels_dict[n] = nodelabels[n].tolist()

    # for n in node_labels_dict:
    #     print(n)
    #     break

    predictions = []
    actual_labels = []
    nodes_labeled = []
    output_label = dict()
    model.eval()

    with torch.no_grad():
        for feat, edge, ew in dataloader_val:
            features = feat.squeeze_(0)#.to(device)
            edges = edge.squeeze_(0)#.to(device)
            edge_weights = ew.squeeze_(0)
            # Create a tensor for labels, initially filled with NaN
            labels = torch.full((features.shape[0],), float('nan'), dtype=torch.float32)
            # print(np.shape(labels))
            # Assign labels to corresponding nodes based on features
            for i, feature_vector in enumerate(features):
                feature_tuple = str(round_function(feature_vector.tolist(), 3))
                # print(feature_tuple)
                if feature_tuple in node_labels_dict:
                    labels[i] = np.nanmean(node_labels_dict[feature_tuple])
                    # print(labels[i])
                # break
            binary_labels = binarize_labels(labels, threshold)
            data = Data(x=features, edge_index=edges, y=binary_labels, edge_attr=edge_weights)
            loader = NeighborLoader(
                        data,
                        num_neighbors=[4] * 2,
                        batch_size=32,
                        input_nodes=None,
                    )

            loss_for_one_graph = 0
            for subdata in loader:
                # print(subdata)
                # print(subdata)
                out, embed = model(subdata.x, subdata.edge_index)

                label_mask = ~torch.isnan(subdata.y) 
                if label_mask.any():
                    nodes_labeled.append(subdata.x[label_mask].cpu().numpy())
                    # predictions.append(out[label_mask].cpu().numpy())
                    predictions.append(torch.argmax(out[label_mask], dim=1).cpu().numpy())
                    actual_labels.append(subdata.y[label_mask].cpu().numpy())
    # Confusion Matrix
    plt.figure(0)
    all_predictions = np.concatenate(predictions)
    all_actuals = np.concatenate(actual_labels)
    cm = confusion_matrix(all_actuals, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    plt.figure(1)
    all_predictions_train = np.concatenate(predictions_train)
    all_actuals_train = np.concatenate(actuals_train)
    cm = confusion_matrix(all_actuals_train, all_predictions_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # inp = []
    # outp = []
    # nodes_ = []



    # for p, a, n in zip(predictions, actual_labels, nodes_labeled):
    #     for pp, aa, nn in zip(p, a, n):
    #         inp.append(aa)
    #         outp.append(pp[0])
    #         nodes_.append(tuple(nn))
            

    # df = pd.DataFrame({'nodes': nodes_, 'inputs': inp, 'outputs': outp})
    # grouped_data = df.groupby(by=['nodes'], as_index=False).agg(list)#.reset_index()
    # # print(grouped_data.nodes)

    # # print(df.inputs.tolist())
    # plt.figure(0)
    # # for row in 
    # error_fill1 = []
    # error_fill2 = []
    # meanpoints = []
    # ids = []
    # meanpreds = []
    # medianpoints = []
    # errpred25 = []
    # errpred75 = []
    # medianpreds = []
    # for id, (i, o, n) in enumerate(zip(grouped_data.inputs, grouped_data.outputs, grouped_data.nodes)):
    #     # print(i)
    #     # print(o)
    #     # break
    #     #  why is it missing from dict??
    #     # print(tuple(n), node_labels_dict[str(tuple(n))])
    #     # break
    #     i = list(node_labels_dict[str(tuple(n))])
    #     medianpoints.append(np.nanmedian(i))
    #     perc25 = np.nanpercentile(i, 25)
    #     # print(perc25)
    #     perc75 = np.nanpercentile(i, 75)
    #     error_fill1.append(perc25)
    #     error_fill2.append(perc75)
    #     meanpoints.append(np.nanmean(i))
    #     ids.append(id)
    #     meanpreds.append(np.nanmean(o))
    #     errpred25.append(np.nanpercentile(o, 25))
    #     errpred75.append(np.nanpercentile(o, 75))
    #     medianpreds.append(np.nanmedian(o))



    # meanpoints = np.array(meanpoints)
    # arr = np.argsort(meanpoints)
    # meanpoints = meanpoints[arr]
    # # print(meanpoints)
    # # arr = ids
    # error_fill2 = np.array(error_fill2)[arr]
    # error_fill1 = np.array(error_fill1)[arr]
    # medianpoints = np.array(medianpoints)[arr]
    # meanpreds = np.array(meanpreds)[arr]
    # ids = np.array(ids)
    # errpred25 = np.array(errpred25)[arr]
    # errpred75 = np.array(errpred75)[arr]
    # medianpreds = np.array(medianpreds)[arr]
    # # print(errpred25[0] - medianpreds[0])
    # # print(medianpreds-errpred25)
    # plt.errorbar(ids, medianpreds, yerr=[medianpreds-errpred25, errpred75-medianpreds], fmt='', lw = 0.0, elinewidth=2.0)
    # # print(arr)
    # plt.plot(ids, medianpoints, 'ro-')
    # # print(medianpoints[0] - error_fill1[0], medianpoints[0] + error_fill2[0])
    # plt.fill_between(ids, error_fill1, error_fill2, color='red', alpha=0.2)
    # plt.plot(ids, meanpreds, 'b+')
    # plt.hlines(0.0, -5, len(ids) + 5, 'k', 'dashed')

    # plt.show()

    # folder_graph = './AAAI/data/lots_of_node_samples/test_new_random_sample_64/graphs/'
    # files = os.listdir(folder_graph)
    # files = [file for file in files if file.endswith('.pickle') and file.startswith('1')] #and (file not in files_test)

    # # nodelabels = pd.read_csv(folder_graph+'times.csv')
    # # # print(nodelabels['(0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0, 0.667, 1.0, 1.0, 0.0)'])

    # node_labels_dict = dict()
    # for n in nodelabels.columns:
    #     if n != 'Unnamed: 0':
    #         node_labels_dict[n] = nodelabels[n].tolist()

    # # for n in node_labels_dict:
    #     # print(n)
    #     # break


    # inp = []
    # outp = []
    # nodes_ = []



    # for p, a, n in zip(predictions_train, actuals_train, nodes_trained):
    #     for pp, aa, nn in zip(p, a, n):
    #         inp.append(aa)
    #         outp.append(pp[0])
    #         nodes_.append(tuple(nn))
            

    # df = pd.DataFrame({'nodes': nodes_, 'inputs': inp, 'outputs': outp})
    # grouped_data = df.groupby(by=['nodes'], as_index=False).agg(list)#.reset_index()
    # # print(grouped_data.nodes)

    # # print(df.inputs.tolist())
    # plt.figure(0)
    # # for row in 
    # error_fill1 = []
    # error_fill2 = []
    # meanpoints = []
    # ids = []
    # meanpreds = []
    # medianpoints = []
    # errpred25 = []
    # errpred75 = []
    # medianpreds = []
    # for id, (i, o, n) in enumerate(zip(grouped_data.inputs, grouped_data.outputs, grouped_data.nodes)):
    #     # print(i)
    #     # print(o)
    #     # print(n)
    #     # break
    #     #  why is it missing from dict??
    #     # print(tuple(n), node_labels_dict[str(tuple(n))])
    #     # break
    #     i = list(node_labels_dict[str(tuple(n))])
    #     medianpoints.append(np.nanmedian(i))
    #     perc25 = np.nanpercentile(i, 25)
    #     # print(perc25)
    #     perc75 = np.nanpercentile(i, 75)
    #     error_fill1.append(perc25)
    #     error_fill2.append(perc75)
    #     meanpoints.append(np.nanmean(i))
    #     ids.append(id)
    #     meanpreds.append(np.nanmean(o))
    #     errpred25.append(np.nanpercentile(o, 25))
    #     errpred75.append(np.nanpercentile(o, 75))
    #     medianpreds.append(np.nanmedian(o))


    #     meanpoints = np.array(meanpoints)
    #     arr = np.argsort(meanpoints)
    #     meanpoints = meanpoints[arr]
    #     # print(meanpoints)

    #     error_fill2 = np.array(error_fill2)[arr]
    #     error_fill1 = np.array(error_fill1)[arr]
    #     medianpoints = np.array(medianpoints)[arr]
    #     meanpreds = np.array(meanpreds)[arr]
    #     ids = np.array(ids)
    #     errpred25 = np.array(errpred25)[arr]
    #     errpred75 = np.array(errpred75)[arr]
    #     medianpreds = np.array(medianpreds)[arr]
    #     # print(errpred25[0] - medianpreds[0])
    #     # print(medianpreds-errpred25)
    #     yErr=[medianpreds-errpred25, errpred75-medianpreds]
    #     plt.errorbar(ids, medianpreds, yerr=yErr, fmt='')
    #     # plt.errorbar(ids, np.clip(meanpreds, a_min=0.001, a_max = 10000), yerr=[np.clip(medianpreds-errpred25, a_min=0.001, a_max=10000), np.clip(errpred75-medianpreds, a_min=0.001, a_max=10000)], fmt='')
    #     # print(arr)
    #     plt.plot(ids, medianpoints, 'ro-')
    #     # print(medianpoints[0] - error_fill1[0], medianpoints[0] + error_fill2[0])
    #     plt.fill_between(ids, error_fill1, error_fill2, color='red', alpha=0.2)
    #     plt.hlines(0.0, -10, len(ids) + 10, 'k', 'dashed')
    #     plt.plot(ids, meanpreds, 'b+')
    #     plt.show()
    #     # for m, e1, e2 in zip(medianpreds, errpred25, errpred75):
    #     #     if m-e1 <= 0 :
    #     #         print(m, e1, e2)
    #     #         break
