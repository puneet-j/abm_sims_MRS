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
from torch_geometric.data import Data
# from newGraphDat import GraphDataset
from GraphDatAll import GraphDataset
from sklearn.manifold import TSNE
import matplotlib as mpl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
# import numpy as np
import scipy.io
# from graphDat import GraphDataset
# from ClassGraphDat import GraphDataset

torch.manual_seed(42)
 
# def visualize(h, data, title = "Scaatter Plot"):
#     colorlist = ['#e41a1c', '#984ea3', '#377eb8', '#4daf4a', '#ff7f00', '#ffff33', '#a65628']
#     z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

#     plt.figure(figsize=(10,10))
#     plt.xticks([])
#     plt.yticks([])
#     # alpha = [0.1 if G.nodes[node]["num_trajectories"] < LOWER_CONFIDENCE else 0.8 for node in G.nodes]
    


#     for class_number in range(data):
#         index_list = extract_nodes_by_class(data.y,class_number)
#         #a = [alpha[i] for i in index_list]
#         #plt.scatter(z[index_list, 0], z[index_list, 1], s=10, c=colorlist[class_number], alpha = a)
#         plt.scatter(z[index_list, 0], z[index_list, 1], s=10, c=colorlist[class_number], alpha = 0.7)

#     #plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     _ = plt.legend(bin_dict.values(),bbox_to_anchor=(1, 1), loc='upper left')
#     plt.title(title)
#     plt.show()

# def extract_nodes_by_class(data,class_number):
#     index_list = []
#     for index, item in enumerate(data):
#         if data[index] == class_number:
#             index_list.append(index)
#     return index_list

# class Encoder(torch.nn.Module):
#     def __init__(self, dim_in, dim_h, dim_o, pr):
#         super().__init__()
#         self.hidden_layer_1 = GCNConv(dim_in, 4 * dim_h)
#         self.hidden_layer_2 = GCNConv(4 * dim_h, 2 * dim_h)
#         self.hidden_layer_3 = GCNConv(2 * dim_h, dim_h)
#         self.linear = nn.Linear(dim_h, dim_o)
#         self.shortcut = nn.Linear(dim_in, dim_o)
#         self.pr = pr
#     def forward(self, x, edge_index):
#         h_init = x
#         h = F.dropout(x, p=self.pr, training = self.training)
#         h = self.hidden_layer_1(h, edge_index)
#         h = F.elu(h)
#         h = F.dropout(h, p=self.pr, training = self.training)
#         h = self.hidden_layer_2(h, edge_index)
#         h = F.elu(h)
#         h = F.dropout(h, p=self.pr, training = self.training)
#         h = self.hidden_layer_3(h, edge_index)
#         h = self.linear(h)
#         h = h + self.shortcut(h_init)
#         return h

# model = GATNet(inC, hC, outchannels, drops).to(device)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p):
        super(GATNet, self).__init__()
        # Define the first GAT convolution layer
        self.conv1 = GATConv(in_channels, hidden_channels*4, heads=8, dropout=p)
        # Define the second GAT convolution layer
        self.conv2 = GATConv(hidden_channels*4 * 8, hidden_channels*2, heads=8, concat=True, dropout=p)
        # Define the third GAT convolution layer
        self.conv3 = GATConv(hidden_channels*2 * 8, out_channels, heads=1, concat=True, dropout=p)
        # Define a linear layer to refine the outputs to the desired size
        # self.linear = nn.Linear(hidden_channels, out_channels)
        # self.shortcut = nn.Linear(in_channels, out_channels)
        self.p = p

    def forward(self, x, edge_index):
        # init = x
        # x, edge_index = data.x, data.edge_index
        # Apply dropout to the input features and pass through the first GAT layer
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the second GAT layer
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        # Pass through the third GAT layer
        x = self.conv3(x, edge_index)#F.dropout(x, p=self.p, training=self.training)
        # x = F.elu(self.conv3(x, edge_index))
        # Apply the linear layer
        # x = self.linear(x)
        # x = x + self.shortcut(init)
        return x
    
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

# def train(model, dataloader, optimizer, device):#, timeclasses):
    
#     model.train()
#     total_loss = 0
#     embeddings = []
#     origs = []
#     qs = []
#     agents = []
#     printlosses = []
#     for feat, edge, _, _, q, nA, ts in dataloader:
#         optimizer.zero_grad()
#         features = feat.squeeze_(0).to(device)
#         edges = edge.squeeze_(0).to(device)
#         embed = model(features, edges)
        
#         qs.append(q)
#         agents.append(nA)

#         # time_classes = times.squeeze_(0).to(device)
#         # succ_classes = succ.squeeze_(0).to(device)
#         # time_succ_class = torch.cat([time_classes, succ_classes], axis=-1)
#         # pdb.set_trace()
#         loss = loss_function(embed, ts.squeeze_(0).to(device)) #cosine_loss(data[1][0].to(device), embed)
#         origs.append(ts.detach())
#         # loss = cosine_loss(embed, classes) 
#         # loss = focal_loss(embed, data[2][0].to(device)) #cosine_loss(data[1][0].to(device), embed)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.detach().item()
#         printlosses.append(loss.detach().item())
#         if len(printlosses)%1000 == 0:
#             printlosses = printlosses[-1000:]
#             print(np.sum(printlosses)/1000.0)
#         embeddings.append(embed.detach())
#         # print(total_loss)/10.0
#     return total_loss / len(dataloader), embeddings, origs, qs, agents

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    embeddings = []
    origs = []
    # qs = []
    # agents = []
    # printlosses = []
    for feat, edge, edge_weight, _, _, _, _, tS in dataloader:
        optimizer.zero_grad()
        # pdb.set_trace()
        # print(np.shape(feat), np.shape(edges))
        # print('in main: ', np.shape(feat), np.shape(edges))
        features = feat.squeeze_(0).to(device)
        # globalInfo = glob.squeeze_(0).to(device)
        edges = edge.squeeze_(0).to(device)
        ew = edge_weight.squeeze_(0).to(device)
        # embed = model(torch.cat([globalInfo, features], dim=1), edges)
        origs.append(tS)
        # print(np.shape(embed), np.shape(data[2][0]))
        # print()
        # print(data[2][0][0], embed[0].detach())
        # loss = lossfunc(embed, data[2][0].to(device))
        # print(np.shape(embed), np.shape(data[2][0]), np.shape(data[0][0]), np.shape(data[1][0]))
        # loss = loss_function(embed, data[2][0].to(device)) #cosine_loss(data[1][0].to(device), embed)
        embed = model(features, edges)
        loss = cosine_loss(edges, embed, ew)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        embeddings.append(embed.detach())
        # print(total_loss)
    return total_loss / len(data_loader), embeddings, origs

# Copy code
def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    counter = 0 
    validation_embeddings = []
    # validation_classes = []
    qs = []
    agents = []
    origs = []
    with torch.no_grad():
        for feat, edge, ew, _, _, q, nA, ts in data_loader:
            counter += 1
            features = feat.squeeze_(0).to(device)
            edges = edge.squeeze_(0).to(device)
            embed = model(features, torch.empty((2,1), dtype=torch.int64))
            validation_embeddings.append(embed.detach())
            qs.append(q)
            agents.append(nA)
            origs.append(ts.detach())
            # if counter < 5:
            #     print(embed)
            loss = cosine_loss(edges, embed, ew)
            total_loss += loss.item()
    return total_loss / len(data_loader), validation_embeddings, origs, qs, agents



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


if __name__ == '__main__':
    device = "cpu"

    inC = 40

    nW = 8

    # fname = 'small_graphs.pth'
    valfolder = './RS/single_graph_things/25pGraphs_val/'
    files_val= os.listdir(valfolder)
    files_val = [file for file in files_val if file.endswith('.pickle')]

    folder_graph = './RS/single_graph_things/25pGraphs/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle') and file.startswith('(')] #and (file not in files_test)

    dataset = GraphDataset(folder_graph, files)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=nW)

    dataset_test = GraphDataset(valfolder, files_val)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=nW)

    # train_loss_end = []
    # test_loss_end = []
    for hC in [8, 16, 32, 64]:#[32]:#[8, 16]: #[8, 64]:#[8, 16, 32, 64]:#[8, 16, 32, 64]:
        for drops in [0.4, 0.6]:#[0.4, 0.6]: #[0.2, 0.4, 0.6, 0.8]:#[0.2, 0.4, 0.6, 0.8]:
            for lrate in [0.01]:#[0.001, 0.05, 0.005, 0.01]: #[0.01, 0.05, 0.001, 0.005]:
                for decays in [0.0, 0.01]:#[0.01, 0.0]: #[0]: #[5e-5, 0]:#[5e-4, 5e-5, 5e-3, 0.0]:
                    
# hidden count:  16 drops:  0.6 lr:  0.01 decays:  0.01
                    print('hidden count: ', hC, 'drops: ', drops, 'lr: ', lrate, 'decays: ', decays)

                    actuals = []
                    final_loss = []
                    embeddings = []
                    outchannels = 4
                    # validation_loss = 100
                    model = GATNet(inC, hC, outchannels, drops).to(device)
                    optimizer = optim.AdamW(model.parameters(), lr=lrate, weight_decay=decays)
                    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
                    # losses = []
                    # embeds = []
                    # torch.autograd.set_detect_anomaly(True)
                    # acts = []
                    for epoch in range(1, 5):  # Number of epochs
                        # loss, embed, orig = train(model, dataloader, optimizer, device)
                        loss, embed, tsclass = train(model, dataloader, optimizer, device)
                        # embeds.append(embed)
                        # acts.append(orig)
                        print(f'Epoch {epoch}, Loss: {loss:.4f}')
                        if epoch % 2 == 0:
                            # In your main training loop:
                            # total_loss / len(data_loader), validation_embeddings, origs, qs, agents
                            validation_loss, ve, TSve, qs, ags  = validate(model, dataloader_test, device)
                            # validation_loss, _, _, _, _  = validate(model, dataloader_test, device)
                            print(f'Validation Loss: {validation_loss:.4f}')
                    print('end train and val loss: ', loss, validation_loss)
    
    # model.eval()
    # encoder_state_dict = model.state_dict()
    # optim_state_dict = optimizer.state_dict()
    # torch.save(encoder_state_dict, './RS/single_graph_things/LATEST_'+str(decays)+'_'+str(lrate)+'_'+str(drops)+'_'+str(hC)+'_Finished_Large_LinNew_encoder_state_dict'+str(outchannels)+'.pth')
    # torch.save(optim_state_dict, './RS/single_graph_things/LATEST_'+str(decays)+'_'+str(lrate)+'_'+str(drops)+'_'+str(hC)+'_Finished_Large_LinNew_Optim_state_dict'+str(outchannels)+'.pth')

    # pdb.set_trace()
    # CLIST =  ["Slow/Failure", "Slow/Success", "Fast/Failure", "Fast/Success"]
    # X = []
    # y = []
    # for vv, tsts in zip(ve, TSve):
    # # pdb.set_trace()
    #     for v in vv:
    #         X.append(v.tolist())
    #     y+=tsts[0].tolist()     
    # # pdb.set_trace()
    # trainX = []
    # trainy = []
    # for vv, tsts in zip(embed, tsclass):
    # # pdb.set_trace()
    #     for v in vv:
    #         trainX.append(v.tolist())
    #     trainy+=tsts[0].tolist()           
    # pdb.set_trace()
    # np.save('./RS/single_graph_things/'+str(hC)+'_trainX.npy', np.array(trainX))
    # np.save('./RS/single_graph_things/'+str(hC)+'_trainy.npy', np.array(trainy))
    # np.save('./RS/single_graph_things/'+str(hC)+'_X.npy', np.array(X))
    # np.save('./RS/single_graph_things/'+str(hC)+'_y.npy', np.array(y))
    

    # # # X = []
    # est = GradientBoostingClassifier(n_estimators=200, max_depth=3) # n_estimators is the number of trees and max_depth is the maximum depth of any individual tree.
    # est.fit(trainX,trainy)

    # y_pred = est.predict(X)
    # cm = confusion_matrix(y, y_pred)#, labels = bin_dict.values())
    # # # CLIST =  bin_dict.values()
    # np.save('cm.npy', np.array(cm))
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=CLIST, yticklabels=CLIST)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix for 2-level GraphSage embedding')
    
    # arrdict = {'x':[], 'y':[], 'z':[], 'col':[], 'sz':[]}
    # for i in cm:
    #     pdb.set_trace*()
    #     arrdict['x'].append(i[0])
    #     arrdict['y'].append(i[1])
    #     arrdict['z'].append(i[2])
    #     # arrdict['col'].append(i[3])
    #     # arrdict['sz'].append(i[4])


    # scipy.io.savemat('./RS/single_graph_things/CM.mat', arrdict)

    # plt.show() 


#     COMMIT_THRESHOLD=0.5
#     MAX_DIST = 1000
#     # folder_graph = './RS/single_graph_things/finished_some/'
#     # files = os.listdir(folder_graph)
#     # files = [file for file in files if file.endswith('.pickle')]
#     arr = dict()
#     arr1 = []
#     arrpos = []
#     arrqual = []
#     arrcol = []
#     arr_global_info = []
#     arr_num_agents = []
#     # metadata = pd.read_csv('./graphsage_results/CDC/multiple_agent_env_results/metadata.csv')
#     for file in files_to_val[:-1]:
#         fil =  open(folder_graph+file, 'rb')
#         G = pickle.load(fil)
#         fil.close()
#         # pdb.set_trace()
#         nA = file.split(')')[-1][:2]
#         if nA[-1] == '_':
#             num_agents = int(file.split(')')[-1][0])
#         else:
#             num_agents = int(file.split(')')[-1][:2])

#         for node in G.nodes(data=True):
#             # pdb.set_trace()
#             if node[1]['x'] in arr:
#                 continue
#             else:
#                 # pdb.set_trace()
#                 new_color = get_color(node[1]['x'], node[1]['quals'], num_agents)
#                 arrpos.append(get_complete_poses(node[1]['poses']))
#                 arrqual.append(get_complete_quals(node[1]['quals']))
#                 arrcol.append(new_color)
#                 arr_global_info.append(get_complete_global(node[1]['global_info']))
#                 arr_num_agents.append(num_agents)
#                 # arr2.append([node[0], node[1]['x'], node[1]['colors'], node[1]['quals'], node[1]['poses'], node[1]['global_info'], num_agents])
#                 # pdb.set_trace()
#                 arr1.append(list(node[1]['x']))#+list(node[1]['global_info']))
#                 # arr1.append(list(node[1]['x'])+list(node[1]['global_info']))
#                 # arr.append([list(node[1]['x'])+[a for a in node[1]['global_info']]])
#                 arr[node[1]['x']] = node[1]['colors']
#     model.eval()
#     emb = []
#     arrtosave = []
#     # for i in arr1: 
#     for i in arr1: 
#         # pdb.set_trace()
#         emb.append(model(torch.tensor([i], dtype=torch.float), torch.empty((2,1), dtype=torch.int64)).detach().tolist()[0])
#         arrtosave.append(i)
    


#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     # np.shape(embeds[-1][0])
#     # e = embeds[-1][0]
#     arr = []
#     counter = 0
#     for ee, qq, cc in zip(emb, arrqual, arrcol):
#         counter += 1
#         # for id, (e, o, q) in enumerate(zip(ee, oo, qq)):
#             # pdb.set_trace()
#         if cc=='k':
#             color = 'k'
#             # print(id, o[0:4])
#             sz = 50
#         elif cc=='r':
#             color = 'r'
#             sz = 200
#         elif cc=='g':
#             color = 'g'
#             sz = 100
#         else:
#             color = 'b'
#             sz = 5
#         # print(e)
#         # break
#         arr.append([ee[0], ee[1], ee[2], color, sz])
#         if counter < 5000:
#             ax.scatter(ee[0], ee[1], ee[2], marker='.', c=color, s=sz)
#         # if counter == 10000:
#         #     break

#     import numpy as np
#     import scipy.io

#     # x = np.linspace(0, 2 * np.pi, 100)
#     # y = np.cos(x)
#     arrdict = {'x':[], 'y':[], 'z':[], 'col':[], 'sz':[]}
#     for i in arr:
#         arrdict['x'].append(i[0])
#         arrdict['y'].append(i[1])
#         arrdict['z'].append(i[2])
#         arrdict['col'].append(i[3])
#         arrdict['sz'].append(i[4])


#     scipy.io.savemat('./RS/single_graph_things/NEW_with_self_loops_with_residual_test_all_3d_embed.mat', arrdict)

# plt.show()
