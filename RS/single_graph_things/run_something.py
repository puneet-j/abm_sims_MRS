import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv, GraphConv, GraphSAGE
import torch
# from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborSampler
import networkx as nx 
from torch.utils.data import Dataset, DataLoader
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
import pickle
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'tools'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from sklearn.neighbors import kneighbors_graph
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ClassGraphDat import GraphDataset
torch.manual_seed(42)
import psutil

TIME_LIMIT =  400

# def round_function(x, d):
#     new = []
#     for r in x:
#         new.append(np.round(r,decimals=d))
#     # pdb.set_trace()
#     return tuple(new)

def get_class(a):
    if a > TIME_LIMIT:
        retVal = torch.tensor(0, dtype=torch.long)
        return retVal
    else:
        retVal = torch.tensor(1, dtype=torch.long)
        return retVal
    # arr = []
    # if a > TIME_LIMIT:
    #     arr.append(0)
    # else:
    #     arr.append(1)

    # try:
    #     retVal = torch.tensor(arr, dtype=torch.long)
    #     return retVal
    # except:
    #     pdb.set_trace()
    # edges

class GraphEncoderWithResidual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoderWithResidual, self).__init__()
        # self.conv1 = SAGEConv(in_channels, hidden_channels*4, flow='target_to_source', root_weight=False)
        # self.conv2 = SAGEConv(hidden_channels*4, hidden_channels*2, flow='target_to_source', root_weight=False)
        # self.conv3 = SAGEConv(hidden_channels*2, hidden_channels, flow='target_to_source', root_weight=False)
        self.conv1 = SAGEConv(in_channels, hidden_channels*4, flow='source_to_target', root_weight=False)
        self.conv2 = SAGEConv(hidden_channels*4, hidden_channels*2, flow='source_to_target', root_weight=False)
        self.conv3 = SAGEConv(hidden_channels*2, hidden_channels, flow='source_to_target', root_weight=False)
        self.lin1 = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.lin2 = nn.Linear(int(hidden_channels/2), out_channels)
        # Linear transformation to match dimensions for residual connection
        self.shortcut = nn.Linear(in_channels, out_channels)
        # self.sm = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = F.dropout(F.relu(self.lin1(x)), p=0.2)
        x = self.lin2(x)
        # Applying shortcut and adding it to the output of conv3
        identity = self.shortcut(identity)
        x += identity  # Element-wise addition
        # x = self.sm(x)
        return x
    
def cosine_loss(adj, emb):
    norms = torch.norm(emb, p=2, dim=1, keepdim=True)
    normalized_embeddings = emb / norms.clamp(min=1e-4)
    cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
    target_adjacency = torch.zeros_like(cosine_similarity_matrix)
    target_adjacency[adj[0], adj[1]] = 1
    # Directly using logits; no need to apply sigmoid.
    return F.binary_cross_entropy_with_logits(cosine_similarity_matrix, target_adjacency)

# from memory_profiler import profile

# @profile
def cosine_loss_batchwise(adj, emb, eDict):
    # Normalize the embeddings along the last dimension
    norms = torch.norm(emb, p=2, dim=-1, keepdim=True)
    normalized_embeddings = emb / norms.clamp(min=1e-4)
    # pdb.set_trace()
    # Calculate cosine similarity in batches to save memory
    batch_size = 10  # Adjust batch size based on your GPU/CPU memory
    losses = 0
    num_nodes = emb.size(0)
    
    # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    for i in range(0, num_nodes, batch_size):
        endnum = min(i + batch_size, num_nodes)
        batch_emb = normalized_embeddings[i:endnum]
        batch_cosine_sim = torch.matmul(batch_emb, normalized_embeddings.transpose(0, 1))
        
        # Creating a target matrix for the current batch
        batch_adj = torch.zeros_like(batch_cosine_sim)
        
        # Filling the target adjacency matrix only for relevant entries
        # We iterate over all edges and check if they fall into the current batch
        currentnodeIDs = list(range(i,endnum))
        # print(currentnodeIDs)
        for nodeIDnow in currentnodeIDs:
            # print(nodeIDnow)
            # print(eDict)
            for target in eDict[nodeIDnow]:
                batch_adj[nodeIDnow-i, target] = 1
        # for edge in range(adj[0].size(0)):
            # if adj[0][edge] >= i and adj[0][edge] < end:
            #     src_index = adj[0][edge] - i  # Adjust index for the current batch
            #     tgt_index = adj[1][edge]
            #     batch_adj[src_index, tgt_index] = 1
            #     batch_adj[src_index, tgt_index] = 1  # Assuming undirected graph for symmetry

        # Compute loss for this batch using binary cross-entropy with logits
        batch_loss = F.binary_cross_entropy_with_logits(batch_cosine_sim, batch_adj)
        losses += batch_loss
        # print('RAM Used (GB) after batch:',i,  psutil.virtual_memory()[3]/1000000000)

    # print('RAM Used (GB) after batch processing:', psutil.virtual_memory()[3]/1000000000)

    # Average the losses across batches
    total_loss = losses/batch_size

    return total_loss


def train(model, optimizer, device, n, e, eDict):
    model.train()
    total_loss = 0
    embeddings = []
    origs = []
    # lossfunc = nn.CrossEntropyLoss()
    # print(data_loader)
    # printlosses = []
    # for dat in data_loader:
    # pdb.set_trace()
    # print(np.shape(nodes), np.shape(edges), nodes, edges)
    # n = dat[0] #torch.tensor(dat[1], dtype = torch.float) #nodes.squeeze_(0)
    # e = dat[1] #torch.tensor(dat[2], dtype = torch.long) #edges.squeeze_(0)
    # classes = dat[0] #torch.tensor(dat[2], dtype = torch.long)
    # class_now = get_class(meta, n)
    # print(np.shape(nodes), np.shape(edges), np.shape(class_now))

    optimizer.zero_grad()
    # adjs = [adj.to(device) for adj in adjs]
    # pdb.set_trace()
    # e = adjs[0][0]
    # optimizer.zero_grad()
    # out = model(data.x[n_id], adjs[0][0])
    # loss = F.cross_entropy(out, data.y[n_id[:batch_size]])
    # n = nt[n_id]
    try:
        if n.dim() == 1:
            n = n.unsqueeze(1)

        embed = model(n, e)

    except Exception as e:
        pdb.set_trace()
    # print(eDict)
    # print(len(data[0][0]))
    # pdb.set_trace()
    # print(np.shape(class_now), class_now)
    # if np.mean(class_now.tolist()) < 0.01:
    #     pdb.set_trace()
    loss = cosine_loss_batchwise(e, embed, eDict.copy()) #cosine_loss(data[1][0].to(device), embed)
    # loss = focal_loss(embed, data[2][0].to(device)) #cosine_loss(data[1][0].to(device), embed)
    loss.backward()
    optimizer.step()
    total_loss += loss.detach().item()
    # printlosses.append(loss.detach().item())
    # if len(printlosses)%10000 == 0:
    #     printlosses = printlosses[-10000:]
    #     print(np.sum(printlosses)/10000.0)
    # embeddings.append(embed.detach())
    # print(total_loss)/10.0

    return total_loss, embeddings, origs

# timeClasses = dict()
# for n in meanTimes:
#     timeClasses[n] = get_class(meanTimes[n])

# times_ = []
# for t in timeClasses.values():
#     times_.append(t)
def sample_neighbors(adj_list, nodes, batch_size, num_neighbors):
    # Randomly select batch_size nodes
    all_nodes = list(range(0,len(nodes)))
    batch_nodes = np.random.choice(all_nodes, batch_size, replace=False)
    
    sampled_subgraph = set()
    sampled_edges = []
    # For each node in the batch, sample num_neighbors from its neighbors
    for node in batch_nodes:
        neighbors = adj_list[node]
        if len(neighbors) > num_neighbors:
            # Sample num_neighbors if there are enough neighbors
            sampled_neighbors = np.random.choice(neighbors, num_neighbors, replace=False)
        else:
            # Otherwise take all neighbors
            sampled_neighbors = neighbors

        sampled_subgraph.update(sampled_neighbors)
        sampled_subgraph.add(node)  # Also add the node itself to the subgraph

        for neighbor in sampled_neighbors:
                sampled_edges.append((node, neighbor))
    
    sampled_nodes_tensor = torch.tensor(list(sampled_subgraph), dtype=torch.long)
    sampled_edges_tensor = torch.tensor(sampled_edges, dtype=torch.long)
    return sampled_nodes_tensor, sampled_edges_tensor





def _main_():
    folder_graph = './'
    device = "cpu"
    # fname = 'small_graphs.pth'
    # files = os.listdir(folder_graph)
    # files = [file for file in files if file.endswith('.pickle')]
    files = ['alledges', 'allsucc', 'alltime', 'edgesList', 'nodeIDs']
    hC = 10
    inC = 40
    outchannels = 3

    # fil2 = open('MULTIPLEalltime.pickle', 'rb')
    # times = pickle.load(fil2)
    # fil2.close()

    # fil2 = open('MULTIPLEallsucc.pickle', 'rb')
    # successes = pickle.load(fil2)
    # fil2.close()

    fil2 = open('./RS/single_graph_things/edgeList.pickle', 'rb')
    edges = pickle.load(fil2)
    fil2.close()

    fil2 = open('./RS/single_graph_things/nodeIDs.pickle', 'rb')
    nodes = pickle.load(fil2)
    fil2.close()

    # meanTimes = dict()
    # for n in times:
    #     meanTimes[n] = np.mean(times[n])

    nodesList = []
    for n in nodes:
        nodesList.append(n)

    from collections import defaultdict
    edgeDict = defaultdict(list)
    for e in edges:
        # if e[0] not in edgeDict:
            # edgeDict[e[0]] = e[1]
            # pdb.set_trace()
        edgeDict[e[0]].append(e[1])


    # pprint(edgeDict)


    # dataset = GraphDataset(folder_graph, files)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
    # dataloader = NeighborSampler(torch.tensor(edges, dtype=torch.long).t().to(device), sizes=[10, 10, 10], batch_size=1, shuffle=True, num_workers=12)

    nodeTensor = torch.tensor(nodesList, dtype=torch.float).to(device)
    edgeTensor = torch.tensor(edges, dtype=torch.long).t().to(device)


    actuals = []
    final_loss = []
    embeddings = []
    # dataloader = [torch.tensor(nodesList, dtype=torch.float).to(device), torch.tensor(edges, dtype=torch.long).t().to(device)]
    model = GraphEncoderWithResidual(in_channels=inC, hidden_channels=hC, out_channels=outchannels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    losses = []
    embeds = []
    acts = []

    # print(edgeDict[0])

    for epoch in range(1, 30):  # Number of epochs
        # sampled_nodes, sampled_edges = sample_neighbors(edges, nodesList, batch_size=10, num_neighbors=10)
        # sampled_features = nodeTensor[sampled_nodes]
        loss, embed, orig = train(model, optimizer, device, nodeTensor, edgeTensor, edgeDict)
        # embeds.append(embed)
        # acts.append(orig)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
        # losses.append(loss)
        # scheduler.step(loss)
    # epoch_loss = loss / len(dataloader)
    # embeddings.append(embeds)
    # final_loss.append(losses)
    # actuals.append(acts)
    model.eval()
    # Assuming encoder and decoder are your model's components
    encoder_state_dict = model.state_dict()
    # decoder_state_dict = model.decoder.state_dict()
    # Save the state dictionaries
    torch.save(encoder_state_dict, './RS/single_graph_things/Embed_encoder_state_dict'+str(outchannels)+'.pth')
    # torch.save(decoder_state_dict, './CDC/decoder_state_dict'+str(outchannels)+'.pth')
            # print(np.shape(embed), np.shape(data[2][0]), np.shape(data[0][0]), np.shape(data[1][0]))


_main_()