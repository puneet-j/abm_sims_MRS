import torch
import networkx as nx 
from torch.utils.data import Dataset, DataLoader
import pickle
import pdb 
import numpy as np
from torch_geometric.data import Data

# STATES_LIST = ['RECRUIT', 'ASSESS', 'TRAVEL_HOME_TO_RECRUIT', 'TRAVEL_SITE', 'OBSERVE', 'EXPLORE', 'TRAVEL_HOME_TO_OBSERVE']
# STATES = {'RECRUIT':0.0/6.0, 'ASSESS':1.0/6.0, 'TRAVEL_HOME_TO_RECRUIT':2.0/6.0, 'TRAVEL_SITE':3.0/6.0, 
#           'OBSERVE':4.0/6.0, 'EXPLORE':5.0/6.0, 'TRAVEL_HOME_TO_OBSERVE':6.0/6.0}

class GraphDataset(Dataset):
    def __init__(self, dat):#folder, file_paths):
        fil =  open(dat, 'rb')
        self.data = pickle.load(fil)
        fil.close()
        # self.data = pickle.load(dat)
        # self.folder = folder 
        
    def __len__(self):
        return len(self.data['id'])

    def get_complete_global(self, a, l):
        arr = [0]*4
        nA = np.ceil(l/4.0)
        for i in range(0,len(a)):
            # pdb.set_trace()
            arr[i] = a[i]/nA
        # print(arr)
        # pdb.set_trace()
        return arr


    # state_dict = {'id':[], 'feat':[], 'edges':[], 'classes':[], 'global_info':[]}

    def __getitem__(self, idx):
        f = self.data['feat'][idx]#.squeeze_(0)
        e = self.data['edges'][idx]#.squeeze_(0)
        c = self.data['classes'][idx]#.squeeze_(0)
        g = self.data['global_info'][idx]#.squeeze_(0)
        # print('in graphdata: ', np.shape(f), np.shape(e))
        # print('in graphdat: ', np.shape(self.data['feat'][idx].squeeze_(0)), np.shape(self.data['edges'][idx].squeeze_(0)))
        # return Data(x=torch.cat([g, f], dim=1), edge_index=e, is_directed=True, y=c)
        return f, e, c, g 
        #self.dat[idx]['feat'], self.dat[idx][]
        # return node_features, edge_index, classes
        