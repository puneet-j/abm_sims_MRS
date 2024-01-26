import pickle
import networkx as nx
import os
import numpy as np
import pandas as pd
import pdb 

folder_graph = './graphs/graphsage_graph/'
new_metadata_file = folder_graph + 'metadata.csv'
print(os.listdir('.'))
files = os.listdir(folder_graph)
files = [file for file in files if file.endswith('.pickle')]
files = np.sort(files)
metadata_file = folder_graph + 'metadata.csv'
metadata = pd.read_csv(metadata_file)

for id, row in enumerate(metadata.iterrows()):
    fname = str(row[1][1]) + str(row[1][2]) + str(row[1][3]) + '.pickle'
    try:
        fil =  open(folder_graph+fname, 'rb')
        graph = pickle.load(fil)
        fil.close()
    except:
        continue
    # pdb.set_trace()


for id, node in enumerate(graph.nodes(data=True)):
    try:
        graph.nodes[id]['x'] = str(node[1]['x'])
        if np.isnan(node[1]['success']):
            graph.nodes[id]['success'] = 0
    except:
        pdb.set_trace()

G = nx.write_gexf(graph, folder_graph+fname+".gexf")
