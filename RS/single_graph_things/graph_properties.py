
import numpy as np 
import networkx as nx 
import os
import pickle 
import pdb 

if __name__ == '__main__':
    folder_graph = './RS/single_graph_things/finished_some_train/'
    device = "cpu"

    folder_graph_test = './RS/single_graph_things/finished_some_train/'
    # files_test = os.listdir(folder_graph_test)
    # files_test = [file for file in files_test if file.endswith('.pickle')]

    #  and (file not in files_test)
    # fname = 'small_graphs.pth'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle') and file.startswith('(')] #and (file not in files_test)
    radiuses = []
    diam = []
    clustcoeff = []
    degrees = []
    eccentrs = []
    densities = []
    triangles = []
    assorts = []

    for file in files: 


        fil =  open(folder_graph+file, 'rb')
        G = pickle.load(fil)
        fil.close()
        assorts.append(nx.degree_assortativity_coefficient(G))
        # pdb.set_trace()
        if nx.radius(G) == 0:
            continue 
        sorted_quals = np.sort(G.nodes[0]['quals'])
        # if sorted_quals[-1] - sorted_quals[-2] < 0.5:
        #     continue
        # if len(G.nodes) < 50:
        #     continue

        fil =  open(folder_graph_test+file, 'wb')
        pickle.dump(G, fil)
        fil.close()
            # pdb.set_trace()
        radiuses.append(nx.radius(G))
        diam.append(nx.diameter(G))
        eccentrs.append([x[1] for x in nx.eccentricity(G).items()])
        densities.append(nx.density(G))
        # pdb.set_trace()
        degrees.append([x[1] for x in G.degree()])
        triangles.append([x[1] for x in nx.triangles(G).items()])
        clustcoeff.append([x[1] for x in nx.clustering(G).items()])
        print(file, np.mean(G.degree()))
pdb.set_trace()   
# np.save('graph_params.npy', {'radius': tuple(radiuses), 'diameter': tuple(diam), 'eccentr': tuple(eccentrs), 'densities': tuple(densities), 'degrees': tuple(degrees),'triangeles': tuple(triangles), 'clustering': tuple(clustcoeff)})