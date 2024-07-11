
import pickle 
import numpy as np 
import networkx as nx 
import os 
import shutil 
import pdb 


TIME_LIMIT = 2500
SUCC_LIMIT = 0.95

def get_succtimeclass(s, t, fl):

    if s <= SUCC_LIMIT and t > TIME_LIMIT:
        return 0 # slow fail
    elif s > SUCC_LIMIT and t > TIME_LIMIT:
        return 1 # slow success
    elif s <= SUCC_LIMIT and t <= TIME_LIMIT:
        return 2 # fast fail
    elif s > SUCC_LIMIT and t <= TIME_LIMIT:
        return 3 # fast success
    else:
        print(s, t, fl)
        pdb.set_trace()

folder_main = './AAAI/data/quorum_sims_60_40_split/'
dataset = 'test'
# folder = folder_main + dataset + '/'
def get_ratio(time, succ):
    class_arr = []

    for times, successes in zip(time, succ):
        class_arr.append(get_succtimeclass(times, successes, ''))

    percent_class = [0]*4
    for i in class_arr:
        percent_class[i] += 1.0
    total = np.sum(percent_class)
    percent_class = [p/total for p in percent_class]
    return percent_class

if __name__ == '__main__':
    fil2 = open(folder_main + 'graphs/all'+dataset+'time.pickle', 'rb')
    metatime = pickle.load(fil2)
    fil2.close()
    fil2 = open(folder_main + 'graphs/all'+dataset+'succ.pickle', 'rb')
    metasucc = pickle.load(fil2)
    fil2.close()

    def round_function(x, d):
        new = []
        for r in x:
            new.append(np.round(r,decimals=d))
        # pdb.set_trace()
        return tuple(new)

    folder_graph = folder_main+'graphs/'+dataset+'/'
    new = folder_main+'graphs/'+dataset+'_means/'
    files = os.listdir(folder_graph)
    files = [file for file in files if file.endswith('.pickle') and file.startswith('(')]# and file.startswith('(0.934, 0.973, 0.131, 0.546)((200.0, 0.0), (0.0, 200.0), (-200.0, 0.0), (-0.0, -200.0))10')]
    maxquals = []
    minquals = []
    nSites = []
    siteDist = []

    counter = 0
    for file in files:
        # if file.startswith('(0.934, 0.973, 0.131, 0.546)((200.0, 0.0), (0.0, 200.0), (-200.0, 0.0), (-0.0, -200.0))10'):
            print(file[0:10], counter)
            counter += 1
            # if counter < 4000:
            #     continue
            # break
            fil =  open(folder_graph+file, 'rb')
            G = pickle.load(fil)
            fil.close()

            # print(file[-52:-50], counter)
            nodes_to_remove = []
            # nA = 5 if file[-52:-50] == ')5' else 10
            
            sorted_quals = np.sort(G.nodes[0]['quals'])
            maxquals.append(sorted_quals[-1])
            minquals.append(sorted_quals[0])
            nSites.append(len(sorted_quals))
            # pdb.set_trace()
            pos1=G.nodes[0]['poses'][0]
            siteDist.append(np.round(np.sqrt(pos1[0]**2 + pos1[1]**2)))
            # nx.set_node_attributes(G, nA, 'agents')
            nx.set_node_attributes(G, 0.0, 'AvgTime')
            nx.set_node_attributes(G, 0.0, 'AvgSucc')
            nx.set_node_attributes(G, 0.0, 'num_occur')
            nx.set_node_attributes(G, 0.0, 'ratio_classes')
            # print(G.nodes[0])
            for node in G.nodes(data=True):
                entry = round_function(node[1]['xold'], 3)
                # pdb.set_trace()
                if len(metatime[entry]) < 5:
                    nodes_to_remove.append(node[0])
                #     # G.remove_node(node[0])
                #     continue
                # pdb.set_trace()
                # try:
                if np.isnan(np.mean(metatime[entry])) or np.isnan(np.mean(metasucc[entry])):
                    pdb.set_trace()
                G.nodes[node[0]]['AvgTime'] = np.mean(metatime[entry])
                G.nodes[node[0]]['AvgSucc'] = np.mean(metasucc[entry])
                G.nodes[node[0]]['ratio_classes'] = get_ratio(metatime[entry], metasucc[entry])
                G.nodes[node[0]]['num_occur'] = len(metatime[entry])
                # except Exception as e:
                #     print(e)
                #     pdb.set_trace()

            for node in nodes_to_remove:
                G.remove_node(node)

            old_labels = list(G.nodes)
            new_labels = list(range(len(old_labels)))
            mapping = dict(zip(old_labels, new_labels))

            # # Relabel nodes
            G = nx.relabel_nodes(G, mapping)

            # print(list(G.nodes))
            # print(list(G.edges))
            # pdb.set_trace()
            # if sorted_quals[-1] - sorted_quals[-2] > 0.5:
            #     continue
            # if len(G.nodes) < 50:
            #     continue

            print('dumping this file')
            fil =  open(new+file, 'wb')
            pickle.dump(G, fil)   
            fil.close() 
        # else:
        #     print('skipped')

