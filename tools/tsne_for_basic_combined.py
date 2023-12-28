from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pdb
import copy
import pickle
from joblib import parallel_backend
import matplotlib.pyplot as plt
import seaborn as sns


folder_graph = './graphs/basic_test/'
fname = 'combined_data_basic.pickle'

fil =  open(folder_graph+fname, 'rb')
nodes = pickle.load(fil)
fil.close()
input_data = []
test_data = []
for node in nodes:
    # pdb.set_trace()
    input_data.append(node[:-2])
    test_data.append(node[-2:])

# pdb.set_trace()
tsne = TSNE()

with parallel_backend('threading', n_jobs=20):
    pcafit = pca.fit(input_data)

fnamecov = 'combined_data_basic_cov.pickle'
fil =  open(folder_graph+fnamecov, 'wb')
pickle.dump(pcafit, fil)   
fil.close() 


# fnamecov = 'combined_data_basic_cov.pickle'
# fil =  open(folder_graph+fnamecov, 'rb')
# pcafit = pickle.load(fil)   
# fil.close() 

nSites = 2
nAgents = 10

columnList = ['Time', 'Threshold']
for i in range(0,nSites):
    columnList.append('Site ' + str(i+1)+ ' Quality')
    columnList.append('Site ' + str(i+1)+ ' X')
    columnList.append('Site ' + str(i+1)+ ' Y')

for i in range(0,nAgents):
    columnList.append('Agent X')
    columnList.append('Agent Y')
    columnList.append('Agent State')
    columnList.append('Agent Site')

# pdb.set_trace()

# df = pd.DataFrame(pcafit, columns=columnList)
# 
cov = pcafit.get_covariance()
# cov_exclude_first = cov[1:]
# pdb.set_trace()
df = pd.DataFrame(cov)
df = df.drop(0, axis=1)
df = df.drop(0, axis=0)
sns.heatmap(df, 
            xticklabels=columnList[1:],
            yticklabels=columnList[1:])


plt.show()