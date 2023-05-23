import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb
from ast import literal_eval
import time 
import matplotlib.patches as pat
from params import *
# import pylab as pl

AGENT_COLORS = {'REST':'b', 'EXPLORE': 'r', 'ASSESS': 'k', 'DANCE': 'm', 
                'TRAVEL_HOME_TO_REST': 'y', 'TRAVEL_HOME_TO_DANCE': 'y', 'TRAVEL_SITE': 'y'}


folder = './sim_results/'
files = os.listdir(folder)
files = [file for file in files if file.endswith('.csv')]
files = np.sort(files)
lst = []
data_files = []
metadata_files = []

for i in range(0,len(files)-1,2):
    # pdb.set_trace()
    data_files.append(folder+files[i])
    metadata_files.append(folder+files[i+1])

# pdb.set_trace()
metadata_concat = pd.concat([pd.read_csv(f) for f in metadata_files ], ignore_index=True)
metadata_concat.site_qualities = metadata_concat.site_qualities.apply(literal_eval)
metadata_concat['max_qual'] = metadata_concat.site_qualities.apply(lambda x: np.max(x))
metadata_concat['success'] = (metadata_concat.max_qual - metadata_concat.site_converged) < 0.000001
# pdb.set_trace()

metadata_concat.to_csv(folder+'combined_results_metadata_longsim.csv')