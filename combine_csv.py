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
metadata_concat.to_csv('combined_results_metadata.csv')

'''
# data_concat = pd.concat([pd.read_csv(f) for f in data_files ], ignore_index=True)
# metadata_concat = pd.concat([pd.read_csv(f) for f in metadata_files ], ignore_index=True)

for i in range(0,len(files)-1,2):
    print(i)
    df = pd.read_csv(folder+files[i])
    metadata = pd.read_csv(folder+files[i+1])
    df.agent_sites = df.agent_sites.apply(literal_eval)
    df.agent_states = df.agent_states.apply(literal_eval)

    metadata.site_positions = metadata.site_positions.apply(literal_eval)
    metadata.site_qualities = metadata.site_qualities.apply(literal_eval)

    # pdb.set_trace()
    lst.append([list(metadata.site_positions[0]), list(metadata.site_qualities[0]), df.agent_sites.iloc[-1], df.agent_states.iloc[-1], df.time.iloc[-1]])
    # pdb.set_trace()

df_main = pd.DataFrame(lst,columns = ['site_pos', 'site_qual', 'agent_sites_end', 'agent_states_end', 'time_end'])

df_main.to_csv('combined_results.csv')
'''