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
# 5, 20, 50, 100, 200

AGENT_COLORS = {'REST':'b', 'EXPLORE': 'r', 'ASSESS': 'k', 'DANCE': 'm', 
                'TRAVEL_HOME_TO_REST': 'y', 'TRAVEL_HOME_TO_DANCE': 'y', 'TRAVEL_SITE': 'y'}

AGENT_NUM_COLORS = {5:'k', 20: 'b', 50: 'm', 100: 'r', 200: 'y'}#, 'TRAVEL_HOME_TO_DANCE': 'y', 'TRAVEL_SITE': 'y'

SITE_NUM_SHAPES = {2:'o', 3:'v', 4:'s'}

# markersize=df.distances[i]/10.0, alpha=100.0/df.distances[i]
DIST_SIZES = {100:10.0, 200:20.0}
DIST_ALPHAS = {100:1.0, 200:0.5}

def get_mean_no_inf(x):
    result = [np.mean(x) if np.sum(x)!=0 else 0]
    return result[0]

df = pd.read_csv('../sim_results/combined_results_metadata_longsim.csv')

df.site_qualities = df.site_qualities.apply(literal_eval)
df.site_positions = df.site_positions.apply(literal_eval)

df['success_01'] = df.success.apply(lambda x: int(x))
df['distances'] = df.site_positions.apply(lambda x: tuple(np.round(np.sqrt(np.power(np.array(x)[:,0],2) + np.power(np.array(x)[:,1],2)),-1)))
df['qual_diff'] = df.site_qualities.apply(lambda x: np.max(x) - np.sort(x)[-2])

# pdb.set_trace()
df = df.groupby(by=['qual_diff', 'distances', 'num_agents', 'num_sites'], as_index=False).agg(lambda x: x.tolist())
# df = df.groupby(by=['site_qualities', 'num_agents', 'distances'], as_index=False).agg(lambda x: x.tolist())
# plt.plot()
df['meanSuccess'] = df.success_01.apply(lambda x: get_mean_no_inf(x))
# pdb.set_trace()
df['meanTime'] = df.time_converged.apply(lambda x: get_mean_no_inf(x))
df['distances'] = df['distances'].apply(lambda x: x[0])
df = df.drop(df[df.meanTime==35000].index)
# df['num_a']

df = df.reset_index()

# df = df.groupby(by=['site_qualities', 'num_agents'], as_index=False).agg(lambda x: x.tolist())

# pdb.set_trace()

plt.figure(1)
for i, na in enumerate(df.num_agents):
    print(i)
    # pdb.set_trace()
    try:
        plt.plot(df.qual_diff[i], df.meanSuccess[i], AGENT_NUM_COLORS[na]+SITE_NUM_SHAPES[df.num_sites[i]], 
                markersize=df.distances[i]/10.0, alpha=100.0/df.distances[i])
        
    except:
        pdb.set_trace()
        
    plt.xlabel('quality difference')
    plt.ylabel('mean success')
    plt.title('mean success vs qual diff')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    # markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in AGENT_NUM_COLORS.values()]
    # plt.legend(markers, AGENT_NUM_COLORS.keys(), numpoints=1, loc='lower right')


plt.figure(2)
for i, na in enumerate(df.num_agents):
    try:
        plt.plot(df.qual_diff[i], df.meanTime[i], AGENT_NUM_COLORS[na]+SITE_NUM_SHAPES[df.num_sites[i]], 
                 markersize=df.distances[i]/10.0, alpha=100.0/df.distances[i])
    
    except:
        pdb.set_trace()
    plt.xlabel('quality difference')
    plt.ylabel('mean time')
    plt.title('mean time vs qual diff')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-500, 35500]) 
    # plt.ylim([-200, 15500.0])
    # plt.ylim([-200, 1550.0])
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in AGENT_NUM_COLORS.values()]
    markers.append(plt.Line2D([0,0],[0,0], marker='.', linestyle='',markersize = 0.01))
    for s, m in SITE_NUM_SHAPES.items():
        markers.append(plt.Line2D([0,0],[0,0], marker=m, linestyle=''))
    markers.append(plt.Line2D([0,0],[0,0], marker='.', linestyle='',markersize = 0.01))
    for (s, m) in zip(DIST_SIZES.values(), DIST_ALPHAS.values()):
        markers.append(plt.Line2D([0,0],[0,0], marker='o', linestyle='', markersize = s, alpha = m))
    # legend = []
    keys1 = [key for key in AGENT_NUM_COLORS.keys()]
    keys2 = [key for key in SITE_NUM_SHAPES.keys()]
    keys3 = [key for key in DIST_SIZES.keys()]
    emptykey = ['----------']
    # pdb.set_trace()
    legend = keys1 + emptykey + keys2 + emptykey + keys3
    plt.legend(markers, legend, numpoints=1)

plt.show()