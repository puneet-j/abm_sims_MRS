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

df = pd.read_csv('./sim_results/combined_results_metadata_longsim.csv')

df.site_qualities = df.site_qualities.apply(literal_eval)
df['success_01'] = df.success.apply(lambda x: int(x))
pdb.set_trace()

df2 = df.groupby(by=['site_qualities'], as_index=False).agg(lambda x: x.tolist())
# plt.plot()
df['meanSuccess'] = df.success_01.apply(lambda x: np.mean(x))
df['qual_diff'] = df.site_qualities.apply(lambda x: np.max(x) - np.sort(x)[-2])
df['meanTime'] = df.time_converged.apply(lambda x: np.mean(x))
# df['num_a']

plt.figure(1)
for i in df.num_agents:
    print(i)
    plt.plot(df.qual_diff[i], df.meanSuccess[i], str(i)+'.')
    plt.title('mean success vs qual diff')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

plt.figure(2)
for i in df.num_agents:
    plt.plot(df.qual_diff[i], df.meanTime[i], str(i)+'.')
    plt.title('mean time vs qual diff')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-10, 10000])


plt.show()