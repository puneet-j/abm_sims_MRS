import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb
from ast import literal_eval
import time 
import sys
import matplotlib.patches as pat
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))
from params import *
# import pylab as pl

AGENT_COLORS = {'OBSERVE':'b', 'EXPLORE': 'r', 'ASSESS': 'k', 'RECRUIT': 'm', 
                'TRAVEL_HOME_TO_OBSERVE': 'c', 'TRAVEL_HOME_TO_RECRUIT': 'y', 'TRAVEL_SITE': 'y'}


folder = './replay/'
files = os.listdir(folder)
files = [file for file in files if file.endswith('.csv')]
files = np.sort(files)

df = pd.read_csv(folder+files[0])
metadata = pd.read_csv(folder+files[1])

df.agent_positions = df.agent_positions.apply(literal_eval)
df.agent_states = df.agent_states.apply(literal_eval)

metadata.site_positions = metadata.site_positions.apply(literal_eval)
metadata.site_qualities = metadata.site_qualities.apply(literal_eval)

fig = plt.figure()
ax = fig.add_subplot()

colorlist = [i for i in AGENT_COLORS.values()]
statelist = [i for i in AGENT_COLORS.keys()]

plot_legend_x = [350]*len(AGENT_COLORS)
plot_legend = [400, 370, 430, 340, 470, 310, 280]
plot_legend_colors = [330]*len(AGENT_COLORS)

for t, (poses, states) in  enumerate(zip(df.agent_positions, df.agent_states)):
    x_coords = [item[0] for item in poses]
    y_coords = [item[1] for item in poses] 
    colors = [AGENT_COLORS[a_s] for a_s in states]
    RECRUITrs = np.sum([1 for x in states if x=='RECRUIT'])
    # pdb.set_trace()
    for site, qual in zip(metadata.site_positions[0], metadata.site_qualities[0]):
        # pdb.set_trace()
        c = plt.Circle(site, radius = SITE_SIZE, edgecolor=[1.0,qual,0], fill = False)
        ax.add_patch(c)
    chub = plt.Circle((0,0), radius = SITE_SIZE, fill=False)
    ax.add_patch(chub)

    plt.scatter(x_coords, y_coords, c=colors)
    
    plt.scatter(plot_legend_colors, plot_legend, c=colorlist, linewidth=3.0)
    [plt.text(x, y, text) for (x, y, text) in zip(plot_legend_x, plot_legend, statelist)]
    plt.text(-10,450,'TIME: '+str(t))
    plt.text(-20,400,'RECRUITRS: '+str(RECRUITrs))


    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.pause(0.001)
    plt.cla()
