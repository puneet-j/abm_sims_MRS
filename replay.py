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

def update_line(hl, new_data):
    hl.set_xdata(new_data)
    hl.set_ydata(new_data)
    # ax.autoscale_view()
    plt.draw()

folder = './sim_results/'
files = os.listdir(folder)
files = [file for file in files if file.endswith('.csv')]
files = np.sort(files)

df = pd.read_csv(folder+files[0])
metadata = pd.read_csv(folder+files[1])

df.agent_positions = df.agent_positions.apply(literal_eval)
df.agent_states = df.agent_states.apply(literal_eval)

metadata.site_positions = metadata.site_positions.apply(literal_eval)

# pdb.set_trace()
# x_coords = [item[0] for item in df.agent_positions[0]]
# y_coords = [item[1] for item in df.agent_positions[0]]

# hl, = plt.plot(x_coords, y_coords,'.')
# plt.show()
# pdb.set_trace()
# fig = plt.figure(1)
# plt.hold()
fig = plt.figure()
ax = fig.add_subplot()

colorlist = [i for i in AGENT_COLORS.values()]
statelist = [i for i in AGENT_COLORS.keys()]
plot_legend_x = [350]*len(AGENT_COLORS)
plot_legend = [400, 370, 430, 340, 470, 310, 280]
plot_legend_colors = [330]*len(AGENT_COLORS)

for poses, states in  zip(df.agent_positions, df.agent_states):
    x_coords = [item[0] for item in poses]
    y_coords = [item[1] for item in poses] 
    colors = [AGENT_COLORS[a_s] for a_s in states]

    # update_line(hl, poses)
    # plt.plot()
    for site in metadata.site_positions:
        c = plt.Circle(site[0], radius = SITE_SIZE, fill=False)
        ax.add_patch(c)
    chub = plt.Circle((0,0), radius = SITE_SIZE, fill=False)
    ax.add_patch(chub)
    # plt.plot(0,0,'co',linewidth=5.0)
    # pdb.set_trace()
    plt.scatter(x_coords, y_coords, c=colors)
    
    plt.scatter(plot_legend_colors, plot_legend, c=colorlist, linewidth=3.0)
    [plt.text(x, y, text) for (x, y, text) in zip(plot_legend_x, plot_legend, statelist)]
    # ax.add_patch(c1)
    # ax.add_patch(c2)

    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.pause(0.01)
    # pdb.set_trace()
    plt.cla()
