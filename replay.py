import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pdb
from ast import literal_eval

AGENT_COLORS = {'REST':'b',}

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

# pdb.set_trace()
x_coords = [item[0] for item in df.agent_positions[0]]
y_coords = [item[1] for item in df.agent_positions[0]]

hl, = plt.plot(x_coords, y_coords,'.')
plt.show()
# pdb.set_trace()

for poses in df.agent_poses:
    x_coords = [item[0] for item in poses]
    y_coords = [item[1] for item in poses]    
    update_line(hl, poses)
