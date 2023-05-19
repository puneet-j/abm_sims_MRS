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

df = pd.read_csv('./sim_results/combined_results_metadata.csv')

df.site_qualities = df.site_qualities.apply(literal_eval)
df['success_01'] = df.success.apply(lambda x: int(x))

df = df.groupby(by=['site_qualities'], as_index=False).agg(lambda x: x.tolist())
# plt.plot()
df['meanSuccess'] = df.success_01.apply(lambda x: np.mean(x))
df['qual_diff'] = df.site_qualities.apply(lambda x: np.max(x) - np.sort(x)[-2])
df['meanTime'] = df.time_converged.apply(lambda x: np.mean(x))

plt.figure(1)
plt.plot(df.qual_diff, df.meanSuccess, 'r.')
plt.title('mean success vs qual diff')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])

plt.figure(2)
plt.plot(df.qual_diff, df.meanTime, 'r.')
plt.title('mean time vs qual diff')
plt.xlim([-0.1, 1.1])
plt.ylim([-10, 10000])

plt.show()