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

df = pd.read_csv('combined_results_metadata.csv')

df.site_qualities = df.site_qualities.apply(literal_eval)
df.max_qual = df.site_qualities.apply(lambda x: np.max(x))
df['success'] = df.max_qual == df.site_converged

pdb.set_trace()