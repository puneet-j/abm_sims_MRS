import pandas as pd
import numpy as np
import pdb
from ast import literal_eval
# import matplotlib.plt as plot

def get_convTimes(x, distfactor = True):
    # pdb.set_trace()
    if distfactor == True:
        return [f for f, t, h in zip(x.time_converged, x.maxTime, x.hubStart) if ((x.dists[0]==150.0) and (f != t) and (t==35000) and (h==1))]
    else:
        return [f for f, t, h in zip(x.time_converged, x.maxTime, x.hubStart) if ( (f != t) and (t==35000) and (h==1))]
def get_succ(x, distfactor = True):
    # pdb.set_trace()
    if distfactor == True:
        return [f/np.max(x.site_qualities) for f, t, h in zip(x.site_converged, x.maxTime, x.hubStart) if ((not np.isnan(f)) and (x.dists[0]==150.0) and (h==1) and (t==35000))]
    else:
        return [f/np.max(x.site_qualities) for f, t, h in zip(x.site_converged, x.maxTime, x.hubStart) if ((not np.isnan(f)) and (h==1) and (t==35000))]
def get_dist(x):
    # pdb.set_trace()
    return [np.round(abs(f[0][0]),3) for f, t, h in zip(x.site_positions, x.maxTime, x.hubStart) if ( (h==1) and (t==35000))]

def get_all_dist(x):
    # pdb.set_trace()
    return [np.round(abs(f[0][0]),3) for f, t, h in zip(x.site_positions, x.maxTime, x.hubStart) if ((t==35000))]


fl = pd.read_csv('./graphsage_results/CDC/metadata.csv')
fl.site_qualities = fl.site_qualities.apply(literal_eval)
fl.site_positions = fl.site_positions.apply(literal_eval)
fl['hubStart'] = fl.start_state.apply(lambda x: 1 if x == '()' else 0)
fl = fl.groupby('site_qualities', as_index=False).agg(lambda x: x.tolist())[['hubStart', 'site_qualities', 'time_converged', 'site_converged', 'site_positions', 'maxTime']]
# pdb.set_trace()
# print(fl.columns)
# pdb.set_trace()
# newfl = fl.groupby('hubStart', as_index=False).agg(lambda x: x.tolist())
# flHub = newfl.iloc[1]
# flHub = fl.apply(lambda x: x if x.hubStart==1 else None, axis=1)

fl['dists'] = fl.apply(lambda x: get_dist(x), axis=1)
fl['convTimes'] = fl.apply(lambda x: get_convTimes(x, distfactor = False), axis=1)
fl['successes'] = fl.apply(lambda x: get_succ(x, distfactor = False), axis=1)
# fl['alldists'] = fl.apply(lambda x: get_all_dist(x), axis=1)
fl['maxQual'] = fl.site_qualities.apply(lambda x: np.max(x))

# pdb.set_trace()
# times = fl.apply(lambda x: x.time_converged if ((x.hubStart == 1) and (x.time_converged != x.maxTime)), axis=1)
# flHub = []
# pdb.set_trace()

# fl['lenT'] = fl.convTimes.apply(lambda x: len(x))
# fl['lenS'] = fl.successes.apply(lambda x: len(x))

# times = [f for f, t in zip(flHub.time_converged, flHub.maxTime) if f != t]
# times = fl.convTimes.values()

fl = fl[fl.successes.apply(lambda x: len(x) > 0)]

# pdb.set_trace()

fl['meanTimes'] = fl.convTimes.apply(lambda x: np.mean(x))
fl['iqt1'] = fl.convTimes.apply(lambda x: np.percentile(x, 25))
fl['iqt3'] = fl.convTimes.apply(lambda x: np.percentile(x, 75))

fl['meansuccess'] = fl.successes.apply(lambda x: np.mean(x))
fl['iqs1'] = fl.successes.apply(lambda x: np.percentile(x, 25))
fl['iqs3'] = fl.successes.apply(lambda x: np.percentile(x, 75))

fl['qualdiff'] = fl.site_qualities.apply(lambda x: np.abs(x[1] - x[0]))
fl['dists'] = fl.dists.apply(lambda x: x[0])
# pdb.set_trace()
# fl['dist'] = fl.site_positions.apply(lambda x: np.abs(x[1] - x[0]))

fl = fl[['meanTimes', 'iqt1', 'iqt3', 'meansuccess', 'iqs1', 'iqs3', 'qualdiff', 'dists', 'maxQual']]#, 'lenT', 'lenS']]#,'site_positions']]
# pdb.set_trace()
fl = fl.sort_values(by='qualdiff')
# fl = fl[['meanTimes', 'iqt1', 'iqt3', 'meansuccess', 'iqs1', 'iqs3', 'site_qualities','site_positions']]
# pdb.set_trace()


fl.to_csv('ABM_plot_CDC.csv')
# pdb.set_trace()


# iqtime1 = np.percentile(times, 25)
# iqtime3 = np.percentile(times, 75)
# pdb.set_trace()
# successes = [f/np.max(m) for f,m in zip(flHub.site_converged, flHub.site_qualities) if f is not np.nan]
# meanSuccess = np.mean(successes)
# iqsuccess1 = np.percentile(successes, 25)
# iqsuccess3 = np.percentile(successes, 75)
# pdb.set_trace()
