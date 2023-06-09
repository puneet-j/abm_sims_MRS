import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import pdb

STATES = {'REST':0, 'DANCE':1, 'ASSESS':2, 'TRAVEL_SITE':3, 'TRAVEL_HOME_TO_DANCE':4, 'TRAVEL_HOME_TO_REST':4}

def getSiteID(site):
    if site == 'H':
        return 0
    else:
        return (int(site[-1]) + 1)


def get_current_state(astate, asite, nsites):
    current_state = np.zeros((nsites, 5))
    for site, state in zip(asite, astate):
        if state == 'DANCE':
            current_state[0, STATES[state]] += 1
    
        if site:
            current_state[getSiteID(site),STATES[state]] += 1

        elif state == 'REST':
            current_state[getSiteID(site),STATES[state]] += 1
        
        elif state == 'EXPLORE':
            continue
        
        elif state == 'TRAVEL_HOME_TO_REST':
            continue
        
        else:
            pdb.set_trace()
    
    return current_state


