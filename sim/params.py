# DESCRIBES THE PARAMETERS FOR THE SIMULATION
import numpy as np
import pdb

'''Environment'''

HUB_LOCATION = (0, 0)

SITE_SIZE = 20

ENVIRONMENT_BOUNDARY_X = (-1000, -1000, 1000, 1000)
ENVIRONMENT_BOUNDARY_Y = (-1000, 1000, 1000, -1000)

TIME_LIMIT = 35000

COMMIT_THRESHOLD = 0.5

'''Agent'''

AGENT_SPEED = 5.0
AGENT_AGENT_SENSING_DISTANCE = 20.0
EXPLORE_RANDOMNESS = 0.1
EXPLORE_DIRECTION_CHANGE = np.pi/30
SITE_QUALITY_ASSESS_THRESHOLD = 0.25

BINOMIAL_COEFF_OBSERVE_TO_ASSESS_PER_RECRUITR = 0.05 # OBSERVE to Travel_Site for Assess
BINOMIAL_COEFF_ASSESS_TO_RECRUIT = 0.1 # Assess to Travel_Hub for RECRUIT
BINOMIAL_COEFF_EXPLORE_TO_OBSERVE = 0.02 # Explore to Travel_Hub for OBSERVE
BINOMIAL_COEFF_RECRUIT_TO_OBSERVE = 0.98 # RECRUIT to OBSERVE (1-x)
BINOMIAL_COEFF_OBSERVE_TO_EXPLORE = 0.01 # OBSERVE to Explore

# switch from RECRUIT probs
# base_RECRUIT_TO_ASSESS = 10.0
# pwr_RECRUIT_TO_ASSESS = -2.0
pwr_RECRUIT_TO_TS = 1.0
ADD_RECRUIT_QUAL = 0.5
MULTIPLIER_RECRUIT_QUAL = -7.0

# only for testing 2 site systems
qualities2 = [[0.01, 1.0], [0.05, 0.7], [0.1, 1.0], [0.5, 0.6], [0.3, 0.75], [0.2, 1.0], [0.2, 0.75], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0], 
              [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0], 
              [0.99, 1.0], [0.1, 0.75],   [0.4, 0.75], [0.5, 0.75], [0.6, 0.75], [0.7, 0.75], [0.8, 0.75], 
              [0.9, 0.75], [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6],  [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6]]

# qualities2 = [[0.99, 1.0], [0.1, 0.75], [0.2, 0.75], [0.3, 0.75], [0.4, 0.75], [0.5, 0.75], [0.6, 0.75], [0.7, 0.75], [0.8, 0.75], 
#               [0.9, 0.75], [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6], [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6]]