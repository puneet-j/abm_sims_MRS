# DESCRIBES THE PARAMETERS FOR THE SIMULATION
import numpy as np
import pdb

'''Environment'''

HUB_LOCATION = (0, 0)

SITE_SIZE = 20

ENVIRONMENT_BOUNDARY_X = (-1000, -1000, 1000, 1000)
ENVIRONMENT_BOUNDARY_Y = (-1000, 1000, 1000, -1000)

TIME_LIMIT = 35000

COMMIT_THRESHOLD = 0.3

'''Agent'''

AGENT_SPEED = 5.0
AGENT_AGENT_SENSING_DISTANCE = 20.0
EXPLORE_RANDOMNESS = 0.1
EXPLORE_DIRECTION_CHANGE = np.pi/30
SITE_QUALITY_ASSESS_THRESHOLD = 0.25

BINOMIAL_COEFF_REST_TO_ASSESS_PER_DANCER = 0.05 # Rest to Travel_Site for Assess
BINOMIAL_COEFF_ASSESS_TO_DANCE = 0.05 # Assess to Travel_Hub for Dance
BINOMIAL_COEFF_EXPLORE_TO_REST = 0.02 # Explore to Travel_Hub for Rest
BINOMIAL_COEFF_DANCE_TO_REST = 0.95 # Dance to Rest
BINOMIAL_COEFF_REST_TO_EXPLORE = 0.01 # Rest to Explore

# switch from Dance probs
base_DANCE_TO_ASSESS = 10.0
pwr_DANCE_TO_ASSESS = -2.0
pwr_DANCE_TO_TS = 5.0
ADD_DANCE_QUAL = 0.5
MULTIPLIER_DANCE_QUAL = -5.0

# only for testing 2 site systems
qualities2 = [[0.01, 1.0], [0.05, 0.7], [0.1, 1.0], [0.5, 0.6], [0.3, 0.75], [0.2, 1.0], [0.2, 0.75], [0.3, 1.0], [0.4, 1.0], [0.5, 1.0], 
              [0.6, 1.0], [0.7, 1.0], [0.8, 1.0], [0.9, 1.0], 
              [0.99, 1.0], [0.1, 0.75],   [0.4, 0.75], [0.5, 0.75], [0.6, 0.75], [0.7, 0.75], [0.8, 0.75], 
              [0.9, 0.75], [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6],  [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6]]

# qualities2 = [[0.99, 1.0], [0.1, 0.75], [0.2, 0.75], [0.3, 0.75], [0.4, 0.75], [0.5, 0.75], [0.6, 0.75], [0.7, 0.75], [0.8, 0.75], 
#               [0.9, 0.75], [0.1, 0.6], [0.2, 0.6], [0.3, 0.6], [0.4, 0.6], [0.5, 0.6], [0.6, 0.6], [0.7, 0.6], [0.8, 0.6], [0.9, 0.6]]