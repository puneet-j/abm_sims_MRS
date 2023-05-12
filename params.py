# DESCRIBES THE PARAMETERS FOR THE SIMULATION
import numpy as np
import pdb

'''Environment'''

HUB_LOCATION = (0, 0)

SITE_SIZE = 10

ENVIRONMENT_BOUNDARY_X = (-1000, -1000, 1000, 1000)
ENVIRONMENT_BOUNDARY_Y = (-1000, 1000, 1000, -1000)

TIME_LIMIT = 35000

'''Agent'''

AGENT_SPEED = 5.0
AGENT_AGENT_SENSING_DISTANCE = 20.0
EXPLORE_RANDOMNESS = 0.1
EXPLORE_DIRECTION_CHANGE = np.pi/180

BINOMIAL_COEFF_REST_TO_ASSESS = 0.05 # Rest to Travel_Site for Assess
BINOMIAL_COEFF_ASSESS_TO_DANCE = 0.05 # Assess to Travel_Hub for Dance
BINOMIAL_COEFF_EXPLORE_TO_REST = 0.005 # Explore to Travel_Hub for Rest
BINOMIAL_COEFF_DANCE_TO_REST = 0.05 # Dance to Rest
BINOMIAL_COEFF_REST_TO_EXPLORE = 0.02 # Rest to Explore

# for the detachment
base_DANCE_TO_ASSESS = 5.0
pwr_DANCE_TO_ASSESS = -1.0

# don't need this since other params do the trick.
# PROB_DA = 0.05 # Dance to Travel_Site for Assess 
# PROB_EA = 0.05 # Explore to Assess
