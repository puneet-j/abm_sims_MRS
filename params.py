# DESCRIBES THE PARAMETERS FOR THE SIMULATION
import numpy as np

'''Environment'''

HUB_LOCATION = (0, 0)

SITE_SIZE = 10

ENVIRONMENT_BOUNDARY_X = (-1000, -1000, 1000, 1000)
ENVIRONMENT_BOUNDARY_Y = (-1000, 1000, 1000, -1000)

'''Agent'''

AGENT_SPEED = 5.0
AGENT_AGENT_SENSING_DISTANCE = 20.0
EXPLORE_RANDOMNESS = 0.1
EXPLORE_DIRECTION_CHANGE = np.pi/180

PROB_RA = 0.05 # Rest to Travel_Site for Assess
PROB_AD = 0.05 # Assess to Travel_Hub for Dance
PROB_ER = 0.005 # Explore to Travel_Hub for Rest
PROB_DR = 0.05 # Dance to Rest

# for the detachment
base_DA = 5.0
pwr_DA = -1.0

# don't need this since other params do the trick.
# PROB_DA = 0.05 # Dance to Travel_Site for Assess 
# PROB_RE = 0.02 # Rest to Explore
# PROB_EA = 0.05 # Explore to Assess
