import numpy as np
from helper_functions import *
from params import *

def get_REST_TO_EXPLORE_PROB(agent, p, toTravelSite):
    return p*(1.0 - toTravelSite)

# check dancers to get recruited
def get_REST_TO_TRAVEL_SITE_PROB(agent, p):
    dancers = get_dancers_by_site(agent)
    # pdb.set_trace()
    if np.sum(dancers) != 0:
        dancers = dancers/np.sum(dancers)
        togo = np.random.choice(agent.world.sites, p=dancers)
    else:
        p = 0
        togo = None
    return p, togo

# site quality assessment while exploring uses linear site quality
def get_EXPLORE_TO_ASSESS_PROB(ag):
    site = get_site_from_id(ag)
    return site.quality, site

def get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, p, toAssess):
    return p*(1.0 - toAssess)

# assessment for now is just a binomial random variable
def get_ASSESS_TO_TRAVEL_HOME_PROB(agent, p):
    return p

# site quality assessment while thinking about re-assessing uses power law site quality
def get_DANCE_TO_TRAVEL_SITE_PROB(agent, p):
    return p*np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS * agent.assigned_site.quality) 
        
def get_DANCE_TO_REST_PROB(agent, p):
    return p*(1.0 - np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS * agent.assigned_site.quality))

# functions below just check if the agent has reached the hub
def get_TRAVEL_SITE_TO_ASSESS_PROB(agent):
    if agent.assigned_site.id == agent.at_site:
        return 1.0
    else:
        return 0.0

def get_TRAVEL_HOME_TO_DANCE_PROB(agent):
    if agent.at_hub:
        return 1.0
    else:
        return 0.0
    
def get_TRAVEL_HOME_TO_REST_PROB(agent):
    if agent.at_hub:
        return 1.0
    else:
        return 0.0