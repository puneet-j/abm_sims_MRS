import numpy as np
from helper_functions import *
from params import *
import copy 

def get_REST_TO_EXPLORE_PROB(agent, p, toTravelSite):
    if agent_at_hub(agent):
        return p*(1.0 - toTravelSite)
    else:
        return 0.0

# changed to not include binomial p in calculations. 
# check dancers to get recruited
def get_REST_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        dancers = get_dancers_by_site(agent)
        # pdb.set_trace()
        if np.sum(dancers) != 0:
            dancers = dancers/np.sum(dancers)
            togo = np.random.choice(agent.world.sites, p=dancers)
            prob = copy.deepcopy(dancers[togo.ID])
        else:
            prob = 0
            togo = None
        
        return prob, togo #return prob*p, togo
    else:
        return 0.0, None

# site quality assessment while exploring uses linear site quality
def get_EXPLORE_TO_ASSESS_PROB(ag):
    site = agent_at_any_site(ag)
    if site:
        return site.quality, site
    else:
        return 0.0, None

def get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, p, toAssess):
    if not (agent_at_any_site(agent)) and  not(agent_at_hub(agent)):
        return p*(1.0 - toAssess)
    else:
        return 0

# assessment for now is just a binomial random variable
def get_ASSESS_TO_TRAVEL_HOME_PROB(agent, p):
    if agent_at_any_site(agent):
        return p
    else:
        pdb.set_trace()

# site quality assessment while thinking about re-assessing uses power law site quality
def get_DANCE_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        detach_prob_due_to_qual = 1.0 - np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS * agent.assigned_site.quality)
        return p*detach_prob_due_to_qual
    else:
        pdb.set_trace()
        
def get_DANCE_TO_REST_PROB(agent, p):
    if agent_at_hub(agent):
        detach_prob_due_to_qual = 1.0 - np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS * agent.assigned_site.quality)
        return p*(1.0 - detach_prob_due_to_qual)
    else:
        pdb.set_trace()

# functions below just check if the agent has reached the hub
def get_TRAVEL_SITE_TO_ASSESS_PROB(agent):
    site = agent_at_any_site(agent)
    if not site:
        return 0.0
    elif agent.assigned_site.id == site.id:
        return 1.0


def get_TRAVEL_HOME_TO_DANCE_PROB(agent):
    if agent_at_hub(agent):
        return 1.0
    else:
        return 0.0
    
def get_TRAVEL_HOME_TO_REST_PROB(agent):
    if agent_at_hub(agent):
        return 1.0
    else:
        return 0.0