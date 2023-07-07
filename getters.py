import numpy as np
from helper_functions import *
from params import *
import copy 

def get_REST_TO_EXPLORE_PROB(agent, p, toTravelSite):
    if agent_at_hub(agent):
        return p*(1.0 - toTravelSite)
    else:
        return 0.0


def get_REST_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        dancers = get_dancers_by_site(agent)
        if p != 0:
            prob = 1.0*p/np.sum(dancers)
            probs_to_choose = copy.deepcopy([d/np.sum(dancers) for d in dancers])
            togo = agent.world.sites[np.random.choice(range(0,len(dancers)), p = probs_to_choose)]
            # break
        else:
            prob = 0.0
            togo = None
        
        return prob, togo #return prob*p, togo
    else:
        return 0.0, None
    
# def get_REST_TO_TRAVEL_SITE_PROB(agent, p):
#     if agent_at_hub(agent):
#         dancers = get_dancers_by_site(agent)
#         if p != 0:
            
#         if np.sum(dancers) != 0:
#             for id, d in enumerate(dancers): 
#                 if d != 0:
#                     prob = np.random.binomial(1, BINOMIAL_COEFF_REST_TO_ASSESS_PER_DANCER)
#                     # pdb.set_trace()
#                     if prob > 0:
#                         togo = agent.world.sites[id]
#                         break
#                     else:
#                         togo = None
#         else:
#             prob = 0.0
#             togo = None
        
#         return prob, togo #return prob*p, togo
#     else:
#         return 0.0, None
# # changed to not include binomial p in calculations. 
# # check dancers to get recruited
# def get_REST_TO_TRAVEL_SITE_PROB(agent, p):
#     # BINOMIAL_COEFF_EXPLORE_TO_REST
#     if agent_at_hub(agent):
#         dancers = get_dancers_by_site(agent)
#         # pdb.set_trace()
#         if np.sum(dancers) != 0:
#             for d in dancers: 
#             d_morphed = [d**1.0 for d in dancers]
#             dancers_norm = copy.deepcopy(d_morphed/np.sum(d_morphed))
#             togo = np.random.choice(agent.world.sites, p=dancers_norm)
#             # pdb.set_trace()
#             prob = copy.deepcopy(1 - p)
#         else:
#             prob = 0.0
#             togo = None
        
#         return prob, togo #return prob*p, togo
#     else:
#         return 0.0, None

# site quality assessment while exploring uses linear site quality
def get_EXPLORE_TO_ASSESS_PROB(ag):
    site = agent_at_any_site(ag)
    if not(site is None):
        return site.quality, site
        # # try:
        # if site.quality > SITE_QUALITY_ASSESS_THRESHOLD:
        #     return 1.0, site
        # else:
        #     return 0.0, None

        # return site.quality**2.0, site
        # except:
        #     pdb.set_trace()
    else:
        return 0.0, None

def get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, p, toAssess):
    if not(agent_at_hub(agent)): #(agent_at_any_site(agent) is None) and 
        return p*(1.0 - toAssess)
    else:
        return 0.0

# assessment for now is just a binomial random variable
def get_ASSESS_TO_TRAVEL_HOME_PROB(agent, p):
    if not(agent_at_any_site(agent) is None):
        return p
    else:
        pdb.set_trace()

# site quality assessment while thinking about re-assessing uses power law site quality
def get_DANCE_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        ''' squash function of q through sigmoid and then compute the expected number of cycles for max and min q'''
        stay_attach_prob_due_to_qual = 1.0 - np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS * (agent.assigned_site.quality + 0.2))
        # print(stay_attach_prob_due_to_qual) np.exp(agent.assigned_site.quality)/np.exp(1.0)  #
        # return stay_attach_prob_due_to_qual 
        return (1.0 - p)*stay_attach_prob_due_to_qual
    else:
        pdb.set_trace()
        
def get_DANCE_TO_REST_PROB(agent, p):
    if agent_at_hub(agent):
        stay_attach_prob_due_to_qual = 1.0 - np.power(base_DANCE_TO_ASSESS, pwr_DANCE_TO_ASSESS *( agent.assigned_site.quality + 0.2))
        return 1.0 - stay_attach_prob_due_to_qual
        # return p*(1.0 - stay_attach_prob_due_to_qual) np.exp(agent.assigned_site.quality)/np.exp(1.0)   #
    else:
        pdb.set_trace()

# functions below just check if the agent has reached the hub
def get_TRAVEL_SITE_TO_ASSESS_PROB(agent):
    site = agent_at_any_site(agent)
    # pdb.set_trace()
    if site is None:
        return 0.0
    elif agent.assigned_site.id == site.id:
        return 1.0
    else:
        return 0.0


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