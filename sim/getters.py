import numpy as np
from helper_functions import *
from params import *
import copy 
import pandas as pd 

def get_OBSERVE_TO_EXPLORE_PROB(agent, p, toTravelSite):
    if agent_at_hub(agent):
        return p*(1.0 - toTravelSite)
    else:
        return 0.0


def get_OBSERVE_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        RECRUITrs = get_RECRUITrs_by_site(agent)
        if p != 0:
            prob = 1.0*p/np.sum(RECRUITrs)
            probs_to_choose = copy.deepcopy([d/np.sum(RECRUITrs) for d in RECRUITrs])
            togo = agent.world.sites[np.random.choice(range(0,len(RECRUITrs)), p = probs_to_choose)]
            # break
        else:
            prob = 0.0
            togo = None
        
        return prob, togo #return prob*p, togo
    else:
        return 0.0, None

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
        # pdb.set_trace()
        return 0.0, None

def get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, p, toAssess):
    if not(agent_at_hub(agent)): #(agent_at_any_site(agent) is None) and 
        return p*(1.0 - toAssess)
    else:
        return 0.0

# assessment for now is just a binomial random variable
def get_ASSESS_TO_TRAVEL_HOME_PROB(agent, p):
    if not(agent_at_any_site(agent) is None):
        return p, agent.assigned_site
    else:
        pdb.set_trace()

# site quality assessment while thinking about re-assessing uses power law site quality
def get_RECRUIT_TO_TRAVEL_SITE_PROB(agent, p):
    if agent_at_hub(agent):
        ''' squash function of q through sigmoid and then compute the expected number of cycles for max and min q'''
        # stay_attach_prob_due_to_qual = 1.0 - np.power(base_RECRUIT_TO_ASSESS, pwr_RECRUIT_TO_ASSESS*agent.assigned_site.quality)
        stay_attach_prob_due_to_qual = agent.assigned_site.quality**pwr_RECRUIT_TO_TS
        # print(stay_attach_prob_due_to_qual) np.exp(agent.assigned_site.quality)/np.exp(1.0)  #
        # return stay_attach_prob_due_to_qual 
        RECRUIT_qual_factor = ADD_RECRUIT_QUAL/(ADD_RECRUIT_QUAL + np.exp(MULTIPLIER_RECRUIT_QUAL*agent.assigned_site.quality))
        return stay_attach_prob_due_to_qual*(1.0 - p*RECRUIT_qual_factor), agent.assigned_site #(1.0 - p)*stay_attach_prob_due_to_qual
        # return 
    else:
        pdb.set_trace()
        
def get_RECRUIT_TO_OBSERVE_PROB(agent, p):
    if agent_at_hub(agent):
        # stay_attach_prob_due_to_qual = 1.0 - np.power(base_RECRUIT_TO_ASSESS, pwr_RECRUIT_TO_ASSESS*agent.assigned_site.quality)
        stay_attach_prob_due_to_qual = agent.assigned_site.quality**pwr_RECRUIT_TO_TS
        # RECRUIT_qual_factor = ADD_RECRUIT_QUAL/(ADD_RECRUIT_QUAL + np.exp(MULTIPLIER_RECRUIT_QUAL*agent.assigned_site.quality))
        return (1.0 - stay_attach_prob_due_to_qual)*(1.0 - p*stay_attach_prob_due_to_qual) #1.0 - stay_attach_prob_due_to_qual
        # return p*(1.0 - stay_attach_prob_due_to_qual) np.exp(agent.assigned_site.quality)/np.exp(1.0)   #
    else:
        pdb.set_trace()

# functions below just check if the agent has reached the hub or site
def get_TRAVEL_SITE_TO_ASSESS_PROB(agent):
    site = agent_at_any_site(agent)
    # pdb.set_trace()
    if site is None:
        return 0.0, agent.assigned_site
    elif agent.assigned_site.id == site.id:
        return 1.0, agent.assigned_site
    else:
        return 0.0, agent.assigned_site


def get_TRAVEL_HOME_TO_RECRUIT_PROB(agent):
    # site = agent_at_any_site(agent)
    # site = agent.assigned_site
    if agent_at_hub(agent):
        return 1.0, agent.assigned_site
    else:
        return 0.0, agent.assigned_site
    
def get_TRAVEL_HOME_TO_OBSERVE_PROB(agent):
    if agent_at_hub(agent):
        return 1.0
    else:
        return 0.0