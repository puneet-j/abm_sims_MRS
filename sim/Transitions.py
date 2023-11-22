import numpy as np
from params import *
from helper_functions import *
from getters import *
import copy 
import pandas as pd 

class State_Transitions:
    def __init__(self):
        self.new_site = None
    
    def get_transition(self, agent):
        # transition_probabilities = {}

        if agent.state == 'OBSERVE':
            transition_probabilities = self.transition_probabilities_OBSERVE(agent)
        
        elif agent.state == 'EXPLORE':
            transition_probabilities = self.transition_probabilities_explore(agent)
        
        elif agent.state == 'ASSESS':
            transition_probabilities = self.transition_probabilities_assess(agent)
        
        elif agent.state == 'RECRUIT':
            transition_probabilities = self.transition_probabilities_RECRUIT(agent)
        
        elif agent.state == 'TRAVEL_HOME_TO_RECRUIT':
            transition_probabilities = self.transition_probabilities_travel_home_to_RECRUIT(agent)
        
        elif agent.state == 'TRAVEL_HOME_TO_OBSERVE':
            transition_probabilities = self.transition_probabilities_travel_home_to_OBSERVE(agent)
        
        elif agent.state == 'TRAVEL_SITE':
            transition_probabilities = self.transition_probabilities_travel_site(agent)

        else:
            pdb.set_trace()

        try:
            new_state = np.random.choice(list(transition_probabilities.keys()), p=list(transition_probabilities.values()))
        except:
            pdb.set_trace()

        if new_state == agent.state:
            self.new_site = agent.assigned_site

        if new_state == 'EXPLORE' or new_state == 'TRAVEL_HOME_TO_OBSERVE' or new_state == 'OBSERVE':
            self.new_site = None

        if not(self.new_site is None):
            site_to_attach = copy.deepcopy(self.new_site)
            self.new_site = None
            return new_state, site_to_attach
        else:
            return new_state, None


    def transition_probabilities_OBSERVE(self, agent):
        transition_probabilities = {}
        RECRUITrs = get_RECRUITrs_by_site(agent)
        OBSERVE_to_assess_or_not =  np.random.binomial(np.sum(RECRUITrs), BINOMIAL_COEFF_OBSERVE_TO_ASSESS_PER_RECRUITR)
        # pdb.set_trace()
        # OBSERVE_to_assess_or_not = (1 - BINOMIAL_COEFF_OBSERVE_TO_ASSESS_PER_RECRUITR)**np.sum(RECRUITrs) #np.random.binomial(np.sum(RECRUITrs), 1.0 - BINOMIAL_COEFF_OBSERVE_TO_ASSESS_PER_RECRUITR)/np.sum(RECRUITrs)
        OBSERVE_to_explore_or_not = np.random.binomial(1, BINOMIAL_COEFF_OBSERVE_TO_EXPLORE)
        transition_probabilities['TRAVEL_SITE'], self.new_site = get_OBSERVE_TO_TRAVEL_SITE_PROB(agent, OBSERVE_to_assess_or_not) # z
        transition_probabilities['EXPLORE'] = get_OBSERVE_TO_EXPLORE_PROB(agent, OBSERVE_to_explore_or_not, 
                                                            transition_probabilities['TRAVEL_SITE']) # p_1(1-z)
        # pdb.set_trace()
        transition_probabilities['OBSERVE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_RR(agent) # (1-p_1)(1-z)
        return transition_probabilities
    
    def transition_probabilities_explore(self, agent):
        ''' TODO: For future work, make this so you can get probability of rediscovery from the arc ratio of the site at the hub'''
        transition_probabilities = {}
        explore_to_OBSERVE_or_not = np.random.binomial(1, BINOMIAL_COEFF_EXPLORE_TO_OBSERVE)
        # try:
        transition_probabilities['ASSESS'], self.new_site = get_EXPLORE_TO_ASSESS_PROB(agent) # y
        # except:
        #     pdb.set_trace()
        transition_probabilities['TRAVEL_HOME_TO_OBSERVE'] = get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, explore_to_OBSERVE_or_not, 
                                                                        transition_probabilities['ASSESS']) # p_3(1-y)
        transition_probabilities['EXPLORE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_EE(agent) # (1-p_3)(1-y)
        # if transition_probabilities['ASSESS'] > 0.0:
            # pdb.set_trace()
        return transition_probabilities

    def transition_probabilities_assess(self, agent):
        transition_probabilities = {}
        assess_to_RECRUIT_or_not = np.random.binomial(1, BINOMIAL_COEFF_ASSESS_TO_RECRUIT)
        transition_probabilities['TRAVEL_HOME_TO_RECRUIT'], self.new_site = get_ASSESS_TO_TRAVEL_HOME_PROB(agent, assess_to_RECRUIT_or_not) # p_4
        transition_probabilities['ASSESS'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_AA(agent) # 1-p_4
        return transition_probabilities

    def transition_probabilities_RECRUIT(self, agent):
        transition_probabilities = {}
        RECRUIT_to_OBSERVE_or_not = np.random.binomial(1, BINOMIAL_COEFF_RECRUIT_TO_RECRUIT)
        transition_probabilities['TRAVEL_SITE'], self.new_site = get_RECRUIT_TO_TRAVEL_SITE_PROB(agent, RECRUIT_to_OBSERVE_or_not) # gamma*(1 - p_2x)
        transition_probabilities['OBSERVE'] = get_RECRUIT_TO_OBSERVE_PROB(agent, RECRUIT_to_OBSERVE_or_not) # (1 - gamma)(1-p_2x)
        transition_probabilities['RECRUIT'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_DD(agent) # p_2x
        return transition_probabilities
    
    def transition_probabilities_travel_home_to_RECRUIT(self, agent):
        transition_probabilities = {}
        transition_probabilities['RECRUIT'], self.new_site  = get_TRAVEL_HOME_TO_RECRUIT_PROB(agent) # 1 if at hub
        transition_probabilities['TRAVEL_HOME_TO_RECRUIT'] = 1.0 - transition_probabilities['RECRUIT'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_home_to_OBSERVE(self, agent):
        transition_probabilities = {}
        transition_probabilities['OBSERVE'] = get_TRAVEL_HOME_TO_OBSERVE_PROB(agent) # 1 if at hub 
        transition_probabilities['TRAVEL_HOME_TO_OBSERVE'] = 1.0 - transition_probabilities['OBSERVE'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_site(self, agent):
        transition_probabilities = {}
        transition_probabilities['ASSESS'], self.new_site  = get_TRAVEL_SITE_TO_ASSESS_PROB(agent) # 1 if at assigned site
        transition_probabilities['TRAVEL_SITE'] = 1.0 - transition_probabilities['ASSESS'] # 1 if not at site
        return transition_probabilities
    
    '''old ignore'''    
    # def transition_probabilities_RECRUIT(self, agent):
    #     transition_probabilities = {}
    #     RECRUIT_to_OBSERVE_or_not = np.random.binomial(1, BINOMIAL_COEFF_RECRUIT_TO_OBSERVE)
    #     transition_probabilities['TRAVEL_SITE'] = get_RECRUIT_TO_TRAVEL_SITE_PROB(agent, RECRUIT_to_OBSERVE_or_not) # p_2(x)
    #     transition_probabilities['OBSERVE'] = get_RECRUIT_TO_OBSERVE_PROB(agent, RECRUIT_to_OBSERVE_or_not) # p_2(1-x)
    #     transition_probabilities['RECRUIT'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_DD(agent) # 1-p_2
    #     # for val in transition_probabilities.values():
    #     #     if val < 0.0:
    #     #         pdb.set_trace()
    #     # print(transition_probabilities)
    #     return transition_probabilities
    