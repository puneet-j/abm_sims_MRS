import numpy as np
from params import *
from helper_functions import *
from getters import *

class State_Transitions:
    def __init__(self, world):
        self.world = world
        self.new_site = None
    
    def get_transition(self, agent):

        if agent.state == 'REST':
            transition_probabilities = self.transition_probabilities_rest(agent)
        
        elif agent.state == 'EXPLORE':
            transition_probabilities = self.transition_probabilities_explore(agent)
        
        elif agent.state == 'ASSESS':
            transition_probabilities = self.transition_probabilities_assess(agent)
        
        elif agent.state == 'DANCE':
            transition_probabilities = self.transition_probabilities_dance(agent)
        
        elif agent.state == 'TRAVEL_HOME_TO_DANCE':
            transition_probabilities = self.transition_probabilities_travel_home_to_dance(agent)
        
        elif agent.state == 'TRAVEL_HOME_TO_REST':
            transition_probabilities = self.transition_probabilities_travel_home_to_rest(agent)
        
        elif agent.state == 'TRAVEL_SITE':
            transition_probabilities = self.transition_probabilities_travel_site(agent)


        new_state = np.random.choice(list(transition_probabilities.keys()), p=list(transition_probabilities.values()))
        
        if self.new_site:
            site_to_attach = self.new_site
            self.new_site = None
            return new_state, site_to_attach
        else:
            return new_state, None


    def transition_probabilities_rest(self, agent):
        transition_probabilities = {}
        rest_to_explore_or_not = np.random.binomial(BINOMIAL_COEFF_REST_TO_EXPLORE, 1)
        rest_to_assess_or_not = np.random.binomial(BINOMIAL_COEFF_REST_TO_ASSESS, 1)
        transition_probabilities['TRAVEL_SITE'], self.new_site = get_REST_TO_TRAVEL_SITE_PROB(agent, rest_to_assess_or_not) # z
        transition_probabilities['EXPLORE'] = get_REST_TO_EXPLORE_PROB(agent, rest_to_explore_or_not, 
                                                            transition_probabilities['TRAVEL_SITE']) # p_1(1-z)
        # pdb.set_trace()
        transition_probabilities['REST'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_RR(agent) # (1-p_1)(1-z)
        return transition_probabilities
    
    def transition_probabilities_explore(self, agent):
        transition_probabilities = {}
        explore_to_rest_or_not = np.random.binomial(BINOMIAL_COEFF_EXPLORE_TO_REST, 1)
        transition_probabilities['ASSESS'], self.new_site = get_EXPLORE_TO_ASSESS_PROB(agent) # y
        transition_probabilities['TRAVEL_HOME_EXPLORE'] = get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, explore_to_rest_or_not, 
                                                                        transition_probabilities['ASSESS']) # p_3(1-y)
        transition_probabilities['EXPLORE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_EE(agent) # (1-p_3)(1-y)
        return transition_probabilities

    def transition_probabilities_assess(self, agent):
        transition_probabilities = {}
        assess_to_dance_or_not = np.random.binomial(BINOMIAL_COEFF_ASSESS_TO_DANCE, 1)
        transition_probabilities['TRAVEL_HOME_ASSESS'] = get_ASSESS_TO_TRAVEL_HOME_PROB(agent, assess_to_dance_or_not) # p_4
        transition_probabilities['ASSESS'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_AA(agent) # 1-p_4
        return transition_probabilities
    
    def transition_probabilities_dance(self, agent):
        transition_probabilities = {}
        dance_to_rest_or_not = np.random.binomial(BINOMIAL_COEFF_DANCE_TO_REST, 1)
        transition_probabilities['TRAVEL_SITE'] = get_DANCE_TO_TRAVEL_SITE_PROB(agent, dance_to_rest_or_not) # p_2(x)
        transition_probabilities['REST'] = get_DANCE_TO_REST_PROB(agent, dance_to_rest_or_not) # p_2(1-x)
        transition_probabilities['DANCE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_DD(agent) # 1-p_2
        return transition_probabilities
    
    def transition_probabilities_travel_home_to_dance(self, agent):
        transition_probabilities = {}
        transition_probabilities['DANCE'] = get_TRAVEL_HOME_TO_DANCE_PROB(agent) # 1 if at hub
        transition_probabilities['TRAVEL_HOME_ASSESS'] = 1.0 - transition_probabilities['DANCE'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_home_to_rest(self, agent):
        transition_probabilities = {}
        transition_probabilities['REST'] = get_TRAVEL_HOME_TO_REST_PROB(agent) # 1 if at hub 
        transition_probabilities['TRAVEL_HOME_ASSESS'] = 1.0 - transition_probabilities['REST'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_site(self, agent):
        transition_probabilities = {}
        transition_probabilities['ASSESS'] = get_TRAVEL_SITE_TO_ASSESS_PROB(agent) # 1 if at assigned site
        transition_probabilities['TRAVEL_SITE'] = 1.0 - transition_probabilities['ASSESS'] # 1 if not at site
        return transition_probabilities