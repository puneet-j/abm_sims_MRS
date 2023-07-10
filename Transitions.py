import numpy as np
from params import *
from helper_functions import *
from getters import *
import copy 

class State_Transitions:
    def __init__(self, world):
        self.world = world
        self.new_site = None
    
    def get_transition(self, agent):
        # transition_probabilities = {}

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

        else:
            pdb.set_trace()

        # pdb.set_trace()
        try:
            new_state = np.random.choice(list(transition_probabilities.keys()), p=list(transition_probabilities.values()))
        except:
            pdb.set_trace()
        # if new_state = 
        if not(self.new_site is None):
            site_to_attach = copy.deepcopy(self.new_site)
            self.new_site = None
            return new_state, site_to_attach
        else:
            return new_state, None


    def transition_probabilities_rest(self, agent):
        transition_probabilities = {}
        dancers = get_dancers_by_site(agent)
        rest_to_assess_or_not =  np.random.binomial(np.sum(dancers), BINOMIAL_COEFF_REST_TO_ASSESS_PER_DANCER)
        # pdb.set_trace()
        # rest_to_assess_or_not = (1 - BINOMIAL_COEFF_REST_TO_ASSESS_PER_DANCER)**np.sum(dancers) #np.random.binomial(np.sum(dancers), 1.0 - BINOMIAL_COEFF_REST_TO_ASSESS_PER_DANCER)/np.sum(dancers)
        rest_to_explore_or_not = np.random.binomial(1, BINOMIAL_COEFF_REST_TO_EXPLORE)
        transition_probabilities['TRAVEL_SITE'], self.new_site = get_REST_TO_TRAVEL_SITE_PROB(agent, rest_to_assess_or_not) # z
        transition_probabilities['EXPLORE'] = get_REST_TO_EXPLORE_PROB(agent, rest_to_explore_or_not, 
                                                            transition_probabilities['TRAVEL_SITE']) # p_1(1-z)
        # pdb.set_trace()
        transition_probabilities['REST'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_RR(agent) # (1-p_1)(1-z)
        return transition_probabilities
    
    def transition_probabilities_explore(self, agent):
        ''' TODO: For future work, make this so you can get probability of rediscovery from the arc ratio of the site at the hub'''
        transition_probabilities = {}
        explore_to_rest_or_not = np.random.binomial(1, BINOMIAL_COEFF_EXPLORE_TO_REST)
        # try:
        transition_probabilities['ASSESS'], self.new_site = get_EXPLORE_TO_ASSESS_PROB(agent) # y
        # except:
        #     pdb.set_trace()
        transition_probabilities['TRAVEL_HOME_TO_REST'] = get_EXPLORE_TO_TRAVEL_HOME_PROB(agent, explore_to_rest_or_not, 
                                                                        transition_probabilities['ASSESS']) # p_3(1-y)
        transition_probabilities['EXPLORE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_EE(agent) # (1-p_3)(1-y)
        return transition_probabilities

    def transition_probabilities_assess(self, agent):
        transition_probabilities = {}
        assess_to_dance_or_not = np.random.binomial(1, BINOMIAL_COEFF_ASSESS_TO_DANCE)
        transition_probabilities['TRAVEL_HOME_TO_DANCE'] = get_ASSESS_TO_TRAVEL_HOME_PROB(agent, assess_to_dance_or_not) # p_4
        transition_probabilities['ASSESS'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_AA(agent) # 1-p_4
        return transition_probabilities

    def transition_probabilities_dance(self, agent):
        transition_probabilities = {}
        dance_to_rest_or_not = np.random.binomial(1, BINOMIAL_COEFF_DANCE_TO_REST)
        transition_probabilities['TRAVEL_SITE'] = get_DANCE_TO_TRAVEL_SITE_PROB(agent, dance_to_rest_or_not) # gamma*(1 - p_2x)
        transition_probabilities['REST'] = get_DANCE_TO_REST_PROB(agent, dance_to_rest_or_not) # (1 - gamma)(1-p_2x)
        transition_probabilities['DANCE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_DD(agent) # p_2x
        return transition_probabilities
    
    def transition_probabilities_travel_home_to_dance(self, agent):
        transition_probabilities = {}
        transition_probabilities['DANCE'] = get_TRAVEL_HOME_TO_DANCE_PROB(agent) # 1 if at hub
        transition_probabilities['TRAVEL_HOME_TO_DANCE'] = 1.0 - transition_probabilities['DANCE'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_home_to_rest(self, agent):
        transition_probabilities = {}
        transition_probabilities['REST'] = get_TRAVEL_HOME_TO_REST_PROB(agent) # 1 if at hub 
        transition_probabilities['TRAVEL_HOME_TO_REST'] = 1.0 - transition_probabilities['REST'] # 1 if not at hub
        return transition_probabilities

    def transition_probabilities_travel_site(self, agent):
        transition_probabilities = {}
        transition_probabilities['ASSESS'] = get_TRAVEL_SITE_TO_ASSESS_PROB(agent) # 1 if at assigned site
        transition_probabilities['TRAVEL_SITE'] = 1.0 - transition_probabilities['ASSESS'] # 1 if not at site
        return transition_probabilities
    
    '''old ignore'''    
    # def transition_probabilities_dance(self, agent):
    #     transition_probabilities = {}
    #     dance_to_rest_or_not = np.random.binomial(1, BINOMIAL_COEFF_DANCE_TO_REST)
    #     transition_probabilities['TRAVEL_SITE'] = get_DANCE_TO_TRAVEL_SITE_PROB(agent, dance_to_rest_or_not) # p_2(x)
    #     transition_probabilities['REST'] = get_DANCE_TO_REST_PROB(agent, dance_to_rest_or_not) # p_2(1-x)
    #     transition_probabilities['DANCE'] = 1.0 - np.sum(list(transition_probabilities.values())) # self.get_DD(agent) # 1-p_2
    #     # for val in transition_probabilities.values():
    #     #     if val < 0.0:
    #     #         pdb.set_trace()
    #     # print(transition_probabilities)
    #     return transition_probabilities
    