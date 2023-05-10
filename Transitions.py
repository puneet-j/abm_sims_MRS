import numpy as np
from params import *
from helper_functions import *

class State_Transitions(self):
    def __init__(self, world):
        self.world = world
        self.new_site = None
    
    def get_transition(self, agent):
        transition_probabilities = {}
        if agent.state == 'REST':
            p_1 = np.random.binomial(PROB_RE,1)
            p_5 = np.random.binomial(PROB_RA,1)
            transition_probabilities['TRAVEL_SITE'], self.new_site = self.get_RTS(agent, p_5) # z
            transition_probabilities['EXPLORE'] = self.get_RE(agent, p_1, transition_probabilities['TRAVEL_SITE']) # p_1(1-z)
            transition_probabilities['REST'] = 1.0 - np.sum(transition_probabilities.values()) # self.get_RR(agent) # (1-p_1)(1-z)
        
        elif agent.state == 'EXPLORE':
            p_3 = np.random.binomial(PROB_ER,1)
            transition_probabilities['ASSESS'], self.new_site = self.get_EA(agent) # y
            transition_probabilities['TRAVEL_HOME_EXPLORE'] = self.get_ETH(agent, p_3, transition_probabilities['ASSESS']) # p_3(1-y)
            transition_probabilities['EXPLORE'] = 1.0 - np.sum(transition_probabilities.values()) # self.get_EE(agent) # (1-p_3)(1-y)
        
        elif agent.state == 'ASSESS':
            p_4 = np.random.binomial(PROB_AD,1)
            transition_probabilities['TRAVEL_HOME_ASSESS'] = self.get_ATH(agent, p_4) # p_4
            transition_probabilities['ASSESS'] = 1.0 - np.sum(transition_probabilities.values()) # self.get_AA(agent) # 1-p_4
        
        elif agent.state == 'DANCE':
            p_2 = np.random.binomial(PROB_DR,1)
            transition_probabilities['TRAVEL_SITE'] = self.get_DTS(agent, p_2) # p_2(x)
            transition_probabilities['REST'] = self.get_DR(agent, p_2) # p_2(1-x)
            transition_probabilities['DANCE'] = 1.0 - np.sum(transition_probabilities.values()) # self.get_DD(agent) # 1-p_2
        
        elif agent.state == 'TRAVEL_HOME_ASSESS':
            transition_probabilities['DANCE'] = self.get_THA(agent) # 1 if at hub
            transition_probabilities['TRAVEL_HOME_ASSESS'] = 1.0 - transition_probabilities['DANCE'] # 1 if not at hub
        
        elif agent.state == 'TRAVEL_HOME_EXPLORE':
            transition_probabilities['REST'] = self.get_THE(agent) # 1 if at hub 
            transition_probabilities['TRAVEL_HOME_ASSESS'] = 1.0 - transition_probabilities['REST'] # 1 if not at hub
        
        elif agent.state == 'TRAVEL_SITE':
            transition_probabilities['ASSESS'] = self.get_TSA(agent) # 1 if at assigned site
            transition_probabilities['TRAVEL_SITE'] = 1.0 - transition_probabilities['ASSESS'] # 1 if not at site


        new_state = np.random.choice(list(transition_probabilities.keys()), p=list(transition_probabilities.values()))
        
        if self.new_site:
            site_to_attach = self.new_site
            self.new_site = None
            return new_state, site_to_attach
        else:
            return new_state, None


    def get_RE(self, agent, p_1, tTS):
        return p_1*(1.0 - tTS)
    
    # check dancers to get recruited
    def get_RTS(self, agent, p_5):
        dancers = get_dancers_by_site(agent)
        togo = np.random.choice(agent.world.sites, p=dancers)
        return p_5, togo

    # site quality assessment while exploring uses linear site quality
    def get_EA(self, agent):
        site = get_site_from_id(ag)
        return site.quality, site

    def get_ETH(self, agent, p_3, tA):
        return p_3*(1.0 - tA)

    # assessment for now is just a binomial random variable
    def get_ATH(self, agent, p_4):
        return p_4
    
    # site quality assessment while thinking about re-assessing uses power law site quality
    def get_DTS(self, agent, p_2):
        return p_2*np.power(base_DA, pwr_DA * agent.assigned_site.quality) 
            
    def get_DR(self, agent, p_2):
        return p_2*(1.0 - np.power(base_DA, pwr_DA * agent.assigned_site.quality))

    # functions below just check if the agent has reached the hub
    def get_TSA(self, agent):
        if agent.assigned_site.id == agent.at_site:
            return 1.0
        else:
            return 0.0
    
    def get_THA(self, agent):
        if agent.at_hub:
            return 1.0
        else:
            return 0.0
        
    def get_THE(self, agent):
        if agent.at_hub:
            return 1.0
        else:
            return 0.0