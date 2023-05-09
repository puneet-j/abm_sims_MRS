import numpy as np
from params import *

class State_Transitions(self):
    def __init__(self, world):
        self.world = world
    
    def get_transition(self, agent):
        transition_probabilities = {}
        if agent.state == 'REST':
            transition_probabilities['TRAVEL_SITE'] = self.get_RTS(agent) # z
            transition_probabilities['EXPLORE'] = self.get_RE(agent) # p_1(1-z)
            transition_probabilities['REST'] = 1.0 - np.sum(transition_probabilities.keys()) # self.get_RR(agent) # (1-p_1)(1-z)
        elif agent.state == 'EXPLORE':
            transition_probabilities['ASSESS'] = self.get_EA(agent) # y
            transition_probabilities['TRAVEL_HOME_EXPLORE'] = self.get_ETH(agent) # p_3(1-y)
            transition_probabilities['EXPLORE'] = 1.0 - np.sum(transition_probabilities.keys()) # self.get_EE(agent) # (1-p_3)(1-y)
        elif agent.state == 'ASSESS':
            transition_probabilities['TRAVEL_HOME_ASSESS'] = self.get_ATH(agent) # p_4
            transition_probabilities['ASSESS'] = 1.0 - np.sum(transition_probabilities.keys()) # self.get_AA(agent) # 1-p_4
        elif agent.state == 'DANCE':
            transition_probabilities['TRAVEL_SITE'] = self.get_DTS(agent) # p_2(x)
            transition_probabilities['REST'] = self.get_DR(agent) # p_2(1-x)
            transition_probabilities['DANCE'] = 1.0 - np.sum(transition_probabilities.keys()) # self.get_DD(agent) # 1-p_2
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
 
        return new_state


    def get_TSA(self, agent):
        return agent.assigned_site.id == self.at_site
    def get_THA(self, agent):
        return
    def get_THE(self, agent):
        return
    def get_DTS(self, agent):
        return    
    def get_DTR(self, agent):
        return
    def get_DD(self, agent):
        return
    def get_ATH(self, agent):
        return
    def get_AA(self, agent):
        return 
    def get_RE(self, agent):
        return
    def get_RTS(self, agent):
        return   
    def get_RR(self, agent):
        return    
    def get_EA(self, agent):
        site_quality = self.assigned_site.quality
        if site_quality >= self.recruitment_threshold:
            self.recruit_agents()
        return
    def get_ETH(self, agent):
        return
    def get_EE(self, agent):
        return