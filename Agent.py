import numpy as np
from params import *
from helper_functions import *
from Transitions import State_Transitions
import copy 

class Agent:
    def __init__(self, pos, world):
        self.pos = pos
        self.state = 'REST'
        self.assigned_site = None
        self.dir = [0.0, 1.0]
        self.speed = 0.0
        self.prev_state = None
        self.at_site = 10000 # site number if at site, else 10000
        self.at_hub = True
        self.world = world
        self.transitions = State_Transitions(self)

    def step(self):
        self.at_hub, self.at_site = get_at_hub_site(self)
        next_state, site_new = self.transitions.get_transition(self)
        # print(self.prev_state, self.state, 'got this next state: ', next_state)
        self.prev_state = copy.deepcopy(self.state)
        self.state = copy.deepcopy(next_state)
        # print(self.prev_state, self.state)
        if site_new:
            self.assigned_site = site_new

        if self.state == 'TRAVEL_HOME_TO_DANCE' or self.state == 'TRAVEL_HOME_TO_REST':
            if not agent_at_hub(self):
                self.goHome()
        elif self.state == 'TRAVEL_SITE':
            if not agent_at_site(self):
                self.goSite()
        elif self.state == 'EXPLORE':
            if not agent_at_any_site(self):
                self.explore()
        else:
            self.speed = 0.0
            self.dir = [0.0, 1.0]
        return
    
    def goHome(self):
        self.speed = AGENT_SPEED
        self.dir = get_home_dir(self.pos)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
    
    def goSite(self):
        self.speed = AGENT_SPEED
        self.dir = get_site_dir(self.pos, self.assigned_site.pos)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
    
    def explore(self):
        self.speed = AGENT_SPEED
        self.dir = get_explore_dir(self.dir)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
