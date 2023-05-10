import numpy as np
from params import *
from helper_functions import *

class Agent(self):
    def __init__(self, pos, world):
        self.pos = pos
        self.state = 'REST'
        self.assigned_site = None
        self.dir = (0.0, 1.0)
        self.speed = 0.0
        self.prev_state = None
        self.at_site = 10000 # site number if at site, else 10000
        self.at_hub = True
        self.world = world

    def step(self):
        self.at_hub, self.at_site = get_at_hub_site(self)
        next_state = self.transitions.get_transition(self)
        self.prev_state = self.state.copy()
        self.state = next_state

        if self.state == 'TRAVEL_HOME_ASSESS' or self.state == 'TRAVEL_HOME_EXPLORE':
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
            self.dir = (0.0, 1.0)
        return
    
    def goHome(self):
        self.speed = AGENT_SPEED
        self.dir = get_home_dir(self.pos)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
        return
    
    def goSite(self):
        self.speed = AGENT_SPEED
        self.dir = get_site_dir(self.pos, self.assigned_site.pos)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
        return
    
    def explore(self):
        self.speed = AGENT_SPEED
        self.dir = get_explore_dir(self.dir)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
        return       
