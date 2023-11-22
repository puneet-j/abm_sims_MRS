import numpy as np
from params import *
from helper_functions import *
from Transitions import State_Transitions
import copy 

class Agent:
    def __init__(self, world, init_pos=[0.0, 0.0], init_state='OBSERVE', init_site=None, init_speed=0.0, init_dir=[0.0, 1.0]):
        self.pos = copy.deepcopy(init_pos)
        self.state = copy.deepcopy(init_state) #'OBSERVE'
        self.assigned_site = copy.deepcopy(init_site)
        self.dir = copy.deepcopy(init_dir) #[0.0, 1.0]
        self.speed = copy.deepcopy(init_speed) #0.0
        # self.prev_state = None
        # self.at_site = 10000 # site number if at site, else 10000
        # self.at_hub = True
        self.world = world
        self.transitions = State_Transitions()

    def step(self):
        # for ag_temp in self.world.agents:
        #     print('in agent start: ', ag_temp.pos, ag_temp.state)
        # self.at_hub, self.at_site = get_at_hub_site(self)
        next_state, site_new = self.transitions.get_transition(self)
        if next_state == 'RECRUIT' and site_new is None:
            print(self.state, 'got this next state and site: ', next_state, site_new)
            # if site_new is None:
            pdb.set_trace()
        # self.prev_state = copy.deepcopy(self.state)
        if site_new == 0 or site_new == 1:#self.assigned_site == 0 or self.assigned_site == 1:
            pdb.set_trace()
        self.state = copy.deepcopy(next_state)

        # print(self.prev_state, self.state)
        # if not(site_new is None):
        self.assigned_site = copy.deepcopy(site_new)
        # for ag_temp in self.world.agents:
        #     print('in agent after transition: ', ag_temp.pos, ag_temp.state)

        if self.state == 'TRAVEL_HOME_TO_RECRUIT' or self.state == 'TRAVEL_HOME_TO_OBSERVE':
            if not agent_at_hub(self):
                self.goHome()
        elif self.state == 'TRAVEL_SITE':
            if not agent_at_assigned_site(self):
                self.goSite()
        elif self.state == 'EXPLORE':
            if agent_at_any_site(self) is None:
                self.explore()
        else:
            self.speed = 0.0
            random_value = np.random.random()*np.pi*2.0
            self.dir = [np.cos(random_value), np.sin(random_value)]
        # for ag_temp in self.world.agents:
        #     print('in agent end: ', ag_temp.pos, ag_temp.state)
        # if self.state != 'OBSERVE':
        #     pdb.set_trace()
        return
    
    def goHome(self):
        self.speed = AGENT_SPEED
        self.dir = get_home_dir(self.pos)
        # pdb.set_trace()
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
    
    def goSite(self):
        self.speed = AGENT_SPEED
        if self.assigned_site is None:
            pdb.set_trace()
        self.dir = get_site_dir(self.pos, self.assigned_site.pos)
        self.pos[0] += self.speed*self.dir[0]
        self.pos[1] += self.speed*self.dir[1]
    
    def explore(self):
        # for ag_temp in self.world.agents:
            # print('in explore start: ', ag_temp.pos, ag_temp.state)
        self.speed = AGENT_SPEED
        self.dir = get_explore_dir(self.dir)
        # for ag_temp in self.world.agents:
        #     print('in explore after getting dir: ', ag_temp.pos, ag_temp.state, ag_temp.dir)
        self.pos[0] += self.speed*self.dir[0]
        # for ag_temp in self.world.agents:
        #     print('in explore after updating X: ', ag_temp.pos, ag_temp.state, ag_temp.dir)
        self.pos[1] += self.speed*self.dir[1]
        # for ag_temp in self.world.agents:
        #     print('in explore end: ', ag_temp.pos, ag_temp.state)
