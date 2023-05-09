import numpy as np
from params import *
from Site import Site
from Agent import Agent
from Transitions import State_Transition
from helper_functions import *

class World:
    def __init__(self, num_sites, num_agents):
        self.sites = [Site(site_id) for site_id in range(num_sites)]
        self.num_agents = num_agents
        self.agents = []
        self.hub = Site(10000,0,(0,0))
        self.time = 0
        self.transitions = State_Transition(self)

    def add_agents(self):
        for i in range(self.num_agents):
            self.agents.append(Agent(agent_id, (0,0), self) for agent_id in range(self.num_agents))

    def simulate(self):
        for agent in self.agents:
            agent.step()
        self.time += 1
        

    def agent_at_any_site(self, agent):
        for site in self.sites:
            if get_dist_2D(site.pos,agent.pos) < SITE_SIZE:
                return site 
            else:
                return None