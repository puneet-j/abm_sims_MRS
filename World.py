import numpy as np
from params import *
import Site
import Agent
from Transitions import State_Transition
from helper_functions import *
import time
import pandas as pd 

class World:
    def __init__(self, num_sites, site_quals, site_poses, num_agents):
        self.num_sites = num_sites
        self.site_qualities = site_quals
        self.site_poses = site_poses
        self.sites = [Site(site_id, site_qual, site_pos) for 
                      site_id, site_qual, site_pos in zip(range(0,num_sites), site_quals, site_poses)]
        self.num_agents = num_agents
        self.agents = []
        self.hub = Site(10000, 0.0, (0.0,0.0))
        self.time = 0
        self.transitions = State_Transition(self)
        self.fname = './sim_results/' + str(int(time.time()*1000000)) + '.csv'
        self.df = pd.DataFrame(columns = ['time', 'num_sites', 'site_qualities', 'site_positions', 'hub_position', 
                                          'num_agents', 'agent_positions', 'agent_directions', 'agent_states', 'agent_sites'])

    def add_agents(self):
        for i in range(self.num_agents):
            self.agents.append(Agent(agent_id, (0,0), self) for agent_id in range(0,self.num_agents))

    def simulate(self):
        while self.time < TIME_LIMIT:
            for agent in np.random.perumtation(self.agents):
                agent.step()
            agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
            self.df.append([self.time, self.num_sites, self.site_qualities, self.site_poses, self.hub.pos, 
                            self.num_agents, agent_poses, agent_dirs, agent_states, agent_sites])
            self.time += 1
        self.df.to_csv(self.fname)
        
