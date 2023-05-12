import numpy as np
from params import *
from Site import Site
from Agent import Agent
from helper_functions import *
import time
import pandas as pd 
from random import shuffle

class World:
    def __init__(self, params):
        # params = (num_sites, site_quals, site_poses, num_agents)
        self.num_sites = params[0]
        self.site_qualities = params[1]
        self.site_poses = params[2]
        self.sites = [Site(site_id, site_qual, site_pos) for 
                      site_id, site_qual, site_pos in zip(range(0,self.num_sites), self.site_qualities, self.site_poses)]
        self.num_agents = params[3]
        self.agents = []
        self.hub = Site(10000, 0.0, (0.0,0.0))
        self.time = 0
        self.fname = './sim_results/' + str(int(time.time()*1000000)) + '.csv'
        self.df = pd.DataFrame(columns = ['time', 'num_sites', 'site_qualities', 'site_positions', 'hub_position', 
                                          'num_agents', 'agent_positions', 'agent_directions', 'agent_states', 'agent_sites'])

    def add_agents(self):
        self.agents += [Agent((0,0), self) for agent_id in range(0,self.num_agents)]

    def simulate(self):
        self.add_agents()
        while self.time < TIME_LIMIT:
            shuffle(self.agents)
            for agent in self.agents:
                agent.step()
            agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
            ''' append to list and make df in end'''
            self.df.append([self.time, self.num_sites, self.site_qualities, self.site_poses, self.hub.pos, 
                            self.num_agents, agent_poses, agent_dirs, agent_states, agent_sites])
            self.time += 1
        self.df.to_csv(self.fname)
        
