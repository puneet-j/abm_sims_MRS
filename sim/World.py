import numpy as np
from params import *
from Site import Site
from Agent import Agent
from helper_functions import *
import time
import pandas as pd 
from random import shuffle
import copy

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
        self.hub = Site(10000, 0.0, [0.0, 0.0])
        self.time = 0
        time_now = str(int(time.time()*1000000))
        self.fname = '../sim_results/' + time_now + '.csv'
        self.fname_metadata = '../sim_results/' + time_now + 'metadata.csv'
        self.df_metadata_cols = ['num_sites', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged']
        self.df_cols = ['time', 'agent_positions', 'agent_directions', 'agent_states', 'agent_sites']
        self.converged_to_site = None
        self.threshold = COMMIT_THRESHOLD

    def save_metadata(self):
        arr = [self.num_sites, tuple(self.site_qualities), tuple(self.site_poses), self.hub.pos, self.num_agents, self.converged_to_site, self.time]
        df_metadata = pd.DataFrame(np.array(arr, dtype=object).reshape(1,7), columns=self.df_metadata_cols)
        df_metadata.to_csv(self.fname_metadata)
        return
    
    def add_agents(self):
        self.agents += [Agent([0.0, 0.0], self) for agent_id in range(0, self.num_agents)]
        return

    def simulate(self):

        self.add_agents()
        # pdb.set_trace()
        list_for_df = []
        agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
        to_save = [self.time, copy.deepcopy(agent_poses), copy.deepcopy(agent_dirs), copy.deepcopy(agent_states), copy.deepcopy(agent_sites)]
        list_for_df.append(to_save[:])
        while self.time < TIME_LIMIT:
            # shuffle(self.agents)
            for agent in self.agents:
                agent.step()
            
            self.time += 1            
            agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
            to_save = [self.time, copy.deepcopy(agent_poses), copy.deepcopy(agent_dirs), copy.deepcopy(agent_states), copy.deepcopy(agent_sites)]
            list_for_df.append(to_save[:])
            dancers = get_dancers_by_site_for_world(self)
            if np.max(dancers) > self.threshold*self.num_agents:
                self.converged_to_site = self.sites[np.argmax(dancers)].quality
                break


        df = pd.DataFrame(list_for_df, columns = self.df_cols)
        df.to_csv(self.fname)
        self.save_metadata()
        
