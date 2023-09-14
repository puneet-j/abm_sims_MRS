import numpy as np
from params import *
from Site import Site
from Agent import Agent
from helper_functions import *
import time
import pandas as pd 
from random import shuffle
import copy
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))

class World:
    def __init__(self, params):
        # params = (num_sites, site_quals, site_poses, num_agents)
        self.num_sites = params[0]
        self.site_qualities = params[1]
        if self.num_sites != len(self.site_qualities):
            pdb.set_trace()
        self.site_poses = params[2]
        self.sites = [Site(site_id, site_qual, site_pos) for 
                      site_id, site_qual, site_pos in zip(range(0,self.num_sites), self.site_qualities, self.site_poses)]
        self.num_agents = params[3]
        self.agent_configs_init = params[4]
        self.agents = []
        self.hub = Site(10000, 0.0, [0.0, 0.0])
        self.time = 0
        self.file_time = str(int(time.time()*1000000))
        self.fname = './results_for_graphs/' + self.file_time + '.csv'
        self.fname_metadata = './results_for_graphs/metadata.csv'
        self.df_metadata_cols = ['file_name', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged']
        self.df_cols = ['time', 'agent_positions', 'agent_directions', 'agent_states', 'agent_sites']
        self.converged_to_site = None
        self.threshold = COMMIT_THRESHOLD
        self.list_for_df = []

    def save_metadata(self):
        arr = [self.file_time + '.csv', tuple(self.site_qualities), tuple(self.site_poses), self.hub.pos, self.num_agents, self.converged_to_site, self.time]
        df_metadata = pd.DataFrame(np.array(arr, dtype=object).reshape(1,7), columns=self.df_metadata_cols)
        df_metadata.to_csv(self.fname_metadata, header=False, mode='a', index=False)
        return
    
    def add_agents(self):
        for config in self.agent_configs_init:
            if len(config) > 1:
                # print('got agent config')
                ag = Agent(self, init_pos=config['pose'], init_state=config['state'], 
                            init_site=config['site'], init_speed=config['speed'], init_dir=config['dir'])  
            else:
                # print('default agent config')
                ag = Agent(self)
            # print(id(ag.pos), id(ag.state), id(ag.assigned_site), id(ag.speed), id(ag.dir))
            self.agents.append(ag)
        return

    def simulate(self):

        self.add_agents()
        # print(self.agents[0].pos is self.agents[1].pos)

        # pdb.set_trace()
        # print(self.agents[0].pos is self.agents[1].pos)
        agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
        to_save = [self.time, copy.deepcopy(agent_poses), copy.deepcopy(agent_dirs), copy.deepcopy(agent_states), copy.deepcopy(agent_sites)]
        self.list_for_df.append(to_save[:])
        while self.time < TIME_LIMIT:
            # shuffle(self.agents)
            for iter in range(0, self.num_agents):
                ag = self.agents[iter]#copy.deepcopy()
                # print('world itme: ', self.time)
                # print(self.agents[0].pos is self.agents[1].pos)
                # print('before agent step: ', ag.pos, ag.state)
                ag.step()
                # print('after agent step: ', ag.pos, ag.state)
                # pdb.set_trace()

            self.time += 1            
            agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
            # print('after getting all agent poses dirs: ', agent.pos, agent.state)
            to_save = [self.time, copy.deepcopy(agent_poses), copy.deepcopy(agent_dirs), copy.deepcopy(agent_states), copy.deepcopy(agent_sites)]
            self.list_for_df.append(to_save[:])
            # print('after appending to list: ', agent.pos, agent.state)
            dancers = get_dancers_by_site_for_world(self)
            # print('after getting dancers: ', agent.pos, agent.state)
            if np.max(dancers) > self.threshold*self.num_agents:
                self.converged_to_site = self.sites[np.argmax(dancers)].quality
                break


        df = pd.DataFrame(self.list_for_df, columns = self.df_cols)
        df.to_csv(self.fname)
        self.save_metadata()
        
