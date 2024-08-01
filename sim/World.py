import numpy as np
from params import *
from Site import Site
from Agent import Agent
from helper_functions import *
import time
import pandas as pd 
from random import shuffle
import copy
from ast import literal_eval
# import sys, os
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'sim'))

class World:
    def __init__(self, params, fold_name, save=True):
        # params = (num_sites, site_quals, site_poses, num_agents)
        # print(params)
        
        self.save = save
        self.num_sites = params[0]
        self.site_qualities = params[1]
        if self.num_sites != len(self.site_qualities):
            pdb.set_trace()
        self.site_poses = params[2]
        self.sites = [Site(site_id, site_qual, site_pos) for 
                      site_id, site_qual, site_pos in zip(range(0,self.num_sites), self.site_qualities, self.site_poses)]
        self.num_agents = params[3]
        self.agent_configs_init = params[4]
        self.timeLimit = params[5]
        self.spider_flag = params[6]
        self.agents = []
        self.hub = Site(10000, 0.0, [0.0, 0.0])
        self.time = 0
        self.file_time = str(int(time.time()*1000000))
        self.fname = './' + fold_name + '/' + self.file_time + '.csv'
        self.fname_metadata = './' + fold_name + '/' + 'metadata.csv'
        self.df_metadata_cols = ['file_name', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged', 'start_state', 'maxTime', 'timelimitsave', 'node', 'sims_spider', 'sims_train']
        self.df_cols = ['time', 'agent_positions', 'agent_directions', 'agent_states', 'agent_sites', 'node', 'new_node']
        self.converged_to_site = None
        self.threshold = COMMIT_THRESHOLD
        self.list_for_df = []
        # self.convflag = False
        # print('running sim: ', self.file_time)

    def save_metadata(self, st):
        # pdb.set_trace()
        rounded_conv = np.round(self.converged_to_site, 3) if self.converged_to_site is not None else None
        arr = [self.file_time + '.csv', list(np.round(tuple(self.site_qualities), 3)), 
               round_function(tuple(self.site_poses), 3), self.hub.pos, self.num_agents, 
               rounded_conv, self.time, 
               tuple([tuple([conf['state'],conf['site']]) for conf in self.agent_configs_init if len(conf)>1]), 
               self.timeLimit, KEEP_SIMS_LIMIT, st, [], self.file_time + '.csv']
        df_metadata = pd.DataFrame(np.array(arr, dtype=object).reshape(1,13), columns=self.df_metadata_cols)
        df_metadata.to_csv(self.fname_metadata, header=False, mode='a', index=False)
        return
    
    def add_agents(self):
        # pdb.set_trace()
        for config in self.agent_configs_init:
            if len(config) > 1:
                try:
                    site_to_attach_init = None if config['site'] is None else self.sites[config['site']]
                except:
                    print(config['site'])
                    pdb.set_trace()
                # print('got agent config')
                ag = Agent(self, init_pos=config['pose'], init_state=config['state'], 
                            init_site=site_to_attach_init, init_speed=config['speed'], init_dir=config['dir'])  
            else:
                # print('default agent config')
                ag = Agent(self)
            # print(id(ag.pos), id(ag.state), id(ag.assigned_site), id(ag.speed), id(ag.dir))
            self.agents.append(ag)
        return
    
    def append_to_metadata(self, init_state):
        df = pd.read_csv(self.fname_metadata, header=0)
        mask = (df['node'] == str(init_state)) & (df['site_qualities'] == str(list(np.round(tuple(self.site_qualities), 3))))
        # print(len(mask) == 0 or mask.any() == False)
        # pdb.set_trace()
        if len(mask) == 0 or mask.any() == False:
            self.save_metadata(init_state)
            return
        # try:
        df.time_converged=df.time_converged.apply(lambda x: literal_eval(x))
        df.site_converged=df.site_converged.apply(lambda x: literal_eval(x))
        df.sims_train=df.sims_train.apply(lambda x: literal_eval(x))

        rounded_conv = np.round(self.converged_to_site, 3) if self.converged_to_site is not None else None
        # Append the new value to time_converged for rows where the node matches
        for id, m in enumerate(mask):
            if m:
                df.at[id, 'time_converged'].append(self.time)
                df.at[id, 'site_converged'].append(rounded_conv)
                df.at[id, 'sims_train'].append(self.file_time+'.csv')
        df.to_csv(self.fname_metadata, index=False)

        return
    
    def append_to_metadata_sim_name(self, init_state):
        df = pd.read_csv(self.fname_metadata, header=0)
        mask = (df['node'] == str(init_state)) & (df['site_qualities'] == str(list(np.round(tuple(self.site_qualities), 3))))
        if len(mask) == 0 or mask.any() == False:
            pdb.set_trace()
        # Append the new value to time_converged for rows where the node matches
        for id, m in enumerate(mask):
            if m:
                df.at[id, 'sims_spider'].append(self.file_time+'.csv')
        df.to_csv(self.fname_metadata, index=False)

        return

    def simulate(self):

        self.add_agents()
        # print(self.agents[0].pos is self.agents[1].pos)

        # pdb.set_trace()
        # print(self.agents[0].pos is self.agents[1].pos)
        try:
            agent_poses, agent_dirs, agent_states, agent_sites = get_all_agent_poses_dirs_states_sites(self)
        except:
            pdb.set_trace()
        
        rounded_poses = round_function(copy.deepcopy(agent_poses[:]), 3)
        rounded_dirs = round_function(copy.deepcopy(agent_dirs[:]), 3)
        rounded_quals = tuple(np.round(copy.deepcopy(self.site_qualities), 3))
        rounded_s_poses = tuple(round_function(copy.deepcopy(self.site_poses), 3))
        copy_node = get_current_state(copy.deepcopy(agent_states), copy.deepcopy(agent_sites), rounded_poses, rounded_s_poses, rounded_quals)
        new_copy_node = new_state_from_old(copy_node)# pdb.set_trace()
        init_state_for_meta = copy.deepcopy(new_copy_node)
        to_save = [self.time, rounded_poses, rounded_dirs, copy.deepcopy(agent_states), copy.deepcopy(agent_sites), copy_node, copy.deepcopy(new_copy_node)]
        self.list_for_df.append(to_save[:])
        while self.time < TIME_LIMIT:
            # shuffle(self.agents)
            # pdb.set_trace()
            # np.sum([1.0 for i in agent_states if i=='RECRUIT'])
            # np.sum([1.0 for i in agent_states if i=='EXPLORE'])
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
            if np.any([a == 'EXPLORE' and (b is not None) for a,b in zip(agent_states, agent_sites)]):
                pdb.set_trace()
            if np.any([a == 'ASSESS' and (b is None) for a,b in zip(agent_states, agent_sites)]):
                pdb.set_trace()
            
            
            rounded_poses = round_function(copy.deepcopy(agent_poses[:]), 3)
            rounded_dirs = round_function(copy.deepcopy(agent_dirs[:]), 3)
            rounded_quals = tuple(np.round(copy.deepcopy(self.site_qualities), 3))
            rounded_s_poses = tuple(round_function(copy.deepcopy(self.site_poses), 3))
            copy_node = get_current_state(copy.deepcopy(agent_states), copy.deepcopy(agent_sites), rounded_poses, rounded_s_poses, rounded_quals)
            new_copy_node = new_state_from_old(copy_node)
            to_save = [self.time, rounded_poses, rounded_dirs, copy.deepcopy(agent_states), copy.deepcopy(agent_sites), copy_node, copy.deepcopy(new_copy_node)]
            self.list_for_df.append(to_save[:])
            # print('after appending to list: ', agent.pos, agent.state)
            RECRUITrs = get_RECRUITrs_by_site_for_world(self)
            # print('after getting RECRUITrs: ', agent.pos, agent.state)
            # pdb.set_trace()
            if np.max(RECRUITrs) > self.threshold*self.num_agents:
                self.converged_to_site = self.sites[np.argmax(RECRUITrs)].quality
                print(self.converged_to_site)
                break
            if self.spider_flag == 1:
                pdb.set_trace()
                if self.time > 100:
                    break
        
        print(self.time)
        # print(self.time, end=" ")
        if self.save:
            if self.spider_flag == 1:
                pdb.set_trace()
                df = pd.DataFrame(self.list_for_df, columns = self.df_cols)
                df.to_csv(self.fname, chunksize=7000)
                self.append_to_metadata_sim_name(init_state_for_meta)
            elif self.converged_to_site is not None:
                # pdb.set_trace()
                # pdb.set_trace()
                df = pd.DataFrame(self.list_for_df[:self.timeLimit], columns = self.df_cols)
                df.to_csv(self.fname, chunksize=7000)
                self.save_metadata(init_state_for_meta)
                # self.append_to_metadata(init_state_for_meta)
                # self.save_metadata(init_state_for_meta)
                # if self.time < self.timeLimit:
                # if self.time <= KEEP_SIMS_LIMIT:
                # if self.converged_to_site is not None:
                #     self.append_to_metadata(init_state_for_meta)
            else:
                return
        return 
    
