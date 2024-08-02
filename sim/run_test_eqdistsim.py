from World import World
import numpy as np
import pandas as pd
from helper_functions import *
import multiprocessing
import os
from params import *
import copy 

# def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, inits, maxTimes):
#     worlds = []

#     for sites in site_configs:
#         # for sim in range(0,sims_per_config):
#         # pdb.set_trace()
#             # qualities = quals[sim]
#         for agents in agent_configs:
#             for distance in distances:
#                 qualities = get_valid_qualities(sites, sims_per_config)
#                 for sim_dist_iter in range(0,sims_per_distance):
#                     poses = get_poses(sites, distance)
#                     for qual in qualities:
#                         for agent_init in inits: #get_agent_inits(agents, poses, qual):
#                             for mTimes in maxTimes:
#                                 for repeats in range(0,sim_repeats):           
#                                     worlds.append([sites, qual, poses, agents, agent_init, mTimes])
#     return worlds

def convert_to_init_agent(arr):
    arr2 = []
    for ag in range(0,len(arr[1])):
        try:
            # self.df_cols = ['time', 'agent_positions', 
            # 'agent_directions', 'agent_states', 
            # 'agent_sites', 'node', 'new_node']

            # pdb.set_trace()
            new_dict = {}
            new_dict['pose'] = arr[1][ag]
            new_dict['state'] = arr[3][ag]
            new_dict['speed'] = 0.0 if (arr[3][ag] == 'OBSERVE' or arr[3][ag] == 'RECRUIT' 
                                        or arr[3][ag] == 'ASSESS') else 5.0
            new_dict['site'] = arr[4][ag]
            new_dict['dir'] = arr[2][ag]
            arr2.append(new_dict.copy())
        except Exception as e:
            print(e)
            pdb.set_trace()
    return arr2

def get_init_condition_from_df(dflist, df_cols, starts):
    init_condition = []
    df = pd.DataFrame(dflist, columns = df_cols)
    unique_node_ids = df['node'].unique()

    # Dictionary to store indices for each unique node feature
    ids = [df.index[df['node'] == feature].tolist()[0] for feature in unique_node_ids]
    print('total sim length for sampling and samples: ', len(df), len(ids))
    if len(ids) > starts:
        # pdb.set_trace()
        ids = np.random.choice(ids, starts, replace=False)
    for id in ids:
        # pdb.set_trace()
        dat = copy.deepcopy(dflist[id])
        # pdb.set_trace()
        init_condition.append(convert_to_init_agent(dat)+ [id, len(df)])
    return init_condition

def generate_world_configs_from_init_sims(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, starts, fname, mTimes, total_repeats):
    # for agent_init in get_agent_inits(agents, poses, qual):
    worlds = []
    for _ in range(0, total_repeats):
        for sites in site_configs:
            # for sim in range(0,sims_per_config):
            # pdb.set_trace()
                # qualities = quals[sim]
            for agents in agent_configs:
                qualities = np.array([[0.913, 0.196, 0.388, 0.893]])#get_valid_qualities(sites, sims_per_config)
                print('got qualities')
                for qual in qualities:
                    for distance in distances:
                        for sim_dist_iter in range(0,sims_per_distance):
                            poses = get_poses(sites, distance)
                            all_agent_inits = get_agent_inits(agents, poses, qual)
                            print('number of defined start states: ', len(all_agent_inits))
                            for agent_init in all_agent_inits:
                                # print(len())
                                # for repeats in range(0,sim_repeats):   
                                w_init = [sites, qual, poses, agents, agent_init, TIME_LIMIT, 0] 
                                worlds.append(w_init)
                                # pdb.set_trace()
                                w = World(w_init, fname, save=False)
                                w.simulate()
                                print('simulated first world')
                                init_configs = []
                                init_configs.append(agent_init)
                                init_configs += get_init_condition_from_df(w.list_for_df, w.df_cols, starts)
                                print('total init configs from random sampling: ', len(init_configs))
                                for init in init_configs:
                                    # print(len(init_))
                                    for mtime in mTimes:
                                        samples = sim_repeats
                                        # samples = int(init[-2]/init[-1]*L_CONST)
                                        # print(samples)
                                        for _ in range(0,samples):
                                            # print(agents)
                                            # qual = 
                                            worlds.append([sites, qual, poses, agents, init[:-2], mtime, 0])

        # print('got all init configs: ', len(worlds))   
    # if SPIDER_FLAG:
    #     tempworlds = []
    #     for w in worlds:
    #         wcopy = copy.deepcopy(w[:])
    #         wcopy[-1] = 1
    #         tempworlds.append(wcopy)
    #     worlds = worlds + tempworlds        
    return worlds
    # return worlds



def simulate_world(sim, world):
    world.simulate()
    print(sim, ' done')

if __name__ == '__main__':
    site_configs = [4]#[2, 3, 4] #[2, 3]#[2, 3, 4]#[2, 3, 4] #[2, 3, 4]
    distances = [150]#[100, 200, 150] #[100, 200]#, 300]#, 300]
    agent_configs = [10]#[100, 50, 20, 10, 5] #[5, 10, 20] #[5, 20, 50, 100, 200]
    sims_per_config = 1 #10 
    sims_per_distance = 2 # how many times we repeat this configuration
    sim_repeats = 30 # 10 # how many times we go out from each node.
    num_samples_per_starting_condition = 10 # 10 # number of starting points
    maxTimes = [1000]
    total_repeats = 1
    
    # maxTimes = [1000, 10000, 35000]
    fold_name = 'AAAI/data/lots_of_node_samples/test_new_2/'
    fname_metadata = './' + fold_name + 'metadata.csv'
    df_metadata_cols = ['file_name', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged', 'start_state', 'maxTime', 'timelimitsave', 'node', 'sims_spider', 'sims_train']
    empty = pd.DataFrame([], columns=df_metadata_cols)
    file_exists = os.path.exists(fname_metadata)
    if file_exists:
        empty.to_csv(fname_metadata, mode='a', header=False, index=False)
    else:
        empty.to_csv(fname_metadata, index=False)
    
    worlds = generate_world_configs_from_init_sims(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, num_samples_per_starting_condition, fold_name, maxTimes, total_repeats)
    # pdb.set_trace()
    # worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, random_inits, maxTimes)
    # pdb.set_trace()
    print('number of worlds: ', len(worlds))
    print('starting sims')

    '''comment this for testing'''
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool()
    results = [pool.apply_async(simulate_world, args=(sim, World(w, fold_name))) for sim, w in enumerate(worlds)]
    pool.close()
    pool.join()

    '''uncomment this for testing'''
    # for sim, w in enumerate(worlds):
    #     world = World(w, fold_name)
    #     world.simulate()
    #     print(sim, ' done')
