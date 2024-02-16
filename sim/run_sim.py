from World import World
import numpy as np
import pandas as pd
from helper_functions import *
import multiprocessing
import os
from params import *
import copy 

def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, inits, maxTimes):
    worlds = []

    for sites in site_configs:
        # for sim in range(0,sims_per_config):
        # pdb.set_trace()
            # qualities = quals[sim]
        for agents in agent_configs:
            for distance in distances:
                qualities = get_valid_qualities(sites, sims_per_config)
                for sim_dist_iter in range(0,sims_per_distance):
                    poses = get_poses(sites, distance)
                    for qual in qualities:
                        for agent_init in inits: #get_agent_inits(agents, poses, qual):
                            for mTimes in maxTimes:
                                for repeats in range(0,sim_repeats):           
                                    worlds.append([sites, qual, poses, agents, agent_init, mTimes])
    return worlds

def convert_to_init_agent(arr):
    arr2 = []
    for ag in range(len(arr[-1])):
        new_dict = {}
        new_dict['pose'] = arr[1][ag]
        new_dict['state'] = arr[-2][ag]
        new_dict['speed'] = 0.0 if (arr[-2][ag] == 'OBSERVE' or arr[-2][ag] == 'RECRUIT' 
                                    or arr[-2][ag] == 'ASSESS') else 5.0
        new_dict['site'] = arr[-1][ag]
        new_dict['dir'] = arr[2][ag]
        arr2.append(new_dict.copy())
    return arr2

def get_init_condition_from_df(df, starts):
    init_condition = []
    print(len(df), starts)
    ids = np.random.choice(range(len(df)), starts, replace=False)
    for id in ids:
        # pdb.set_trace()
        dat = copy.deepcopy(df[id])
        init_condition.append(convert_to_init_agent(dat))
    return init_condition

def generate_world_configs_from_init_sims(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, starts, fname):
    init_configs = []
    # for agent_init in get_agent_inits(agents, poses, qual):

    for sites in site_configs:
        # for sim in range(0,sims_per_config):
        # pdb.set_trace()
            # qualities = quals[sim]
        for agents in agent_configs:
            for distance in distances:
                qualities = get_valid_qualities(sites, sims_per_config)
                for sim_dist_iter in range(0,sims_per_distance):
                    poses = get_poses(sites, distance)
                    for qual in qualities:
                        for agent_init in get_agent_inits(agents, poses, qual):
                            # for repeats in range(0,sim_repeats):           
                            worlds = [sites, qual, poses, agents, agent_init, TIME_LIMIT]
                            # pdb.set_trace()
                            w = World(worlds, fname, save=False)
                            w.simulate()
                            init_configs.append(agent_init)
                            init_configs += get_init_condition_from_df(w.list_for_df, starts)
                            
    return init_configs
    # return worlds



def simulate_world(sim, world):
    world.simulate()
    print(sim, ' done')

if __name__ == '__main__':
    site_configs = [2]#[2, 3]#[2, 3, 4]#[2, 3, 4] #[2, 3, 4]
    distances = [100, 200, 150]#[100, 200]#, 300]#, 300]
    agent_configs = [10]#[5, 10, 20] #[5, 20, 50, 100, 200]
    sims_per_config = 1 #20
    sims_per_distance = 1 #20
    sim_repeats = 10
    num_samples_per_starting_condition = 29
    maxTimes = [100, 500, 1000, 10000, 35000]
    fold_name = 'graphsage_results'
    fname_metadata = './' + fold_name + '/metadata.csv'
    df_metadata_cols = ['file_name', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged', 'start_state', 'maxTime']
    empty = pd.DataFrame([], columns=df_metadata_cols)
    file_exists = os.path.exists(fname_metadata)
    if file_exists:
        empty.to_csv(fname_metadata, mode='a', header=False, index=False)
    else:
        empty.to_csv(fname_metadata, index=False)
    
    random_inits = generate_world_configs_from_init_sims(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, num_samples_per_starting_condition, fold_name)
    # pdb.set_trace()
    worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats, random_inits, maxTimes)
    # pdb.set_trace()

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
