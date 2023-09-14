from World import World
import numpy as np
import pandas as pd
from helper_functions import *
import multiprocessing


def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats):
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
                        for agent_init in get_agent_inits(agents, poses, qual):
                            for repeats in range(0,sim_repeats):           
                                worlds.append([sites, qual, poses, agents, agent_init])
    return worlds

def simulate_world(sim, world):
    world.simulate()
    print(sim, ' done')

if __name__ == '__main__':
    site_configs = [2]#[2, 3]#[2, 3, 4]#[2, 3, 4] #[2, 3, 4]
    distances = [100, 200]#, 300]#, 300]
    agent_configs = [20] #[5, 20, 50, 100, 200]
    sims_per_config = 10 #20
    sims_per_distance = 10 #20
    sim_repeats = 20
    fname_metadata = './results_for_graphs/metadata.csv'
    df_metadata_cols = ['file_name', 'site_qualities', 'site_positions', 'hub_position', 'num_agents', 'site_converged', 'time_converged']
    empty = pd.DataFrame([], columns=df_metadata_cols)
    empty.to_csv(fname_metadata)
    worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance, sim_repeats)
    # pdb.set_trace()
    '''comment this for testing'''
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool()
    results = [pool.apply_async(simulate_world, args=(sim, World(w))) for sim, w in enumerate(worlds)]
    pool.close()
    pool.join()

    '''uncomment this for testing'''
    # for sim, w in enumerate(worlds):
    #     world = World(w)
    #     world.simulate()
    #     print(sim, ' done')
