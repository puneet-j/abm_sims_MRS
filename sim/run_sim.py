from World import World
import numpy as np
import pandas as pd
from helper_functions import *
import multiprocessing


def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance):
    worlds = []

    for sites in site_configs:
        # for sim in range(0,sims_per_config):
        # pdb.set_trace()
            # qualities = quals[sim]
        for agents in agent_configs:
            for distance in distances:
                qualities = get_valid_qualities(sites, sims_per_config)
                for sim_dist in range(0,sims_per_distance):
                    poses = get_poses(sites, distance)
                    for qual in qualities:                    
                        worlds.append([sites, qual, poses, agents])
    return worlds

def simulate_world(sim, world):
    world.simulate()
    print(sim, ' done')

if __name__ == '__main__':
    site_configs = [2]#[2, 3]#[2, 3, 4]#[2, 3, 4] #[2, 3, 4]
    distances = [100, 200]#, 300]#, 300]
    agent_configs = [20] #[5, 20, 50, 100, 200]
    sims_per_config = 20 #20
    sims_per_distance = 10 #20
    worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance)
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
