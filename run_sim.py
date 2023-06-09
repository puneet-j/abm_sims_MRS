from World import World
import numpy as np
import pandas as pd
from helper_functions import *
import multiprocessing

def get_valid_qualities(num_sites):
    quals = np.random.random(num_sites)
    while np.max(quals) < 0.5:
        quals = np.random.random(num_sites)
    return list(quals)

def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance):
    worlds = []

    for sites in site_configs:
        qualities = []
        for sim in range(0,sims_per_config):
            qualities.append(get_valid_qualities(sites))
        for agents in agent_configs:
            for distance in distances:
                for sim_dist in range(0,sims_per_distance):
                    poses = get_poses(sites, distance)
                    for qual in qualities:                    
                        worlds.append([sites, qual, poses, agents])
    return worlds

def simulate_world(sim, world):
    world.simulate()
    print(sim, ' done')

if __name__ == '__main__':
    site_configs = [2]#[2, 3, 4] #[2, 3, 4]
    distances = [100, 200]
    agent_configs = [20] #[5, 20, 50, 100, 200]
    sims_per_config = 100 #20
    sims_per_distance = 100 #20
    worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool()
    results = [pool.apply_async(simulate_world, args=(sim, World(w))) for sim, w in enumerate(worlds)]
    pool.close()
    pool.join()

        # for sim, w in enumerate(worlds):
        #     world = World(w)
        #     world.simulate()
        #     print(sim, ' done')
