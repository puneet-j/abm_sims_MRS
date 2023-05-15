from World import World
import numpy as np
import pandas as pd
from helper_functions import *

def generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance):
    worlds = []
    for sim in range(0,sims_per_config):
        for distance in distances:
            for sim_dist in range(0,sims_per_distance):
                for sites in site_configs:
                    qualities = list(np.random.random(sites))
                    poses = get_poses(sites, distance)
                    for agents in agent_configs:
                        worlds.append([sites, qualities, poses, agents])
    return worlds

def main():
    site_configs = [2,3,4]
    distances = [100, 200]
    agent_configs = [5, 20, 50, 100, 200]
    sims_per_config = 2
    sims_per_distance = 2
    worlds = generate_world_configs(site_configs, distances, agent_configs, sims_per_config, sims_per_distance)
    for sim, w in enumerate(worlds):
        world = World(w)
        world.simulate()
        print(sim, ' done')

main()