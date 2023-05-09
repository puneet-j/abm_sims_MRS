import World
import numpy as np
import pandas as pd
from helper_functions import *

def generate_world_configs(site_configs, distances, agent_configs):
    worlds = []
    for distance in distances:
        for sites in site_configs:
            qualities = list(np.random.random(sites))
            poses = get_poses(sites, distance)
            for agents in agent_configs:
                worlds.append(sites, qualities, poses, agents)
    return worlds

def main():
    site_configs = [2,3,4]
    distances = [50, 100, 200]
    agent_configs = [20, 50, 100, 200]
    time_limit = 35000
    worlds = generate_world_configs(site_configs, distances, agent_configs)

    for w in worlds:
        world = World(w)
        while world.time < time_limit:
            world.simulate()

main()