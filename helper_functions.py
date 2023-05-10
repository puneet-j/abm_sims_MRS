import numpy as np
from params import *

def get_dist_2D(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_poses(a, d):
    angles = np.pi/a
    poses = []
    for i in range(0,a):
        poses.append((d*np.cos(angles*i),d*np.sin(angles*i)))
    return poses

def get_home_dir(pos):
    dir = (-1.0*pos[0], -1.0*pos[1])
    return dir

def get_site_dir(pos, site_pos):
    dir = (site_pose[0] - pos[0], site_pos[1] - pos[1])
    dir = dir/np.sum(dir)
    return dir

def get_explore_dir(dir):
    if np.random.random > EXPLORE_RANDOMNESS:
        xchange = np.cos(EXPLORE_DIRECTION_CHANGE)
        ychange = np.sin(EXPLORE_DIRECTION_CHANGE)
        dir[0] += xchange 
        dir[1] += ychange 
        dir = dir/np.sum(dir)
    return dir

def agent_at_site(ag):
    if get_dist_2D(ag.pos,ag.assigned_site.pos) < SITE_SIZE:
        return True
    else:
        return False
    
def agent_at_hub(ag):
    if get_dist_2D(ag.pos,HUB_LOCATION) < SITE_SIZE:
        return True
    else:
        return False

def agent_at_any_site(ag):
    for st in ag.world.sites:
        if get_dist_2D(ag.pos,st.pos) < SITE_SIZE:
            return st.id
    return False

def get_at_hub_site(ag):
    at_hub = agent_at_hub(ag)
    at_site = agent_at_any_site(ag)
    return at_hub, at_site

def get_dancers_by_site(ag):
    world = ag.world
    all_agents = world.agents
    dancers = [0]*world.num_sites
    for agent in agents:
        if agent.state == 'DANCE':
            dancers[agent.assigned_site.id] += 1
    return dancers

def get_site_from_id(ag):
    world = ag.world
    all_sites = world.sites
    for site in all_sites:
        if ag.at_site == site.id:
            return site