import numpy as np
from params import *

def get_dist_2D(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_poses(a, d):
    angles = 2*np.pi/a
    poses = []
    for i in range(0,a):
        poses.append([d*np.cos(angles*i),d*np.sin(angles*i)])
    return list(poses)

def get_home_dir(pos):
    dir = [-1.0*pos[0], -1.0*pos[1]]
    # pdb.set_trace()
    dir = dir/np.linalg.norm(dir)
    return list(dir)

def get_site_dir(pos, site_pos):
    dir = [site_pos[0] - pos[0], site_pos[1] - pos[1]]
    dir = dir/np.linalg.norm(dir)
    return list(dir)

def get_explore_dir(dir):
    if np.random.random() > EXPLORE_RANDOMNESS:
        to_change_dir = EXPLORE_DIRECTION_CHANGE*np.random.choice([1, -1],p=[0.5, 0.5])
        xchange = to_change_dir
        ychange = to_change_dir
        dir[0] += xchange 
        dir[1] += ychange 
        dir = dir/np.linalg.norm(dir)
    return list(dir)

def agent_at_site(ag):
    if ag.assigned_site == None:
        return False
    
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
            return st
    return False

# def agent_at_hub_or_site(ag):
#     at_hub = agent_at_hub(ag)
#     at_site = agent_at_any_site(ag)
#     return at_hub, at_site
def get_dancers_by_site_for_world(world):
    all_agents = world.agents
    dancers = [0]*world.num_sites
    for agent in world.agents:
        if agent.state == 'DANCE':
            dancers[agent.assigned_site.id] += 1
    return dancers

def get_dancers_by_site(ag):
    world = ag.world
    all_agents = world.agents
    dancers = [0]*world.num_sites
    for agent in world.agents:
        if agent.state == 'DANCE':
            dancers[agent.assigned_site.id] += 1
    return dancers

# def get_site_from_id(ag):
#     world = ag.world
#     all_sites = world.sites
#     for site in all_sites:
#         if ag.assigned_site == site.id:
#             return site
        
def get_all_agent_poses_dirs_states_sites(world):
    poses = []
    dirs = []
    states = []
    sites = []
    for agent in world.agents:
        poses.append(agent.pos)
        dirs.append(agent.dir)
        states.append(agent.state)
        if agent.assigned_site != None:
            sites.append(agent.assigned_site.pos)
        else:
            sites.append(None)
    return poses, dirs, states, sites
