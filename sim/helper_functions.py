import numpy as np
from params import *
import random

def get_agent_inits(agents, site_poses, site_quals):
    agent_dict_list = []
    agent_dict_list.append(get_all_OBSERVE(agents))
    agent_dict_list.append(get_half_explore(agents))
    agent_dict_list.append(get_10perc_dancing_bad(agents, site_quals))
    return agent_dict_list

def get_all_OBSERVE(agents):
    new_list = []
    for a in range(0,agents):
        new_dict = {}
        new_list.append(new_dict)
    return new_list

def get_half_explore(agents):
    new_list = []
    prob_explore = 0.5
    agents_explore = int(np.floor(0.5*agents))
    explore_and_observe = [1]*agents_explore + [0]*(agents - agents_explore)
    random.shuffle(explore_and_observe)
    for a in explore_and_observe:
        new_dict = {}
        if a==1:
            new_dict['pose'] = [2000.0 * np.random.random() - 1000.0, 2000.0 * np.random.random() - 1000.0]
            new_dict['state'] = 'EXPLORE'
            new_dict['speed'] = AGENT_SPEED
            new_dict['site'] = None
            dirs = [np.random.random(), np.random.random()]
            new_dict['dir'] = [dir/np.sum(dirs) for dir in dirs]
        else:
            new_dict['pose'] = [0.0, 0.0]
            new_dict['state'] = 'OBSERVE'
            new_dict['speed'] = 0.0
            new_dict['site'] = None
            new_dict['dir'] = [0.0, 1.0]
        new_list.append(new_dict)
    return new_list

def get_10perc_dancing_bad(agents, site_quals):
    new_list = []
    agents_dancing_bad = int(np.floor(0.1*agents))
    dancing_and_other = [1]*agents_dancing_bad + [0]*(agents - agents_dancing_bad)
    random.shuffle(dancing_and_other)
    for a in dancing_and_other:
        new_dict = {}
        if a==1:
            new_dict['pose'] = [0.0, 0.0]
            new_dict['state'] = 'RECRUIT'
            new_dict['speed'] = 0.0
            bad_site = np.argmin(site_quals)
            new_dict['site'] = bad_site
            new_dict['dir'] = [0.0, 1.0]
        else:
            new_dict['pose'] = [0.0, 0.0]
            new_dict['state'] = 'OBSERVE'
            new_dict['speed'] = 0.0
            new_dict['site'] = None
            new_dict['dir'] = [0.0, 1.0]
        new_list.append(new_dict)
    return new_list

# def get_valid_qualities_fixed_2(num_configs):
#     quals = qualities2[:num_configs]
#     return quals

def get_valid_qualities(num_sites, num_configs):
    qual_arr = []
    for config in range(0,num_configs):
        quals = np.random.random(num_sites)
        while np.max(quals) < 0.5:
            quals = np.random.random(num_sites)
        qual_arr.append(quals)
    return qual_arr

def get_dist_2D(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_poses(s, d):
    angles = 2*np.pi/s
    poses = []
    # pdb.set_trace()
    for i in range(0,s):
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

def agent_at_assigned_site(ag):
    if ag.assigned_site is None:
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
    return None

# def agent_at_hub_or_site(ag):
#     at_hub = agent_at_hub(ag)
#     at_site = agent_at_any_site(ag)
#     return at_hub, at_site
def get_RECRUITrs_by_site_for_world(world):
    all_agents = world.agents
    RECRUITrs = [0]*world.num_sites
    for agent in world.agents:
        if agent.state == 'RECRUIT':
            RECRUITrs[agent.assigned_site.id] += 1
    return RECRUITrs

def get_RECRUITrs_by_site(ag):
    world = ag.world
    # all_agents = world.agents
    RECRUITrs = [0]*world.num_sites
    for agent in world.agents:
        if agent.state == 'RECRUIT':
            if agent.assigned_site is not None:
                RECRUITrs[agent.assigned_site.id] += 1
            else:
                pdb.set_trace()
    return RECRUITrs

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
        if not(agent.assigned_site is None):
            sites.append(agent.assigned_site.id)
        else:
            sites.append(None)
    return poses, dirs, states, sites
