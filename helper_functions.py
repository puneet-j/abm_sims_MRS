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