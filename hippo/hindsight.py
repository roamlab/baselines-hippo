from copy import copy, deepcopy
import random
from .path import Path
import itertools

def reward_fn(env_fn):
    dummy = env_fn()
    rew_fn = dummy.compute_reward
    dummy.close()
    del dummy
    return rew_fn

def apply_hindsight(path, reward_fn):

    for obs in path.obs:
        obs['desired_goal'] = path.achieved_goal
    return path

    for t in range(len(path)):
        action, obs = path.actions[t], path.obs[t+1]
        info = {'action': action, 'observation': obs['observation']}
        path.rewards[t] = compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

    return path

def split_path(path, num_pieces):
    """ Split path into equal pieces """
    T = len(path)//num_pieces
    subpaths = []
    t = 0
    subpath = Path()
    while t < len(path):
        subpath.append_step((path.obs[t], path.actions[t], path.rewards[t]))
        if (t+1) % T == 0 or t == (len(path)-1):
            subpath.obs.append(path.obs[t+1])
            subpath.done = True
            subpaths.append(path)
            subpath = Path()
        t += 1
    return subpaths

def get_subpath(path, tstart, tstop):
    
    """  
    Return a sub path from 'start' timestep to 'stop' timestep. The returned path does not 
    contain the step resulting from the action at the stop timestep. Therefore, the length of 
    the path returned is stop - start. 
    
    """

    assert tstop > tstart, 'stop timestep must be larger than start timestep'
    subpath = Path()
    for t in range(tstart, tstop):
        subpath.append_step((path.obs[t], path.actions[t], path.rewards[t]))
    subpath.obs.append(path.obs[tstop])
    return subpath

def recompute_path_rewards(path, compute_reward):
    path = deepcopy(path)
    for t in range(len(path)):
            action, obs = path.actions[t], path.obs[t+1]
            info = {'action': action, 'observation': obs['observation']}
            path.rewards[t] = compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
    return path

def update_path_desired_goal(path, desired_goal):
    path = deepcopy(path)
    for obs in path.obs:
        obs['desired_goal'] = desired_goal
    return path

def preprocess_runner_paths(paths):
    
    # The paths obtained by the runner require some pre processing before they can be used for 
    # hindsight. If a path terminates i.e corresponds to the full episode then the final obs
    # does not correspond to the terminal state but the state after resetting the environment.
    # Hence, the achieved goal in the final obs does not have any bearing with the achieved goal of 
    # the path. The final step is removed from the path if it is a terminating path.

    paths = deepcopy(paths)
    for path in paths:
        # If path terminates remove the last step
        if path.done is True:
            path.pop()
            path.done = False
    return paths

class Hindsight:

    """  Generate new paths from the paths sampled from the runner """

    def __init__(self, reward_fn):
        self.compute_reward = reward_fn # reward function required to recompute rewards
        self.data_multiplier = None # indicates increase in data after hindsight path augumentation 
        # ex. if path augumentation results in double the amount of transitions this implies a 
        # data_multiplier of 2 

    def make_hindsight_path(self, path):
        path = update_path_desired_goal(path, path.get_final_achieved_goal())
        path = recompute_path_rewards(path, self.compute_reward)
        path.done = True
        return path
        
    def get_hindsight_paths(self, paths):
        raise NotImplementedError

class FinalAchievedGoal(Hindsight):

    """ Use the achieved goal at the end of the path as the desired goal for 
    generating the new augumented path """

    def __init__(self, reward_fn):
        super().__init__(reward_fn)
        self.data_multiplier = 2 

    def get_hindsight_paths(self, paths):
        paths = preprocess_runner_paths(paths)
        hindsight_paths = []
        for path in paths:
            path = self.make_hindsight_path(path)
            hindsight_paths.append(path)
        return hindsight_paths


class SplitEqual(Hindsight):

    """  In this strategy, a path is split into a number of equal sized pieces and 
    final achieved goal for each of the subpaths is used as the desired goal """
    
    def __init__(self, reward_fn, num_pieces):
        super().__init__(reward_fn)
        self.data_multiplier = 2
        self.num_pieces = num_pieces
    
    def get_hindsight_paths(self, paths):
        hindsight_paths = []
        for path in paths:
            if len(path) < self.num_pieces:
                path = self.make_hindsight_path(path)
                hindsight_paths.append(path)
            else:
                subpaths = split_path(path, self.num_pieces)
                for path in subpaths:
                    path = self.make_hindsight_path(path)
                    hindsight_paths.append(path)
        return hindsight_paths

class RandomSubpath(Hindsight):

    def __init__(self, reward_fn, data_multiplier=2):
        self.data_multiplier = data_multiplier
        super().__init__(reward_fn)

    def get_hindsight_paths(self, paths):
        paths = preprocess_runner_paths(paths)
        nbatch = 0
        npaths = 0
        for path in paths:
            nbatch += len(path)
            npaths += 1
    
        nbatch_hindsight = 0
        hindsight_paths = []
        random.shuffle(paths)
        path_no = 0
        nbatch_hindsight_target = int(nbatch * self.data_multiplier) 
        while nbatch_hindsight < nbatch_hindsight_target:
            # Get as subpath 
            path = paths[path_no]
            tstart, tstop = random.sample(range(len(path)), 2)
            if tstart > tstop: tstart, tstop = tstop, tstart
            subpath = get_subpath(path, tstart, tstop)
            # Chop the end if its too big
            if nbatch_hindsight + len(subpath) > nbatch_hindsight_target:
                subpath = get_subpath(subpath, 0, nbatch_hindsight_target-nbatch_hindsight)
            subpath = self.make_hindsight_path(subpath)
            hindsight_paths.append(subpath)
            nbatch_hindsight += len(subpath)
            path_no += 1
            path_no %= npaths
        return hindsight_paths
