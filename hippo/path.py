class Path:
    """ Contains lists of the obs, actions and rewards for the steps of the path """

    def __init__(self):
        self.obs = []
        self.actions = [] # len(obs) must be len(actions) + 1
        self.rewards = []
        self.done = False
        self.update_no = None # policy that was used in sampling this path

    def __len__(self):
        return len(self.actions)

    def append_step(self, step: tuple):
        """ Append step at the end of the path """

        obs, action, reward = step
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def pop_step(self):
        return self.obs.pop(), self.actions.pop(), self.rewards.pop()

    @property
    def achieved_goal(self):
        return self.obs[-1]['achieved_goal'].copy()

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

