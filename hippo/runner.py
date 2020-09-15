import numpy as np
from hippo.path import Path

def flatten_obs(obs):
    obs = np.hstack([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])
    return obs

class Runner:
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        """ Run vec env with current model to sample paths """
        # Here, we init the lists that will contain the paths
        vec_obs = []
        vec_actions = []
        vec_dones = []
        vec_rewards = []
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # We already have self.obs because Runner superclass run self.obs = env.reset() on init
            # For reference: actions, values, states, neglogpacs = model.step()
            actions, _, self.states, _ = self.model.step(flatten_obs(self.obs), S=self.states, M=self.dones)
            vec_obs.append(self.obs.copy())
            vec_actions.append(actions)
            vec_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            vec_rewards.append(rewards)
        vec_obs.append(self.obs.copy())
        vec_dones.append(self.dones)

        # Collect epsiode paths by splitting at terminal state
        # For any path num_obs = num_actions + 1 
        # For non-terminating episode the final obs is required incalcualting last value for GAE
        paths = []
        for env_idx in range(self.nenv):
            path = Path()
            for t in range(self.nsteps):
                obs, action, nextobs, rewards, nextterminal =\
                     get_step(vec_obs, vec_actions, vec_rewards, vec_dones, env_idx, t)
                path.obs.append(obs)
                path.actions.append(action)
                path.rewards.append(rewards)
                if nextterminal:
                    path.obs.append(nextobs)
                    paths.append(path)
                    path.done = True
                    path = Path()
            if not nextterminal:    # add last trajectory if it is non-terminal
                path.obs.append(nextobs)
                path.done = False
                paths.append(path)
        return paths, epinfos


def get_step(vec_obs, vec_actions, vec_rewards, vec_dones, env_idx, t):
    """ helper function to return the transition tuple """

    def get_obs(t):
        if t not in range(len(vec_obs)):
            return None
        obs = {}
        for key in vec_obs[t].keys():
            obs[key] = vec_obs[t][key][env_idx:env_idx+1, :]
        return obs

    def get_action(t):
        return vec_actions[t][env_idx:env_idx+1, :]

    def get_reward(t):
        return vec_rewards[t][env_idx]

    def get_done(t):
        return vec_dones[t][env_idx]

    return get_obs(t), get_action(t), get_obs(t + 1), get_reward(t), get_done(t + 1)
