import gym
from gym.wrappers.flatten_observation import FlattenObservation
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from hippo.hippo import learn
from hippo.hindsight import reward_fn

import numpy as np

ENV = "FetchReach-v1"

def run_env():
    env = gym.make(ENV)
    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(action)
        env.render()

def make_env():
    env = gym.make(ENV)
    env = env
    return env

if __name__ == '__main__':

    nenvs = 4
    env_fns = [make_env for _ in range(4)]
    env = VecMonitor(DummyVecEnv(env_fns))
    learn(
        network='mlp', 
        env=env, 
        total_timesteps=int(1e6),
        nsteps=2048, 
        nbatch=2*nenvs*2048, 
        log_interval=1, 
        reward_fn=reward_fn(env_fns[0]),
        buffer_capacity=2*nenvs*2048,
        hindsight = 0.5,
        )
