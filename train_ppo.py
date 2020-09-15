import gym
from gym.wrappers.flatten_observation import FlattenObservation
from baselines.ppo2.ppo2 import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor

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
    env = FlattenObservation(env)
    return env

if __name__ == '__main__':

    nenvs = 4
    env_fns = [make_env for _ in range(4)]
    env = VecMonitor(SubprocVecEnv(env_fns))
    learn(network='mlp', env=env, total_timesteps=int(1e5), log_interval=1)