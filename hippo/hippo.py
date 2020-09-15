import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from hippo.policy import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.ppo2 import constfn, safemean
from baselines.ppo2.runner import sf01
from hippo.runner import Runner
from hippo.runner import flatten_obs

def learn(*, network, env, total_timesteps, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, buffer_age=0, exp_ratio=0, hs_strategy='none',
          **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Build policy 
    # But first add shape & dtype attributes to the env's observation space (needed for building policy network)   
    dtype = None
    size = 0
    for key in ['observation', 'achieved_goal', 'desired_goal']:
        space = ob_space.spaces[key]
        shape = space.shape
        dtype = space.dtype
        size += np.prod(shape)
        if dtype is not None:
            assert space.dtype == dtype, 'dtype not same between observation spaces'
    ob_space.shape = (size, )
    ob_space.dtype = dtype

    policy = build_policy(env, network, **network_kwargs)

    # Calculate the batch_size, nbatch is a rough approximation
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    if load_path is not None:
        model.load(load_path)
    
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=nsteps)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    # Start total timer
    tfirststart = time.perf_counter()
    her_timesteps = 0
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # Collect new trajectories here
        paths,  epinfos = runner.run()   #pylint: disable=E0632
        if eval_env is not None:
            eval_paths,  eval_epinfos = eval_runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend('epinfos')

        obs, returns, masks, actions, values, neglogpacs = batch(env, model, gamma, lam, paths)
        _nbatch = (len(obs) // nbatch_train) * nbatch_train
        her_timesteps += _nbatch

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        # Index of each element of batch_size
        # Create the indices array
        inds = np.arange(_nbatch)
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, _nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("total_timesteps_her", her_timesteps)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    return model


def batch(env, model, gamma, lam, paths):
    """ compute values and neglogpacs, GAE for each path and return batch of transitions """

    for path in paths:
        path.values = []
        path.neglogpacs = []
        path.advs = []
        n = len(path)
        for i in range(n):
            vec_obs = np.repeat(flatten_obs(path.obs[i]), env.num_envs, axis=0)
            vec_action = np.repeat(path.actions[i], env.num_envs, axis=0)
            path.values.append(model.value(vec_obs)[0])
            path.neglogpacs.append(model.act_model.neglogp_action(vec_obs, action_input=vec_action)[0])
        # GAE
        rewards = path.rewards
        values = path.values
        nextvalue = 0.0 if path.done else \
            model.value(np.repeat(flatten_obs(path.obs[-1]), env.num_envs, axis=0))[0]
        lastgaelam = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * nextvalue - values[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            path.advs.append(lastgaelam)
            nextvalue = values[t]
        path.advs.reverse()

    mb_obs, mb_actions, mb_rewards, mb_values, mb_neglogpacs, mb_advs, mb_dones = [], [], [], [], [], [], []
    done = False
    for path in paths:
        mb_obs.extend([flatten_obs(path.obs[i]) for i in range(len(path))])
        mb_actions.extend(path.actions)
        mb_rewards.extend(path.rewards)
        mb_values.extend(path.values)
        mb_neglogpacs.extend(path.neglogpacs)
        mb_advs.extend(path.advs)
        mb_dones.extend([done] + [False for _ in range(len(path) - 1)])
        done = True

    mb_obs = np.asarray(mb_obs)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_advs = np.asarray(mb_advs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    mb_returns = mb_advs + mb_values
    mb_obs, mb_actions = map(sf01, (mb_obs, mb_actions))
    
    return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs